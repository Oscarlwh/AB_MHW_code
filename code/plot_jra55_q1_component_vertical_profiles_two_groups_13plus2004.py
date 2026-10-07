#!/usr/bin/env python3
"""Plot JRA-55 Q1 component box-mean vertical profiles for two event groups."""

from __future__ import annotations

import argparse
import gc
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, num2date

from plot_jra55_era5_box_vertical_profiles import (
    BOX,
    LEVELS,
    LAGS,
    OUT_ROOT,
    PEAK_DATES,
    box_weighted_mean,
    jra_file,
    setup_pressure_axis,
    symmetric_xlim,
    validate_peak_dates,
)


DIA_ROOT = Path("/path/to/data/dia")
TAG = "jra55_components_two_groups_13plus2004"

COMPONENTS = {
    "cnvhr": {
        "data_var": "CNVHR_GDS0_ISBL_ave6h",
        "label": "Convective",
        "color": "#d62728",
        "marker": "o",
    },
    "lrghr": {
        "data_var": "LRGHR_GDS0_ISBL_ave6h",
        "label": "Large-scale condensation",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "lwhr": {
        "data_var": "LWHR_GDS0_ISBL_ave6h",
        "label": "Longwave radiation",
        "color": "#1f77b4",
        "marker": "^",
    },
    "swhr": {
        "data_var": "SWHR_GDS0_ISBL_ave6h",
        "label": "Shortwave radiation",
        "color": "#2ca02c",
        "marker": "v",
    },
    "vdfhr": {
        "data_var": "VDFHR_GDS0_ISBL_ave6h",
        "label": "Vertical diffusion",
        "color": "#9467bd",
        "marker": "D",
    },
}

PANEL_SPECS = [
    {
        "panel": "A",
        "key": "mean_13_events",
        "title": "13-event mean",
        "event_indices": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13],
    },
    {
        "panel": "B",
        "key": "event_2004_12_24",
        "title": "2004-12-24",
        "event_indices": [6],
    },
]

YEARS = range(1981, 2024)
CLIM_YEARS = range(1981, 2021)
MONTHS = range(1, 13)
TIME_RANGE_RE = re.compile(r"\.(\d{10})_(\d{10})\.")
FILE_INDEX: dict[tuple[int, str, int, int], list[Path]] | None = None
BOX_INDEX_CACHE: dict[int, tuple[slice, slice, np.ndarray, int]] = {}


def parse_file_time_range(fname: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    match = TIME_RANGE_RE.search(fname)
    if match is None:
        return None, None
    start = pd.to_datetime(match.group(1), format="%Y%m%d%H")
    end = pd.to_datetime(match.group(2), format="%Y%m%d%H")
    return start, end


def build_file_index() -> dict[tuple[int, str, int, int], list[Path]]:
    index: dict[tuple[int, str, int, int], list[Path]] = {}
    for level in LEVELS:
        for component in COMPONENTS:
            var_dir = DIA_ROOT / str(level) / component
            for f in sorted(var_dir.glob("*.nc")):
                if f.name.startswith("._"):
                    continue
                match = TIME_RANGE_RE.search(f.name)
                if match is None:
                    continue
                start = match.group(1)
                year = int(start[:4])
                month = int(start[4:6])
                index.setdefault((level, component, year, month), []).append(f)
    return index


def find_month_file(level: int, component: str, year: int, month: int) -> Path:
    global FILE_INDEX
    if FILE_INDEX is None:
        FILE_INDEX = build_file_index()

    matched = FILE_INDEX.get((level, component, year, month), [])
    if len(matched) != 1:
        raise FileNotFoundError(
            f"Expected 1 file for level={level}, component={component}, "
            f"{year}-{month:02d}; found {len(matched)}"
        )
    return matched[0]


def calc_doy_nl(ts: pd.Timestamp) -> int:
    if ts.month == 2 and ts.day == 29:
        raise ValueError("Feb 29 is excluded from no-leap climatology.")
    doy = ts.dayofyear
    if ts.is_leap_year and doy > 60:
        doy -= 1
    return int(doy)


def target_table() -> pd.DataFrame:
    records = []
    for event, peak in enumerate(PEAK_DATES):
        peak_ts = pd.Timestamp(peak)
        for lag in LAGS:
            target = peak_ts + pd.Timedelta(days=lag)
            records.append(
                {
                    "event": event,
                    "peak_date": peak_ts,
                    "lag": lag,
                    "target_date": target,
                    "year": target.year,
                    "doy_nl": calc_doy_nl(target),
                }
            )
    return pd.DataFrame(records)


def expected_days(year: int) -> int:
    return 366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365


def coverage_check() -> pd.DataFrame:
    records = []
    for level in LEVELS:
        for component in COMPONENTS:
            for year in YEARS:
                for month in MONTHS:
                    try:
                        f = find_month_file(level, component, year, month)
                        status = "OK"
                    except FileNotFoundError:
                        f = None
                        status = "BAD"
                    records.append(
                        {
                            "level_hPa": level,
                            "component": component,
                            "year": year,
                            "month": month,
                            "status": status,
                            "file": "" if f is None else str(f),
                        }
                    )
    df = pd.DataFrame(records)
    bad = df[df["status"] != "OK"]
    if not bad.empty:
        raise FileNotFoundError("Missing or duplicated JRA-55 component files:\n" + bad.to_string(index=False))
    return df


def box_index_for_file(level: int, f: Path) -> tuple[slice, slice, np.ndarray, int]:
    if level in BOX_INDEX_CACHE:
        return BOX_INDEX_CACHE[level]

    with Dataset(f) as ds:
        lat = np.asarray(ds.variables["g0_lat_1"][:], dtype="float64")
        lon = np.asarray(ds.variables["g0_lon_2"][:], dtype="float64")

    lat_idx = np.where((lat >= BOX["lat_min"]) & (lat <= BOX["lat_max"]))[0]
    lon_idx = np.where((lon >= BOX["lon_min"]) & (lon <= BOX["lon_max"]))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(f"Empty box selection for level={level}: {BOX}")
    if not np.array_equal(lat_idx, np.arange(lat_idx.min(), lat_idx.max() + 1)):
        raise ValueError(f"Non-contiguous latitude box indices for level={level}")
    if not np.array_equal(lon_idx, np.arange(lon_idx.min(), lon_idx.max() + 1)):
        raise ValueError(f"Non-contiguous longitude box indices for level={level}")

    lat_slice = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_slice = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
    weights = np.cos(np.deg2rad(lat[lat_slice])).astype("float64")
    nlon = int(lon_idx.size)
    BOX_INDEX_CACHE[level] = (lat_slice, lon_slice, weights, nlon)
    return BOX_INDEX_CACHE[level]


def read_component_month_box_mean(level: int, component: str, year: int, month: int) -> pd.Series:
    info = COMPONENTS[component]
    f = find_month_file(level, component, year, month)
    start, end = parse_file_time_range(f.name)
    if start is None or end is None:
        raise ValueError(f"Cannot parse time range from {f.name}")

    data_var = info["data_var"]
    lat_slice, lon_slice, weights, _ = box_index_for_file(level, f)

    with Dataset(f) as ds:
        if data_var not in ds.variables:
            raise KeyError(f"{data_var} not found in {f}")
        var = ds.variables[data_var]
        units = getattr(var, "units", "").lower().strip()
        if units not in {"k/day", "k day-1"}:
            raise ValueError(f"Unexpected units for {f}: {getattr(var, 'units', '')}")
        raw = var[:, lat_slice, lon_slice]
        arr = np.ma.asarray(raw).filled(np.nan).astype("float64")
        time_var = ds.variables["initial_time0_hours"]
        decoded_time = num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )

    w3 = weights[None, :, None]
    valid = np.isfinite(arr)
    numerator = np.nansum(arr * w3, axis=(1, 2))
    denominator = np.sum(valid * w3, axis=(1, 2))
    box_values = numerator / denominator

    times = pd.to_datetime(decoded_time)
    if len(times) != box_values.size:
        raise ValueError(f"{f.name}: expected {box_values.size} timestamps, got {len(times)}")
    daily = pd.Series(box_values, index=times, name=component).resample("1D").mean()
    return daily


def build_daily_box_series(level: int, component: str) -> pd.Series:
    monthly = []
    for year in YEARS:
        for month in MONTHS:
            monthly.append(read_component_month_box_mean(level, component, year, month))

    series = pd.concat(monthly).sort_index()
    expected_total = sum(expected_days(y) for y in YEARS)
    if len(series) != expected_total:
        raise ValueError(f"{level} hPa {component}: expected {expected_total} days, got {len(series)}")
    if series.index.has_duplicates:
        raise ValueError(f"{level} hPa {component}: duplicate daily timestamps detected")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{level} hPa {component}: daily box series contains NaN/Inf")

    del monthly
    gc.collect()
    return series


def detrend_no_mean(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype="float64")
    t = np.arange(values.size, dtype="float64")
    t_mean = float(t.mean())
    v_mean = float(values.mean())
    slope = float(np.sum((t - t_mean) * (values - v_mean)) / np.sum((t - t_mean) ** 2))
    intercept = v_mean - slope * t_mean
    detrended = values - (slope * t + intercept)
    out = pd.Series(detrended, index=series.index, name=series.name)
    out.attrs.update({"linear_slope": slope, "linear_intercept": intercept})
    return out


def build_no_leap_climatology(detrended: pd.Series) -> pd.Series:
    time_index = pd.DatetimeIndex(detrended.index)
    is_feb29 = (time_index.month == 2) & (time_index.day == 29)
    doy_nl = np.array([calc_doy_nl(ts) for ts in time_index[~is_feb29]], dtype=np.int16)
    clim_input = pd.DataFrame(
        {
            "value": detrended.loc[time_index[~is_feb29]].to_numpy(dtype="float64"),
            "doy_nl": doy_nl,
        },
        index=time_index[~is_feb29],
    )
    clim_period = clim_input.loc["1981-01-01":"2020-12-31"]
    clim = clim_period.groupby("doy_nl")["value"].mean()
    expected = np.arange(1, 366, dtype=np.int16)
    if not np.array_equal(clim.index.to_numpy(dtype=np.int16), expected):
        raise ValueError("No-leap climatology does not contain exactly doy_nl 1..365")
    return clim


def component_profile(level: int, component: str, targets: pd.DataFrame) -> dict[str, float]:
    daily = build_daily_box_series(level, component)
    detrended = detrend_no_mean(daily)
    clim = build_no_leap_climatology(detrended)

    target_values = []
    for _, row in targets.iterrows():
        target = pd.Timestamp(row["target_date"])
        value = float(detrended.loc[target])
        clim_value = float(clim.loc[int(row["doy_nl"])])
        target_values.append(
            {
                "event": int(row["event"]),
                "lag": int(row["lag"]),
                "anomaly": value - clim_value,
            }
        )

    event_lag = pd.DataFrame(target_values)
    profiles = {}
    for spec in PANEL_SPECS:
        sub = event_lag[event_lag["event"].isin(spec["event_indices"])]
        profiles[spec["key"]] = float(sub["anomaly"].mean())

    return profiles


def read_existing_total_q1_profile(level: int) -> dict[str, float]:
    with xr.open_dataset(jra_file(level)) as ds:
        validate_peak_dates(ds, f"JRA level={level}")
        da = ds["Q1_anomaly_event_lag"].sel(lag=LAGS)
        box = box_weighted_mean(da, "g0_lat_1", "g0_lon_2").load()

    profiles = {}
    for spec in PANEL_SPECS:
        profiles[spec["key"]] = float(box.isel(event=spec["event_indices"]).mean(dim=("event", "lag"), skipna=True))
    return profiles


def component_profile_task(args: tuple[int, str, pd.DataFrame]) -> tuple[int, str, dict[str, float]]:
    level, component, targets = args
    prof = component_profile(level, component, targets)
    return level, component, prof


def build_profiles(max_workers: int = 4) -> tuple[pd.DataFrame, xr.Dataset, pd.DataFrame]:
    coverage = coverage_check()
    targets = target_table()

    component_data = {
        spec["key"]: {component: [] for component in COMPONENTS}
        for spec in PANEL_SPECS
    }
    total_existing = {spec["key"]: [] for spec in PANEL_SPECS}

    task_args = [(level, component, targets) for level in LEVELS for component in COMPONENTS]
    result_lookup: dict[tuple[int, str], dict[str, float]] = {}
    print(f"[COMPONENT TASKS] n={len(task_args)} max_workers={max_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(component_profile_task, args) for args in task_args]
        for future in as_completed(futures):
            level, component, prof = future.result()
            result_lookup[(level, component)] = prof
            print(f"[DONE] level={level} hPa component={component}", flush=True)

    for level in LEVELS:
        for component in COMPONENTS:
            prof = result_lookup[(level, component)]
            for spec in PANEL_SPECS:
                component_data[spec["key"]][component].append(prof[spec["key"]])

    for level in LEVELS:
        print(f"[TOTAL Q1] level={level} hPa", flush=True)
        total_prof = read_existing_total_q1_profile(level)
        for spec in PANEL_SPECS:
            total_existing[spec["key"]].append(total_prof[spec["key"]])

    panel_keys = [spec["key"] for spec in PANEL_SPECS]
    component_keys = list(COMPONENTS)
    component_array = np.array(
        [
            [component_data[panel][component] for component in component_keys]
            for panel in panel_keys
        ],
        dtype=np.float32,
    )
    component_sum = component_array.sum(axis=1)
    total_array = np.array([total_existing[panel] for panel in panel_keys], dtype=np.float32)
    closure_diff = component_sum - total_array

    ds = xr.Dataset(
        {
            "component_anomaly": (("panel", "component", "level_hPa"), component_array),
            "Q1_sum_components": (("panel", "level_hPa"), component_sum),
            "Q1_existing_total": (("panel", "level_hPa"), total_array),
            "closure_diff": (("panel", "level_hPa"), closure_diff),
        },
        coords={
            "panel": panel_keys,
            "component": component_keys,
            "level_hPa": np.array(LEVELS, dtype=np.int32),
        },
        attrs={
            "title": "JRA-55 Q1 component box-mean vertical profiles for two event groups",
            "units": "K day-1",
            "box": f"{BOX['lon_min']}-{BOX['lon_max']}E, {BOX['lat_min']}-{BOX['lat_max']}N",
            "lag_average": "lag0-7 mean",
            "grouping": "13-event mean excludes only 2004-12-24; 2004-12-24 is plotted separately",
            "detrending": "component box-mean daily series detrended over 1981-2023 without adding temporal mean back",
            "climatology": "1981-2020 no-leap daily climatology of detrended component box-mean series",
        },
    )

    records = []
    for pi, spec in enumerate(PANEL_SPECS):
        for li, level in enumerate(LEVELS):
            common = {
                "panel": spec["panel"],
                "key": spec["key"],
                "title": spec["title"],
                "event_indices": ",".join(str(i) for i in spec["event_indices"]),
                "event_dates": ",".join(PEAK_DATES[i] for i in spec["event_indices"]),
                "n_events": len(spec["event_indices"]),
                "level_hPa": level,
            }
            for ci, component in enumerate(component_keys):
                records.append(
                    {
                        **common,
                        "series": component,
                        "label": COMPONENTS[component]["label"],
                        "value_K_day": float(component_array[pi, ci, li]),
                    }
                )
            records.extend(
                [
                    {**common, "series": "Q1_sum_components", "label": "Sum of five components", "value_K_day": float(component_sum[pi, li])},
                    {**common, "series": "Q1_existing_total", "label": "Existing total Q1", "value_K_day": float(total_array[pi, li])},
                    {**common, "series": "closure_diff", "label": "Sum - existing total Q1", "value_K_day": float(closure_diff[pi, li])},
                ]
            )

    return pd.DataFrame(records), ds, coverage


def plot_profiles(ds: xr.Dataset, out_root: Path) -> tuple[Path, Path]:
    levels = ds["level_hPa"].values
    component_values = ds["component_anomaly"].values
    total_values = ds["Q1_existing_total"].values
    xlim = symmetric_xlim(component_values, total_values, pad=0.15)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5), sharey=True)
    for i, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        panel = spec["key"]
        for component, info in COMPONENTS.items():
            ax.plot(
                ds["component_anomaly"].sel(panel=panel, component=component),
                levels,
                color=info["color"],
                linewidth=1.5,
                marker=info["marker"],
                markersize=3.5,
                label=info["label"] if i == 0 else None,
            )
        ax.plot(
            ds["Q1_existing_total"].sel(panel=panel),
            levels,
            color="#111111",
            linewidth=2.3,
            marker="o",
            markersize=4.2,
            label="Total Q1" if i == 0 else None,
        )
        setup_pressure_axis(ax)
        ax.set_xlim(*xlim)
        ax.set_title(f"({spec['panel']}) {spec['title']}", loc="left", fontsize=11)
        ax.set_xlabel("Heating anomaly (K day$^{-1}$)")
        if i > 0:
            ax.set_ylabel("")

    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.5)
    fig.suptitle("JRA-55 Q1 component contributions, lag0-7 mean", y=0.97, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 0.88, 0.94])
    main_png = out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}.png"
    main_pdf = out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}.pdf"
    fig.savefig(main_png, dpi=300, bbox_inches="tight")
    fig.savefig(main_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.4), sharey=True)
    for i, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        panel = spec["key"]
        ax.plot(
            ds["closure_diff"].sel(panel=panel),
            levels,
            color="#333333",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
        )
        setup_pressure_axis(ax)
        ax.set_xlim(*xlim)
        ax.set_title(f"({spec['panel']}) {spec['title']}", loc="left", fontsize=11)
        ax.set_xlabel("Sum components - total Q1 (K day$^{-1}$)")
        if i > 0:
            ax.set_ylabel("")

    fig.suptitle("JRA-55 Q1 component closure check", y=0.965, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    closure_png = out_root / f"JRA55_Q1_component_closure_{TAG}.png"
    closure_pdf = out_root / f"JRA55_Q1_component_closure_{TAG}.pdf"
    fig.savefig(closure_png, dpi=300, bbox_inches="tight")
    fig.savefig(closure_pdf, bbox_inches="tight")
    plt.close(fig)
    return main_png, closure_png


def validate_outputs(df: pd.DataFrame, ds: xr.Dataset) -> None:
    expected_rows = len(PANEL_SPECS) * len(LEVELS) * (len(COMPONENTS) + 3)
    if len(df) != expected_rows:
        raise ValueError(f"Unexpected CSV row count: {len(df)} != {expected_rows}")
    expected_sizes = {"panel": 2, "component": 5, "level_hPa": 8}
    if dict(ds.sizes) != expected_sizes:
        raise ValueError(f"Unexpected dataset dimensions: {dict(ds.sizes)}")
    for name in ["component_anomaly", "Q1_sum_components", "Q1_existing_total", "closure_diff"]:
        if not np.isfinite(ds[name].values).all():
            raise ValueError(f"{name} contains NaN or Inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    df, ds, coverage = build_profiles(max_workers=args.max_workers)
    validate_outputs(df, ds)

    csv_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_values.csv"
    nc_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_values.nc"
    coverage_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_coverage.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    ds.to_netcdf(nc_file)
    coverage.to_csv(coverage_file, index=False, encoding="utf-8-sig")
    main_png, closure_png = plot_profiles(ds, args.out_root)

    max_closure = float(np.nanmax(np.abs(ds["closure_diff"].values)))
    print(f"[CSV] {csv_file}")
    print(f"[NC] {nc_file}")
    print(f"[COVERAGE] {coverage_file}")
    print(f"[FIG] {main_png}")
    print(f"[FIG] {closure_png}")
    print(f"[MAX_ABS_CLOSURE_DIFF] {max_closure:.6g} K/day")
    print("Done.")


if __name__ == "__main__":
    main()
