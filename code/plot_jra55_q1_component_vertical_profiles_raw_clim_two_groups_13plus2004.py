#!/usr/bin/env python3
"""Fast JRA-55 Q1 component profiles using raw 1981-2020 daily climatology."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plot_jra55_era5_box_vertical_profiles import BOX, LEVELS, LAGS, OUT_ROOT, PEAK_DATES, setup_pressure_axis, symmetric_xlim
from plot_jra55_q1_component_vertical_profiles_two_groups_13plus2004 import (
    CLIM_YEARS,
    COMPONENTS,
    DIA_ROOT,
    PANEL_SPECS,
    calc_doy_nl,
    coverage_check,
    find_month_file,
    parse_file_time_range,
    read_component_month_box_mean,
)


TAG = "jra55_components_raw_clim_two_groups_13plus2004"


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
                    "month": target.month,
                    "doy_nl": calc_doy_nl(target),
                }
            )
    return pd.DataFrame(records)


def month_series(level: int, component: str, year: int, month: int, cache: dict) -> pd.Series:
    key = (level, component, year, month)
    if key not in cache:
        cache[key] = read_component_month_box_mean(level, component, year, month)
    return cache[key]


def component_profile(level: int, component: str, targets: pd.DataFrame) -> dict[str, float]:
    cache: dict[tuple[int, str, int, int], pd.Series] = {}
    needed_months = sorted(targets["month"].unique())
    clim_parts = []
    for year in CLIM_YEARS:
        for month in needed_months:
            clim_parts.append(month_series(level, component, year, int(month), cache))
    clim_series = pd.concat(clim_parts).sort_index()
    clim_doy = pd.Series(
        clim_series.to_numpy(dtype="float64"),
        index=[calc_doy_nl(ts) for ts in clim_series.index],
    ).groupby(level=0).mean()

    target_records = []
    for _, row in targets.iterrows():
        target = pd.Timestamp(row["target_date"])
        series = month_series(level, component, target.year, target.month, cache)
        value = float(series.loc[target])
        clim_value = float(clim_doy.loc[int(row["doy_nl"])])
        target_records.append({"event": int(row["event"]), "lag": int(row["lag"]), "anomaly": value - clim_value})

    event_lag = pd.DataFrame(target_records)
    profiles = {}
    for spec in PANEL_SPECS:
        sub = event_lag[event_lag["event"].isin(spec["event_indices"])]
        profiles[spec["key"]] = float(sub["anomaly"].mean())
    return profiles


def build_profiles() -> tuple[pd.DataFrame, xr.Dataset, pd.DataFrame]:
    coverage = coverage_check()
    targets = target_table()

    panel_keys = [spec["key"] for spec in PANEL_SPECS]
    component_keys = list(COMPONENTS)
    component_array = np.full((len(panel_keys), len(component_keys), len(LEVELS)), np.nan, dtype=np.float32)

    for li, level in enumerate(LEVELS):
        print(f"[LEVEL] {level} hPa", flush=True)
        for ci, component in enumerate(component_keys):
            print(f"  [COMP] {component}", flush=True)
            profiles = component_profile(level, component, targets)
            for pi, spec in enumerate(PANEL_SPECS):
                component_array[pi, ci, li] = profiles[spec["key"]]

    component_sum = component_array.sum(axis=1)

    ds = xr.Dataset(
        {
            "component_anomaly": (("panel", "component", "level_hPa"), component_array),
            "Q1_sum_components": (("panel", "level_hPa"), component_sum),
        },
        coords={
            "panel": panel_keys,
            "component": component_keys,
            "level_hPa": np.array(LEVELS, dtype=np.int32),
        },
        attrs={
            "title": "JRA-55 Q1 component box-mean vertical profiles using raw 1981-2020 daily climatology",
            "units": "K day-1",
            "box": f"{BOX['lon_min']}-{BOX['lon_max']}E, {BOX['lat_min']}-{BOX['lat_max']}N",
            "lag_average": "lag0-7 mean",
            "grouping": "13-event mean excludes only 2004-12-24; 2004-12-24 is plotted separately",
            "method": "raw component daily anomaly relative to 1981-2020 no-leap daily climatology; no detrending",
            "source": str(DIA_ROOT),
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
            records.append(
                {
                    **common,
                    "series": "Q1_sum_components",
                    "label": "Sum of five components",
                    "value_K_day": float(component_sum[pi, li]),
                }
            )
    return pd.DataFrame(records), ds, coverage


def plot_profiles(ds: xr.Dataset, out_root: Path) -> Path:
    levels = ds["level_hPa"].values
    xlim = symmetric_xlim(ds["component_anomaly"].values, ds["Q1_sum_components"].values, pad=0.15)

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
            ds["Q1_sum_components"].sel(panel=panel),
            levels,
            color="#111111",
            linewidth=2.3,
            marker="o",
            markersize=4.2,
            label="Sum Q1" if i == 0 else None,
        )
        setup_pressure_axis(ax)
        ax.set_xlim(*xlim)
        ax.set_title(f"({spec['panel']}) {spec['title']}", loc="left", fontsize=11)
        ax.set_xlabel("Heating anomaly (K day$^{-1}$)")
        if i > 0:
            ax.set_ylabel("")

    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.5)
    fig.suptitle("JRA-55 Q1 component contributions, raw-clim lag0-7 mean", y=0.97, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 0.88, 0.94])
    out_png = out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}.png"
    out_pdf = out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def validate_outputs(df: pd.DataFrame, ds: xr.Dataset) -> None:
    expected_rows = len(PANEL_SPECS) * len(LEVELS) * (len(COMPONENTS) + 1)
    if len(df) != expected_rows:
        raise ValueError(f"Unexpected CSV row count: {len(df)} != {expected_rows}")
    if dict(ds.sizes) != {"panel": 2, "component": 5, "level_hPa": 8}:
        raise ValueError(f"Unexpected dataset dimensions: {dict(ds.sizes)}")
    for name in ["component_anomaly", "Q1_sum_components"]:
        if not np.isfinite(ds[name].values).all():
            raise ValueError(f"{name} contains NaN or Inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    df, ds, coverage = build_profiles()
    validate_outputs(df, ds)

    csv_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_values.csv"
    nc_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_values.nc"
    coverage_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_coverage.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    ds.to_netcdf(nc_file)
    coverage.to_csv(coverage_file, index=False, encoding="utf-8-sig")
    fig = plot_profiles(ds, args.out_root)

    print(f"[CSV] {csv_file}")
    print(f"[NC] {nc_file}")
    print(f"[COVERAGE] {coverage_file}")
    print(f"[FIG] {fig}")
    print("Done.")


if __name__ == "__main__":
    main()
