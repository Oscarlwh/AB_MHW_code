#!/usr/bin/env python3
"""Plot box-mean vertical profiles comparing JRA-55 Q1 and ERA5 residual Qd."""

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


JRA_ROOT = Path("/path/to/data/非绝热加热异常_JRA-55/JRA55_q1_peak_composite_inputs")
ERA5_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_qd_peak_lag0_7")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_jra55_box_vertical_profiles")

LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]
LAGS = list(range(0, 8))
BOX = {"lon_min": 200.0, "lon_max": 225.0, "lat_min": 40.0, "lat_max": 50.0}
ERA5_TO_K_DAY = 86400.0

PEAK_DATES = [
    "1985-11-29",
    "1986-11-19",
    "1989-11-18",
    "1989-12-24",
    "1991-11-10",
    "1993-11-26",
    "2004-12-24",
    "2015-11-06",
    "2015-11-30",
    "2018-11-19",
    "2019-11-10",
    "2020-11-13",
    "2023-11-03",
    "2023-12-14",
]

PANEL_SPECS = [
    {
        "panel": "A",
        "key": "mean_12_events",
        "title": "12-event mean",
        "event_indices": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13],
    },
    {
        "panel": "B",
        "key": "event_2004_12_24",
        "title": "2004-12-24",
        "event_indices": [6],
    },
    {
        "panel": "C",
        "key": "event_2020_11_13",
        "title": "2020-11-13",
        "event_indices": [11],
    },
]


def jra_file(level: int) -> Path:
    f = JRA_ROOT / str(level) / f"JRA55_q1_{level}hPa_peak_lag0_7_anomaly_detrended_no_mean.nc"
    if f.exists():
        return f
    if level == 925:
        old = JRA_ROOT / "JRA55_q1_925hPa_peak_lag0_7_anomaly_detrended_no_mean.nc"
        if old.exists():
            return old
    raise FileNotFoundError(f)


def era5_file(level: int) -> Path:
    f = ERA5_ROOT / str(level) / f"ERA5_Qd_{level}hPa_peak_lag0_7_anomaly_1981_2020clim.nc"
    if not f.exists():
        raise FileNotFoundError(f)
    return f


def sorted_lat_box(da: xr.DataArray, lat_name: str) -> xr.DataArray:
    if float(da[lat_name].values[0]) > float(da[lat_name].values[-1]):
        da = da.sortby(lat_name)
    return da.sel({lat_name: slice(BOX["lat_min"], BOX["lat_max"])})


def box_weighted_mean(da: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    da = sorted_lat_box(da, lat_name)
    da = da.sel({lon_name: slice(BOX["lon_min"], BOX["lon_max"])})
    if da.sizes.get(lat_name, 0) == 0 or da.sizes.get(lon_name, 0) == 0:
        raise ValueError(f"Empty box selection for {da.name}: {BOX}")
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=(lat_name, lon_name), skipna=True)


def validate_peak_dates(ds: xr.Dataset, source: str) -> list[str]:
    existing = pd.to_datetime(ds["peak_date"].values).strftime("%Y-%m-%d").tolist()
    if existing != PEAK_DATES:
        raise ValueError(f"{source} peak_date order mismatch:\nexpected={PEAK_DATES}\nactual={existing}")
    return existing


def read_jra_level_profile(level: int, spec: dict) -> float:
    with xr.open_dataset(jra_file(level)) as ds:
        validate_peak_dates(ds, f"JRA level={level}")
        da = ds["Q1_anomaly_event_lag"].isel(event=spec["event_indices"]).sel(lag=LAGS)
        box = box_weighted_mean(da, "g0_lat_1", "g0_lon_2")
        return float(box.mean(dim=("event", "lag"), skipna=True))


def read_era5_level_profile(level: int, spec: dict) -> float:
    with xr.open_dataset(era5_file(level)) as ds:
        validate_peak_dates(ds, f"ERA5 level={level}")
        da = ds["Qd_anomaly_event_lag"].isel(event=spec["event_indices"]).sel(lag=LAGS)
        box = box_weighted_mean(da, "lat", "lon")
        return float(box.mean(dim=("event", "lag"), skipna=True) * ERA5_TO_K_DAY)


def build_profiles() -> tuple[pd.DataFrame, xr.Dataset]:
    records = []
    data = {}
    panel_keys = []

    for spec in PANEL_SPECS:
        jra_values = []
        era5_values = []
        for level in LEVELS:
            jra = read_jra_level_profile(level, spec)
            era5 = read_era5_level_profile(level, spec)
            diff = era5 - jra
            jra_values.append(jra)
            era5_values.append(era5)
            records.extend(
                [
                    {
                        "panel": spec["panel"],
                        "key": spec["key"],
                        "title": spec["title"],
                        "event_indices": ",".join(str(i) for i in spec["event_indices"]),
                        "event_dates": ",".join(PEAK_DATES[i] for i in spec["event_indices"]),
                        "level_hPa": level,
                        "algorithm": "JRA55_Q1",
                        "value_K_day": jra,
                    },
                    {
                        "panel": spec["panel"],
                        "key": spec["key"],
                        "title": spec["title"],
                        "event_indices": ",".join(str(i) for i in spec["event_indices"]),
                        "event_dates": ",".join(PEAK_DATES[i] for i in spec["event_indices"]),
                        "level_hPa": level,
                        "algorithm": "ERA5_Qd",
                        "value_K_day": era5,
                    },
                    {
                        "panel": spec["panel"],
                        "key": spec["key"],
                        "title": spec["title"],
                        "event_indices": ",".join(str(i) for i in spec["event_indices"]),
                        "event_dates": ",".join(PEAK_DATES[i] for i in spec["event_indices"]),
                        "level_hPa": level,
                        "algorithm": "ERA5_minus_JRA55",
                        "value_K_day": diff,
                    },
                ]
            )
        key = spec["key"]
        panel_keys.append(key)
        data.setdefault("JRA55_Q1", []).append(jra_values)
        data.setdefault("ERA5_Qd", []).append(era5_values)
        data.setdefault("ERA5_minus_JRA55", []).append((np.array(era5_values) - np.array(jra_values)).tolist())

    coords = {
        "panel": panel_keys,
        "level_hPa": np.array(LEVELS, dtype=np.int32),
    }
    ds = xr.Dataset(
        {
            name: (("panel", "level_hPa"), np.array(values, dtype=np.float32))
            for name, values in data.items()
        },
        coords=coords,
        attrs={
            "title": "Box-mean vertical profiles comparing JRA-55 Q1 and ERA5 residual Qd",
            "units": "K day-1",
            "box": f"{BOX['lon_min']}-{BOX['lon_max']}E, {BOX['lat_min']}-{BOX['lat_max']}N",
            "lag_average": "lag0-7 mean",
            "jra_source": str(JRA_ROOT),
            "era5_source": str(ERA5_ROOT),
            "era5_conversion": "K s-1 multiplied by 86400 to K day-1",
        },
    )
    return pd.DataFrame(records), ds


def symmetric_xlim(*arrays: np.ndarray, pad: float = 0.15, floor: float = 0.1) -> tuple[float, float]:
    max_abs = max(float(np.nanmax(np.abs(arr))) for arr in arrays)
    max_abs = max(max_abs * (1.0 + pad), floor)
    return -max_abs, max_abs


def setup_pressure_axis(ax) -> None:
    ax.set_ylim(1050, 225)
    ax.set_yticks(LEVELS)
    ax.set_ylabel("Pressure (hPa)")
    ax.grid(True, axis="both", linestyle=":", linewidth=0.6, color="0.75")
    ax.axvline(0.0, color="0.45", linewidth=0.8)


def plot_profiles(ds: xr.Dataset, out_root: Path) -> tuple[Path, Path]:
    levels = ds["level_hPa"].values
    jra = ds["JRA55_Q1"].values
    era5 = ds["ERA5_Qd"].values
    diff = ds["ERA5_minus_JRA55"].values
    xlim_main = symmetric_xlim(jra, era5)
    xlim_diff = symmetric_xlim(diff)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.6), sharey=True)
    for i, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        panel = spec["key"]
        ax.plot(
            ds["JRA55_Q1"].sel(panel=panel),
            levels,
            color="#1f77b4",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label="JRA-55 Q1" if i == 0 else None,
        )
        ax.plot(
            ds["ERA5_Qd"].sel(panel=panel),
            levels,
            color="#d62728",
            linewidth=2.0,
            marker="s",
            markersize=4.2,
            label="ERA5 Qd" if i == 0 else None,
        )
        setup_pressure_axis(ax)
        ax.set_xlim(*xlim_main)
        ax.set_title(f"({spec['panel']}) {spec['title']}", loc="left", fontsize=11)
        ax.set_xlabel("Heating anomaly (K day$^{-1}$)")
        if i > 0:
            ax.set_ylabel("")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("Box-mean vertical profiles, lag0-7 mean", y=0.965, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = out_root / "JRA55_ERA5_Qd_box_vertical_profiles_3panels.png"
    out_pdf = out_root / "JRA55_ERA5_Qd_box_vertical_profiles_3panels.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.6), sharey=True)
    for i, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        panel = spec["key"]
        ax.plot(
            ds["ERA5_minus_JRA55"].sel(panel=panel),
            levels,
            color="#333333",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
        )
        setup_pressure_axis(ax)
        ax.set_xlim(*xlim_diff)
        ax.set_title(f"({spec['panel']}) {spec['title']}", loc="left", fontsize=11)
        ax.set_xlabel("ERA5 Qd - JRA-55 Q1 (K day$^{-1}$)")
        if i > 0:
            ax.set_ylabel("")
    fig.suptitle("Algorithm difference in box-mean vertical profiles", y=0.965, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    diff_png = out_root / "ERA5_minus_JRA55_box_vertical_profiles_3panels.png"
    diff_pdf = out_root / "ERA5_minus_JRA55_box_vertical_profiles_3panels.pdf"
    fig.savefig(diff_png, dpi=300, bbox_inches="tight")
    fig.savefig(diff_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, diff_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    df, ds = build_profiles()
    csv_file = args.out_root / "JRA55_ERA5_Qd_box_vertical_profiles_values.csv"
    nc_file = args.out_root / "JRA55_ERA5_Qd_box_vertical_profiles_values.nc"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    ds.to_netcdf(nc_file)
    main_png, diff_png = plot_profiles(ds, args.out_root)

    print(f"[CSV] {csv_file}")
    print(f"[NC] {nc_file}")
    print(f"[FIG] {main_png}")
    print(f"[FIG] {diff_png}")
    print("Done.")


if __name__ == "__main__":
    main()
