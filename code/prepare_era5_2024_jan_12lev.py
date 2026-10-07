#!/usr/bin/env python3
"""Standardize downloaded ERA5 2024-01 pressure-level data to daily 1-degree 12-level files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_FILE = Path("/path/to/data/ERA5_raw/2024.1/dc44c6e45a53bbf3ec1213b0aab180b9.nc")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly_12lev")
VARS = ["t", "u", "v", "w"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]


def normalize(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in {
        "valid_time": "time",
        "pressure_level": "lev",
        "latitude": "lat",
        "longitude": "lon",
    }.items():
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    ds = ds.rename(rename)
    out = ds[VARS].sel(lev=LEVELS)
    if float(out.lat[0]) > float(out.lat[-1]):
        out = out.sortby("lat")
    out = out.assign_coords(lon=out.lon % 360).sortby("lon")
    return out.reset_coords(names=[x for x in out.coords if x not in out.dims], drop=True)


def write_check(ds: xr.Dataset, out_file: Path) -> dict[str, object]:
    return {
        "file": str(out_file),
        "time_count": int(ds.sizes["time"]),
        "level_count": int(ds.sizes["lev"]),
        "lat_count": int(ds.sizes["lat"]),
        "lon_count": int(ds.sizes["lon"]),
        "time_start": str(pd.Timestamp(ds.time.values[0]).date()),
        "time_end": str(pd.Timestamp(ds.time.values[-1]).date()),
        "nan_count": int(sum(np.isnan(ds[v].values).sum() for v in VARS)),
        "inf_count": int(sum(np.isinf(ds[v].values).sum() for v in VARS)),
    }


def build_daily(overwrite: bool) -> dict[str, object]:
    out_file = OUT_ROOT / "2024" / "202401.nc"
    if out_file.exists() and not overwrite:
        with xr.open_dataset(out_file) as ds:
            return write_check(ds, out_file)
    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)
    with xr.open_dataset(RAW_FILE) as raw:
        sub = raw[VARS].sel(pressure_level=LEVELS).isel(
            latitude=slice(None, None, 4),
            longitude=slice(None, None, 4),
        )
        times = pd.DatetimeIndex(sub.valid_time.values)
        if len(times) != 124 or sorted(set(times.hour)) != [0, 6, 12, 18]:
            raise ValueError(f"Unexpected four-times-daily coverage: n={len(times)} hours={sorted(set(times.hour))}")
        data_vars = {}
        for name in VARS:
            values = sub[name].values.reshape(31, 4, len(LEVELS), 181, 360).mean(axis=1, dtype=np.float64)
            data_vars[name] = (("time", "lev", "lat", "lon"), values.astype("float32"))
        daily = xr.Dataset(
            data_vars,
            coords={
                "time": times[::4].normalize().values,
                "lev": sub.pressure_level.values,
                "lat": sub.latitude.values,
                "lon": sub.longitude.values,
            },
        )
    daily = normalize(daily).transpose("time", "lev", "lat", "lon")
    daily.attrs.update(
        {
            "title": "ERA5 daily 1-degree 12-level pressure-level fields, 2024-01",
            "source": str(RAW_FILE),
            "levels_hPa": ",".join(str(x) for x in LEVELS),
            "daily_mean": "mean of 00,06,12,18 UTC",
        }
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in VARS}
    daily.to_netcdf(out_file, encoding=encoding)
    return write_check(daily, out_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    row = build_daily(args.overwrite)
    check = OUT_ROOT / "ERA5_202401_12lev_check.csv"
    pd.DataFrame([row]).to_csv(check, index=False, encoding="utf-8-sig")
    print(f"[NC] {row['file']}", flush=True)
    print(f"[CHECK] {check}", flush=True)


if __name__ == "__main__":
    main()
