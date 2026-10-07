#!/usr/bin/env python3
"""Build daily 1-degree, 12-level ERA5 month files needed by Qe/Fe SOR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_HIGH = Path("/path/to/data/ERA5_raw/1986_2023_10/c27a9d2b3d3f7e29d2673e2792615082.nc")
LOW8_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly")
FILLED17_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly_17lev")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly_12lev")
VARS = ["t", "u", "v", "w"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
HIGH_LEVELS = [600, 200, 150, 100]


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
    out = ds[VARS]
    if float(out.lat[0]) > float(out.lat[-1]):
        out = out.sortby("lat")
    out = out.assign_coords(lon=out.lon % 360).sortby("lon")
    return out.reset_coords(names=[x for x in out.coords if x not in out.dims], drop=True)


def build_high_daily(year: int) -> xr.Dataset:
    with xr.open_dataset(RAW_HIGH) as raw:
        # Apply basic strided indexing before coordinate reversal so netCDF4 does
        # not materialize the complete 0.25-degree global array.
        sub = raw[VARS].sel(valid_time=str(year), pressure_level=HIGH_LEVELS).isel(
            latitude=slice(None, None, 4),
            longitude=slice(None, None, 4),
        )
        times = pd.DatetimeIndex(sub.valid_time.values)
        if len(times) != 124 or sorted(set(times.hour)) != [0, 6, 12, 18]:
            raise ValueError(f"Unexpected four-times-daily coverage for {year}: {len(times)}")
        coords = {
            "time": times[::4].normalize().values,
            "lev": sub.pressure_level.values,
            "lat": sub.latitude.values,
            "lon": sub.longitude.values,
        }
        data_vars = {}
        for name in VARS:
            values = sub[name].values.reshape(31, 4, 4, 181, 360).mean(axis=1, dtype=np.float64)
            data_vars[name] = (("time", "lev", "lat", "lon"), values.astype("float32"))
    daily = xr.Dataset(data_vars, coords=coords)
    return normalize(daily).transpose("time", "lev", "lat", "lon")


def write_month(ds: xr.Dataset, out_file: Path, source: str) -> dict[str, object]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ds = ds[VARS].sel(lev=LEVELS).transpose("time", "lev", "lat", "lon")
    ds.attrs.update(
        {
            "title": "ERA5 daily 1-degree 12-level pressure-level fields",
            "source": source,
            "levels_hPa": ",".join(str(x) for x in LEVELS),
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in VARS}
    ds.to_netcdf(out_file, encoding=encoding)
    vals = np.concatenate([ds[v].values.ravel() for v in VARS])
    return {
        "file": str(out_file),
        "time_count": ds.sizes["time"],
        "level_count": ds.sizes["lev"],
        "lat_count": ds.sizes["lat"],
        "lon_count": ds.sizes["lon"],
        "nan_count": int(np.isnan(vals).sum()),
        "inf_count": int(np.isinf(vals).sum()),
    }


def prepare_october(year: int, overwrite: bool) -> dict[str, object]:
    out_file = OUT_ROOT / str(year) / f"{year}10.nc"
    if out_file.exists() and not overwrite:
        with xr.open_dataset(out_file) as ds:
            return write_check(ds, out_file)
    low_file = LOW8_ROOT / str(year) / f"{year}10.nc"
    if not low_file.exists():
        raise FileNotFoundError(low_file)
    high = build_high_daily(year)
    try:
        with xr.open_dataset(low_file) as low_raw:
            low = normalize(low_raw).load()
        try:
            if not np.array_equal(low.time.values, high.time.values):
                raise ValueError(f"Time mismatch for {year}-10")
            if not np.array_equal(low.lat.values, high.lat.values) or not np.array_equal(low.lon.values, high.lon.values):
                raise ValueError(f"Grid mismatch for {year}-10")
            merged = xr.concat([low, high], dim="lev").sel(lev=LEVELS)
            return write_month(merged, out_file, f"{low_file}; {RAW_HIGH}")
        finally:
            low.close()
    finally:
        high.close()


def write_check(ds: xr.Dataset, out_file: Path) -> dict[str, object]:
    return {
        "file": str(out_file),
        "time_count": ds.sizes["time"],
        "level_count": ds.sizes["lev"],
        "lat_count": ds.sizes["lat"],
        "lon_count": ds.sizes["lon"],
        "nan_count": int(sum(np.isnan(ds[v].values).sum() for v in VARS)),
        "inf_count": int(sum(np.isinf(ds[v].values).sum() for v in VARS)),
    }


def prepare_existing_month(year: int, month: int, overwrite: bool) -> dict[str, object]:
    out_file = OUT_ROOT / str(year) / f"{year}{month:02d}.nc"
    if out_file.exists() and not overwrite:
        with xr.open_dataset(out_file) as ds:
            return write_check(ds, out_file)
    source = FILLED17_ROOT / str(year) / f"{year}{month:02d}.nc"
    if not source.exists():
        raise FileNotFoundError(source)
    with xr.open_dataset(source) as raw:
        ds = normalize(raw).sel(lev=LEVELS).load()
    try:
        return write_month(ds, out_file, str(source))
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = []
    for year in [1986, 2023]:
        rows.append(prepare_october(year, args.overwrite))
    for month in [1, 2, 11, 12]:
        rows.append(prepare_existing_month(1986, month, args.overwrite))
    for month in [11, 12]:
        rows.append(prepare_existing_month(2023, month, args.overwrite))
    check = OUT_ROOT / "ERA5_12lev_selected_months_check.csv"
    pd.DataFrame(rows).sort_values("file").to_csv(check, index=False, encoding="utf-8-sig")
    print(f"[CHECK] {check}", flush=True)


if __name__ == "__main__":
    main()
