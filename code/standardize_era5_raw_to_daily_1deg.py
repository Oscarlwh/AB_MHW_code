#!/usr/bin/env python3
"""Interpolate downloaded ERA5 raw pressure-level data to the existing 1 degree grid.

Input: official 0.25 degree ERA5 files under /path/to/data/ERA5_raw/YYYY/.
Output: daily 1 degree files under
/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg/YYYY.nc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_ROOT = Path("/path/to/data/ERA5_raw")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg")
CHECK_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/checks")
YEARS = [1986, 2023, 2024]
LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]
TARGET_LAT = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
TARGET_LON = np.arange(0.0, 360.0, 1.0, dtype=np.float32)


RENAME = {
    "valid_time": "time",
    "pressure_level": "lev",
    "level": "lev",
    "latitude": "lat",
    "longitude": "lon",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
}


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {k: v for k, v in RENAME.items() if k in ds.variables or k in ds.dims}
    ds = ds.rename(rename)
    if "lon" in ds.coords:
        lon = ds["lon"] % 360
        ds = ds.assign_coords(lon=lon).sortby("lon")
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lev" in ds.coords:
        ds = ds.sel(lev=LEVELS)
    keep = [v for v in ["t", "u", "v", "w"] if v in ds.data_vars]
    missing = sorted(set(["t", "u", "v", "w"]) - set(keep))
    if missing:
        raise KeyError(f"Missing required variables: {missing}")
    return ds[keep]


def files_for_year(year: int, raw_root: Path, filename_globs: list[str]) -> list[Path]:
    files: list[Path] = []
    for filename_glob in filename_globs:
        files.extend(p for p in (raw_root / str(year)).glob(filename_glob) if not p.name.startswith("._"))
    return sorted(set(files))


def open_standardized_inputs(files: list[Path]) -> xr.Dataset:
    datasets = []
    for path in files:
        print(f"[READ] {path}")
        ds = xr.open_dataset(path, chunks="auto", engine="netcdf4")
        ds = normalize_dataset(ds)
        ds = ds.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time", coords="minimal", compat="override")


def standardize_year(year: int, raw_root: Path, out_root: Path, filename_globs: list[str]) -> dict:
    files = files_for_year(year, raw_root, filename_globs)
    if not files:
        raise FileNotFoundError(f"No raw ERA5 files found for {year} under {raw_root / str(year)}")

    print(f"[OPEN] {year}: {len(files)} raw files")
    ds = open_standardized_inputs(files)
    ds = ds.sortby("time")
    daily = ds.resample(time="1D").mean(skipna=True)
    daily = daily.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)
    daily = daily.astype({name: "float32" for name in daily.data_vars})

    daily.attrs.update(
        {
            "title": f"ERA5 pressure-level t/u/v/w daily 1 degree fields, {year}",
            "source": "ERA5 pressure-level official 0.25 degree data, 00/06/12/18 UTC",
            "interpolation": "Bilinear interpolation to lon=0..359, lat=-90..90, 1 degree grid",
            "created_by": "standardize_era5_raw_to_daily_1deg.py",
        }
    )
    for name in daily.data_vars:
        daily[name].attrs.update(ds[name].attrs)

    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"{year}.nc"
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in daily.data_vars}
    print(f"[WRITE] {out_file}")
    daily.to_netcdf(out_file, encoding=encoding)

    rec = {
        "year": year,
        "file": str(out_file),
        "n_time": int(daily.sizes["time"]),
        "n_lev": int(daily.sizes["lev"]),
        "n_lat": int(daily.sizes["lat"]),
        "n_lon": int(daily.sizes["lon"]),
        "time_first": str(pd.to_datetime(daily["time"].values[0])),
        "time_last": str(pd.to_datetime(daily["time"].values[-1])),
        "lev": ",".join(str(int(x)) for x in daily["lev"].values),
    }
    for name in daily.data_vars:
        arr = daily[name]
        rec[f"{name}_nan"] = int(np.isnan(arr.values).sum())
        rec[f"{name}_inf"] = int(np.isinf(arr.values).sum())
        rec[f"{name}_min"] = float(arr.min(skipna=True))
        rec[f"{name}_max"] = float(arr.max(skipna=True))
        rec[f"{name}_mean"] = float(arr.mean(skipna=True))
    ds.close()
    daily.close()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=YEARS)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--check-root", type=Path, default=CHECK_ROOT)
    parser.add_argument(
        "--filename-glob",
        nargs="+",
        default=["*.nc"],
        help=(
            "Raw filename pattern(s) to read, e.g. "
            "'era5_pl_tuvomega_8lev_1deg_4times_*.nc' "
            "'era5_pl_tuvomega_8lev_native025deg_4times_*.nc'."
        ),
    )
    args = parser.parse_args()

    records = [standardize_year(y, args.raw_root, args.out_root, args.filename_glob) for y in args.years]
    args.check_root.mkdir(parents=True, exist_ok=True)
    check_csv = args.check_root / "era5_daily_1deg_standardize_check.csv"
    pd.DataFrame(records).to_csv(check_csv, index=False, encoding="utf-8-sig")
    print(f"[CHECK] {check_csv}")


if __name__ == "__main__":
    main()
