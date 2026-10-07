#!/usr/bin/env python3
"""Convert available monthly ERA5 raw files to daily 1 degree monthly files.

This script is intentionally partial-month friendly. It writes one output file
per source month instead of pretending an incomplete download is a complete
year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_ROOT = Path("/path/to/data/ERA5_raw")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly")
CHECK_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/checks")
EXPECTED_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]
TARGET_LAT = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
TARGET_LON = np.arange(0.0, 360.0, 1.0, dtype=np.float32)


RENAME = {
    "valid_time": "time",
    "pressure_level": "lev",
    "level": "lev",
    "latitude": "lat",
    "longitude": "lon",
}


def normalize(ds: xr.Dataset) -> xr.Dataset:
    rename = {k: v for k, v in RENAME.items() if k in ds.variables or k in ds.dims}
    ds = ds.rename(rename)
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=ds["lon"] % 360)
    levels = [int(x) for x in ds["lev"].values.tolist()]
    if levels != EXPECTED_LEVELS:
        raise ValueError(f"Unexpected levels: {levels}")
    missing = sorted(set(["t", "u", "v", "w"]) - set(ds.data_vars))
    if missing:
        raise KeyError(f"Missing variables: {missing}")
    return ds[["t", "u", "v", "w"]]


def month_from_dataset(ds: xr.Dataset) -> tuple[int, int]:
    times = pd.to_datetime(ds["time"].values)
    months = sorted(set((int(t.year), int(t.month)) for t in times))
    if len(months) != 1:
        raise ValueError(f"Expected one calendar month, found {months}")
    return months[0]


def to_target_grid(ds: xr.Dataset) -> xr.Dataset:
    lat_values = ds["lat"].values
    lon_values = ds["lon"].values
    if len(lat_values) > 1 and len(lon_values) > 1:
        lat_step = abs(float(lat_values[1] - lat_values[0]))
        lon_step = abs(float(lon_values[1] - lon_values[0]))
        lat_stride = int(round(1.0 / lat_step)) if lat_step else 0
        lon_stride = int(round(1.0 / lon_step)) if lon_step else 0
        regular_grid = (
            lat_stride >= 1
            and lon_stride >= 1
            and abs(lat_step * lat_stride - 1.0) < 1e-6
            and abs(lon_step * lon_stride - 1.0) < 1e-6
            and abs(float(lon_values[0])) < 1e-6
        )
        if regular_grid:
            lat_slice = slice(None, None, -lat_stride) if lat_values[0] > lat_values[-1] else slice(None, None, lat_stride)
            ds = ds.isel(lat=lat_slice, lon=slice(None, None, lon_stride))
            return ds.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)

    return ds.sortby("lat").sortby("lon").interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")


def summarize_daily(path: Path, source: Path | None = None) -> dict:
    with xr.open_dataset(path, engine="netcdf4") as daily:
        expected_sizes = {"lev": 8, "lat": 181, "lon": 360}
        for dim, size in expected_sizes.items():
            if int(daily.sizes.get(dim, -1)) != size:
                raise ValueError(f"Unexpected {dim} size in {path}: {daily.sizes.get(dim)}")
        for coord in ["lat", "lon"]:
            if bool(np.isnan(daily[coord].values).any()):
                raise ValueError(f"NaN values found in coordinate {coord}: {path}")
        times = pd.to_datetime(daily["time"].values)
        year = int(times[0].year)
        month = int(times[0].month)
        rec = {
            "source_file": "" if source is None else str(source),
            "out_file": str(path),
            "year": year,
            "month": month,
            "n_time": int(daily.sizes["time"]),
            "n_lev": int(daily.sizes["lev"]),
            "n_lat": int(daily.sizes["lat"]),
            "n_lon": int(daily.sizes["lon"]),
            "time_first": str(times[0]),
            "time_last": str(times[-1]),
        }
        for name in daily.data_vars:
            arr = daily[name]
            if int(np.isnan(arr.values).sum()) == int(arr.size):
                raise ValueError(f"All values are NaN for {name}: {path}")
            rec[f"{name}_nan"] = int(np.isnan(arr.values).sum())
            rec[f"{name}_inf"] = int(np.isinf(arr.values).sum())
            rec[f"{name}_min"] = float(arr.min(skipna=True))
            rec[f"{name}_max"] = float(arr.max(skipna=True))
            rec[f"{name}_mean"] = float(arr.mean(skipna=True))
        return rec


def standardize_file(path: Path, out_root: Path) -> dict:
    print(f"[OPEN] {path}")
    ds = xr.open_dataset(path, chunks="auto", engine="netcdf4")
    try:
        ds = normalize(ds)
        year, month = month_from_dataset(ds)
        ds = to_target_grid(ds)
        ds = ds.sortby("time")
        daily = ds.resample(time="1D").mean(skipna=True)
        daily = daily.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)
        daily = daily.astype({name: "float32" for name in daily.data_vars})
        daily.attrs.update(
            {
                "title": f"ERA5 pressure-level t/u/v/w daily 1 degree fields, {year}-{month:02d}",
                "source_file": str(path),
                "created_by": "standardize_era5_months_to_daily_1deg.py",
            }
        )
        out_dir = out_root / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{year}{month:02d}.nc"
        if out_file.exists():
            try:
                rec = summarize_daily(out_file, path)
                print(f"[SKIP VALID] {out_file}")
                daily.close()
                return rec
            except Exception:
                print(f"[REPLACE INVALID] {out_file}")
        encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in daily.data_vars}
        tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
        if tmp_file.exists():
            tmp_file.unlink()
        print(f"[WRITE] {tmp_file}")
        daily.to_netcdf(tmp_file, encoding=encoding)
        rec = summarize_daily(tmp_file, path)
        tmp_file.replace(out_file)
        print(f"[OK] {out_file}")
        daily.close()
        return rec
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=[1986, 2023])
    parser.add_argument("--months", nargs="*", type=int)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--check-root", type=Path, default=CHECK_ROOT)
    args = parser.parse_args()

    files: list[Path] = []
    for year in args.years:
        for path in sorted((args.raw_root / str(year)).glob("era5_pl_tuvomega_8lev_*_4times_*.nc")):
            if path.name.startswith("._") or "BAD" in path.name:
                continue
            stem = path.stem
            yyyymm = stem.rsplit("_", 1)[-1][:6]
            if not yyyymm.isdigit():
                continue
            month = int(yyyymm[-2:])
            if args.months and month not in args.months:
                continue
            files.append(path)

    records = []
    for path in sorted(files):
        try:
            records.append(standardize_file(path, args.out_root))
        except Exception as exc:
            print(f"[ERROR] {path}: {exc!r}")

    args.check_root.mkdir(parents=True, exist_ok=True)
    check_csv = args.check_root / "era5_monthly_daily_1deg_check.csv"
    pd.DataFrame(records).to_csv(check_csv, index=False, encoding="utf-8-sig")
    print(f"[CHECK] {check_csv}")


if __name__ == "__main__":
    main()
