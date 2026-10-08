from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


LOW_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly_17lev")

HIGH_FILES = {
    (1986, 1): Path("/path/to/data/ERA5_raw/552a3590de21d8e3af8b32b16fcc572.nc"),
    (1986, 2): Path("/path/to/data/ERA5_raw/1986.2_2023.1-2/f4ec7aa3f5ac12e45c5a9d1ce6e2863c.nc"),
    (1986, 11): Path("/path/to/data/ERA5_raw/c78c725e0e836962ce4f98c523f1c1f3.nc"),
    (1986, 12): Path("/path/to/data/ERA5_raw/c78c725e0e836962ce4f98c523f1c1f3.nc"),
    (2023, 1): Path("/path/to/data/ERA5_raw/1986.2_2023.1-2/a1e0f8c7cea231b15e9c5aaffcb5a35a.nc"),
    (2023, 2): Path("/path/to/data/ERA5_raw/1986.2_2023.1-2/a1e0f8c7cea231b15e9c5aaffcb5a35a.nc"),
    (2023, 11): Path("/path/to/data/ERA5_raw/62b4f2b5cc526dde690da52d30ca2513.nc"),
    (2023, 12): Path("/path/to/data/ERA5_raw/62b4f2b5cc526dde690da52d30ca2513.nc"),
}

TASKS = [(1986, 1), (1986, 2), (1986, 11), (1986, 12), (2023, 1), (2023, 2), (2023, 11), (2023, 12)]

VARS = ["t", "u", "v", "w"]
LOW_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]
HIGH_LEVELS = [600, 200, 150, 100, 70, 50, 30, 20, 10]
ALL_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

TARGET_LAT = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
TARGET_LON = np.arange(0.0, 360.0, 1.0, dtype=np.float32)


def month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return start, end


def standardize_high_month(ds_high: xr.Dataset, year: int, month: int) -> xr.Dataset:
    start, end = month_bounds(year, month)
    times = pd.to_datetime(ds_high["valid_time"].values)
    idx_time = np.where((times >= start) & (times <= end + pd.Timedelta(hours=23)))[0]
    if len(idx_time) == 0:
        raise ValueError(f"No high-level records for {year}-{month:02d}")
    if len(idx_time) % 4 != 0:
        raise ValueError(f"Expected 4 records per day for {year}-{month:02d}, got {len(idx_time)} records")

    raw_levels = [float(x) for x in ds_high["pressure_level"].values]
    idx_level = [raw_levels.index(float(level)) for level in HIGH_LEVELS]
    out_time = pd.date_range(start, end, freq="D")
    if len(idx_time) != len(out_time) * 4:
        raise ValueError(f"Unexpected record count for {year}-{month:02d}: {len(idx_time)}")

    data_vars = {}
    for name in VARS:
        print(f"  [HIGH] {year}-{month:02d} {name}", flush=True)
        da = ds_high[name].isel(
            valid_time=idx_time,
            pressure_level=idx_level,
            latitude=slice(None, None, 4),
            longitude=slice(None, None, 4),
        )
        # CDS latitude is north-to-south; existing ERA5 files are south-to-north.
        da = da.isel(latitude=slice(None, None, -1))
        arr = da.values.astype("float32")
        arr = arr.reshape(len(out_time), 4, len(HIGH_LEVELS), len(TARGET_LAT), len(TARGET_LON)).mean(axis=1)
        data_vars[name] = (("time", "lev", "lat", "lon"), arr.astype("float32"))

    daily = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": out_time,
            "lev": np.array(HIGH_LEVELS, dtype=np.float64),
            "lat": TARGET_LAT,
            "lon": TARGET_LON,
        },
    )
    for name in VARS:
        daily[name].attrs.update(ds_high[name].attrs)
    return daily


def read_low_month(year: int, month: int) -> xr.Dataset:
    path = LOW_ROOT / str(year) / f"{year}{month:02d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)[VARS]
    missing = sorted(set(LOW_LEVELS) - set(float(x) for x in ds["lev"].values))
    if missing:
        raise ValueError(f"{path} missing low levels: {missing}")
    return ds.sel(lev=LOW_LEVELS).astype("float32")


def merge_month(year: int, month: int, ds_high_year: xr.Dataset) -> tuple[Path, dict[str, object]]:
    low = read_low_month(year, month)
    high = standardize_high_month(ds_high_year, year, month)

    merged = xr.concat([low, high], dim="lev").sel(lev=ALL_LEVELS)
    merged = merged.assign_coords(
        lev=np.array(ALL_LEVELS, dtype=np.float64),
        lat=TARGET_LAT,
        lon=TARGET_LON,
    )
    merged.attrs.update(
        {
            "title": "ERA5 pressure-level t/u/v/w daily 1 degree fields, merged 17 levels",
            "source_low_8lev": str(LOW_ROOT / str(year) / f"{year}{month:02d}.nc"),
            "source_high_9lev": str(HIGH_FILES[(year, month)]),
            "created_by": Path(__file__).name,
            "note": "Original 8-level files are not overwritten.",
        }
    )
    for name in VARS:
        merged[name].attrs.update(low[name].attrs)

    out_dir = OUT_ROOT / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}{month:02d}.nc"

    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}
        for name in VARS
    }
    merged.to_netcdf(out_path, encoding=encoding)

    nan_count = 0
    inf_count = 0
    mins = []
    maxs = []
    means = []
    for name in VARS:
        arr = merged[name].values
        nan_count += int(np.isnan(arr).sum())
        inf_count += int(np.isinf(arr).sum())
        mins.append(float(np.nanmin(arr)))
        maxs.append(float(np.nanmax(arr)))
        means.append(float(np.nanmean(arr)))
    rec = {
        "year": year,
        "month": month,
        "out_file": str(out_path),
        "time_start": str(pd.to_datetime(merged["time"].values[0]).date()),
        "time_end": str(pd.to_datetime(merged["time"].values[-1]).date()),
        "ntime": int(merged.sizes["time"]),
        "nlev": int(merged.sizes["lev"]),
        "levels": ",".join(str(int(x)) for x in merged["lev"].values),
        "lat_first": float(merged["lat"].values[0]),
        "lat_last": float(merged["lat"].values[-1]),
        "lon_first": float(merged["lon"].values[0]),
        "lon_last": float(merged["lon"].values[-1]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": min(mins),
        "max": max(maxs),
        "mean": float(np.mean(means)),
    }
    low.close()
    high.close()
    merged.close()
    return out_path, rec


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    opened: dict[Path, xr.Dataset] = {}
    try:
        for year, month in TASKS:
            out_path = OUT_ROOT / str(year) / f"{year}{month:02d}.nc"
            if out_path.exists():
                print(f"[SKIP] {out_path}", flush=True)
                continue
            high_file = HIGH_FILES[(year, month)]
            if high_file not in opened:
                if not high_file.exists():
                    raise FileNotFoundError(high_file)
                print(f"[OPEN] {high_file}", flush=True)
                opened[high_file] = xr.open_dataset(high_file)
            print(f"[MERGE] {year}-{month:02d}", flush=True)
            out_path, rec = merge_month(year, month, opened[high_file])
            print(f"[WROTE] {out_path}", flush=True)
            records.append(rec)
    finally:
        for ds in opened.values():
            ds.close()

    existing_rows = []
    old_check = OUT_ROOT / "era5_daily_1deg_monthly_17lev_check.csv"
    if old_check.exists():
        existing_rows = pd.read_csv(old_check).to_dict("records")
    check = pd.DataFrame.from_records(existing_rows + records)
    if not check.empty:
        check = check.drop_duplicates(subset=["year", "month"], keep="last").sort_values(["year", "month"])
    check_path = OUT_ROOT / "era5_daily_1deg_monthly_17lev_check.csv"
    check.to_csv(check_path, index=False)
    print(f"[CHECK] {check_path}", flush=True)


if __name__ == "__main__":
    main()
