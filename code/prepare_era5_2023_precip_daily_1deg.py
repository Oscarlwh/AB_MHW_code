#!/usr/bin/env python3
"""Standardize downloaded ERA5 2023 daily precipitation rates to the project grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


RAW_ROOT = Path(
    "/path/to/data/ERA5_raw/precipitation/77a316c0dd1d18c1ab3e159f944fda58"
)
OUT_ROOT = Path(
    "/path/to/data/非绝热加热异常_ERA5反推/era5_convective_precip_peak_lag0_7"
)
OUT_FILE = OUT_ROOT / "ERA5_precipitation_rates_2023_ND_daily_1deg.nc"
CHECK_FILE = OUT_ROOT / "ERA5_precipitation_2023_ND_standardization_check.csv"

TARGET_LAT = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
TARGET_LON = np.arange(0.0, 360.0, 1.0, dtype=np.float32)

INPUT_SPECS = {
    "mcpr": {
        "pattern": "mean_convective_precipitation_rate*_daily-mean.nc",
        "source_var": "avg_cpr",
        "long_name": "Mean convective precipitation rate",
        "units": "kg m-2 s-1",
        "required": True,
    },
    "mlspr": {
        "pattern": "mean_large_scale_precipitation_rate*_daily-mean.nc",
        "source_var": "avg_lsprate",
        "long_name": "Mean large-scale precipitation rate",
        "units": "kg m-2 s-1",
        "required": False,
    },
    "mtpr": {
        "pattern": "mean_total_precipitation_rate*_daily-mean.nc",
        "source_var": "avg_tprate",
        "long_name": "Mean total precipitation rate",
        "units": "kg m-2 s-1",
        "required": True,
    },
    "mlspf": {
        "pattern": "mean_large_scale_precipitation_fraction*_daily-mean.nc",
        "source_var": "avg_ilspf",
        "long_name": "Mean large-scale precipitation fraction",
        "units": "1",
        "required": False,
    },
}


def find_input(raw_root: Path, pattern: str) -> Path | None:
    paths = [p for p in sorted(raw_root.glob(pattern)) if not p.name.startswith("._")]
    if not paths:
        return None
    if len(paths) > 1:
        raise ValueError(f"Multiple files match {pattern}: {paths}")
    return paths[0]


def normalize_source(path: Path, source_var: str) -> xr.DataArray:
    with xr.open_dataset(path, engine="netcdf4") as source:
        if source_var not in source:
            raise KeyError(f"{source_var} not found in {path}; variables={list(source.data_vars)}")
        da = source[source_var].rename(
            {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
        )
        da = da.transpose("time", "lat", "lon").load()

    if float(da["lat"].values[0]) > float(da["lat"].values[-1]):
        da = da.sortby("lat")
    da = da.assign_coords(lon=da["lon"] % 360).sortby("lon")

    lat = da["lat"].values
    lon = da["lon"].values
    lat_step = abs(float(lat[1] - lat[0]))
    lon_step = abs(float(lon[1] - lon[0]))
    if (
        abs(lat_step - 0.25) < 1e-8
        and abs(lon_step - 0.25) < 1e-8
        and np.isclose(lat[0], -90.0)
        and np.isclose(lon[0], 0.0)
    ):
        out = da.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))
        out = out.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)
    else:
        out = da.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")
    return out.astype("float32")


def validate_dataset(ds: xr.Dataset) -> pd.DataFrame:
    expected_sizes = {"time": 61, "lat": 181, "lon": 360}
    for dim, size in expected_sizes.items():
        if int(ds.sizes.get(dim, -1)) != size:
            raise ValueError(f"Unexpected {dim}: {ds.sizes.get(dim)} != {size}")

    time = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    expected_time = pd.date_range("2023-11-01", "2023-12-31", freq="D")
    if not time.equals(expected_time):
        raise ValueError(
            f"Unexpected time coverage: {time[0]}..{time[-1]}, count={len(time)}"
        )
    if not np.array_equal(ds["lat"].values, TARGET_LAT):
        raise ValueError("Latitude does not match the project 1-degree grid.")
    if not np.array_equal(ds["lon"].values, TARGET_LON):
        raise ValueError("Longitude does not match the project 1-degree grid.")

    rows = []
    for name, da in ds.data_vars.items():
        values = da.values
        finite = np.isfinite(values)
        if not finite.all():
            raise ValueError(f"{name} contains NaN or Inf values.")
        if name in {"mcpr", "mlspr", "mtpr"} and float(values.min()) < -1e-12:
            raise ValueError(f"{name} contains unexpected negative values: {values.min()}")
        rows.append(
            {
                "variable": name,
                "long_name": da.attrs.get("long_name", ""),
                "units": da.attrs.get("units", ""),
                "n_time": int(da.sizes["time"]),
                "n_lat": int(da.sizes["lat"]),
                "n_lon": int(da.sizes["lon"]),
                "time_first": str(time[0].date()),
                "time_last": str(time[-1].date()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-file", type=Path, default=OUT_FILE)
    parser.add_argument("--check-file", type=Path, default=CHECK_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = {}
    source_files = {}

    for target_name, spec in INPUT_SPECS.items():
        path = find_input(args.raw_root, spec["pattern"])
        if path is None:
            if spec["required"]:
                raise FileNotFoundError(
                    f"Required ERA5 input not found: {spec['pattern']} in {args.raw_root}"
                )
            print(f"[OPTIONAL MISSING] {target_name}: {spec['pattern']}")
            continue
        print(f"[READ] {target_name}: {path}")
        da = normalize_source(path, spec["source_var"])
        da.name = target_name
        da.attrs.update(
            {
                "long_name": spec["long_name"],
                "units": spec["units"],
                "source_file": str(path),
            }
        )
        arrays[target_name] = da
        source_files[target_name] = str(path)

    ds = xr.Dataset(arrays)
    ds.attrs.update(
        {
            "title": "ERA5 daily precipitation diagnostics for November-December 2023 on the project 1-degree grid",
            "source_resolution": "0.25 degree",
            "target_grid": "lat=-90..90, lon=0..359, 1 degree",
            "daily_statistic": "daily mean from 1-hourly ERA5 data, UTC+00:00",
            "created_by": Path(__file__).name,
            "note": "mlspf is a precipitation-area fraction and is not used as a precipitation rate.",
        }
    )

    check = validate_dataset(ds)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in ds.data_vars
    }
    tmp = args.out_file.with_suffix(args.out_file.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    ds.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(tmp, engine="netcdf4") as saved:
        validate_dataset(saved)
    tmp.replace(args.out_file)
    check["output_file"] = str(args.out_file)
    check["source_file"] = check["variable"].map(source_files)
    check.to_csv(args.check_file, index=False, encoding="utf-8-sig")

    print(f"[NC] {args.out_file}")
    print(f"[CHECK] {args.check_file}")
    print(f"[VARS] {list(ds.data_vars)}")


if __name__ == "__main__":
    main()
