#!/usr/bin/env python3
"""Build N=13 plus 2004 ERA5 temperature/Z time-pressure sections.

The script creates cosine-latitude-weighted daily area means for 50-60N,
180-200E, removes linear trends over 1981-2024, subtracts a 1981-2020
no-leap daily climatology, and plots centered 5-day means from lead 10 to
lag 10 days relative to each MHW peak.

The large annual ERA5 files are only read once. Small yearly and combined
regional caches are retained under the output directory. Missing complete
years (1986, 2023, and 2024) are downloaded from CDS as regional 1-degree
monthly files without replacing any existing source data.
"""

from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from matplotlib.transforms import ScaledTranslation
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr


G = 9.80665

ERA5_ROOT = Path("/path/to/data/ERA5")
Z500_ANOM_FILE = Path(
    "/path/to/data/ABH_170E240E_1981_2024_5N90N/anomaly/"
    "ERA5_Z500_anom_NOLEAP_1981_2024_vs1981_2020_gpm_170E240E_5N90N_detrended.nc"
)
OUT_ROOT = Path(
    "/path/to/data/ABH_170E240E_1981_2024_5N90N/composite/"
    "time_pressure_TZ_13_plus_2004_lead10_lag10"
)
CACHE_ROOT = OUT_ROOT / "cache"
YEARLY_CACHE_ROOT = CACHE_ROOT / "yearly_area_mean"
DOWNLOAD_ROOT = CACHE_ROOT / "era5_missing_years_regional"

BASE_NAME = (
    "ERA5_TZ_anomaly_time_pressure_group13_plus_2004_peak_"
    "lead10_lag10_5dmean_180E200E_50N60N"
)
OUT_NC = OUT_ROOT / f"{BASE_NAME}.nc"
OUT_PNG = OUT_ROOT / f"{BASE_NAME}.png"
OUT_PDF = OUT_ROOT / f"{BASE_NAME}.pdf"
OUT_CHECK = OUT_ROOT / "check_ERA5_TZ_anomaly_time_pressure_group13_plus_2004.txt"

DAILY_RAW_CACHE = CACHE_ROOT / (
    "ERA5_T_Z_daily_area_mean_1981_2024_180E200E_50N60N_"
    "1000_100hPa_no_leap.nc"
)
DAILY_ANOM_CACHE = CACHE_ROOT / (
    "ERA5_T_Z_daily_area_mean_anomaly_1981_2024_vs1981_2020_"
    "180E200E_50N60N_1000_100hPa_detrended_no_leap.nc"
)

YEARS = list(range(1981, 2025))
MISSING_ANNUAL_YEARS = [1986, 2023, 2024]
LEVELS = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100], dtype=np.int16)
LAT_MIN, LAT_MAX = 50.0, 60.0
LON_MIN, LON_MAX = 180.0, 200.0
RELATIVE_DAYS = np.arange(-10, 11, dtype=np.int16)
ROLL_OFFSETS = np.arange(-2, 3, dtype=np.int16)

PEAK_DATES = pd.to_datetime(
    [
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
)
SPECIAL_PEAK = pd.Timestamp("2004-12-24")
SPECIAL_INDEX = int(np.where(PEAK_DATES == SPECIAL_PEAK)[0][0])
REGULAR_INDICES = np.array([i for i in range(len(PEAK_DATES)) if i != SPECIAL_INDEX], dtype=np.int16)


def calc_doy_nl(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    if ts.month == 2 and ts.day == 29:
        raise ValueError("Feb 29 is excluded from the no-leap calendar")
    return int(pd.Timestamp(2001, ts.month, ts.day).dayofyear)


def expected_no_leap_days(year: int) -> int:
    return 365


def date_strings_from_nc(var: nc.Variable) -> pd.DatetimeIndex:
    calendar_name = getattr(var, "calendar", "standard")
    decoded = nc.num2date(
        var[:],
        units=var.units,
        calendar=calendar_name,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return pd.DatetimeIndex(pd.to_datetime(decoded)).normalize()


def coordinate_indices(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where((values >= lower - 1.0e-6) & (values <= upper + 1.0e-6))[0]


def level_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    indices = []
    for level in LEVELS:
        match = np.where(np.isclose(values, float(level)))[0]
        if len(match) != 1:
            raise ValueError(f"Expected one match for {level} hPa; found {len(match)} in {values}")
        indices.append(int(match[0]))
    return np.asarray(indices, dtype=int)


def weighted_spatial_mean(values: np.ndarray, lat_values: np.ndarray) -> np.ndarray:
    """Area mean for an array shaped time, level, lat, lon."""
    arr = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
    weights = np.cos(np.deg2rad(np.asarray(lat_values, dtype=np.float64)))[None, None, :, None]
    valid = np.isfinite(arr)
    numerator = np.nansum(arr * weights, axis=(2, 3))
    denominator = np.sum(valid * weights, axis=(2, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator
    return result.astype(np.float32)


def validate_year_dataset(ds: xr.Dataset, year: int, context: str) -> None:
    required = {"temperature", "geopotential_height"}
    missing = sorted(required.difference(ds.data_vars))
    if missing:
        raise ValueError(f"{context}: missing variables {missing}")
    if not np.array_equal(ds.level.values.astype(np.int16), LEVELS):
        raise ValueError(f"{context}: unexpected levels {ds.level.values}")
    times = pd.DatetimeIndex(ds.time.values)
    if len(times) != expected_no_leap_days(year):
        raise ValueError(f"{context}: expected 365 days, found {len(times)}")
    if times.has_duplicates:
        raise ValueError(f"{context}: duplicate dates")
    if ((times.month == 2) & (times.day == 29)).any():
        raise ValueError(f"{context}: Feb 29 was not removed")
    for name in required:
        if not np.isfinite(ds[name].values).all():
            raise ValueError(f"{context}: {name} contains NaN/Inf")


def extract_local_annual_file(year: int, source: Path) -> xr.Dataset:
    print(f"[EXTRACT] {year}: {source}", flush=True)
    with nc.Dataset(source) as root:
        for variable in ["t", "z", "time", "lev", "lat", "lon"]:
            if variable not in root.variables:
                raise KeyError(f"{source}: missing {variable}")

        times = date_strings_from_nc(root.variables["time"])
        lev_idx = level_indices(root.variables["lev"][:])
        lat_all = np.asarray(root.variables["lat"][:], dtype=float)
        lon_all = np.asarray(root.variables["lon"][:], dtype=float) % 360.0
        lat_idx = coordinate_indices(lat_all, LAT_MIN, LAT_MAX)
        lon_idx = coordinate_indices(lon_all, LON_MIN, LON_MAX)
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            raise ValueError(f"{source}: empty regional selection")
        if not np.all(np.diff(lat_idx) == 1) or not np.all(np.diff(lon_idx) == 1):
            raise ValueError(f"{source}: regional indices are not contiguous")

        lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
        lon_slice = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
        lat_region = lat_all[lat_idx]
        nt = len(times)
        temperature = np.full((nt, len(LEVELS)), np.nan, dtype=np.float32)
        height = np.full_like(temperature, np.nan)

        for start in range(0, nt, 31):
            end = min(nt, start + 31)
            t_block = root.variables["t"][start:end, lev_idx, lat_slice, lon_slice]
            z_block = root.variables["z"][start:end, lev_idx, lat_slice, lon_slice]
            temperature[start:end] = weighted_spatial_mean(t_block, lat_region)
            height[start:end] = weighted_spatial_mean(z_block, lat_region) / G
            print(f"  {year}: days {start:03d}-{end - 1:03d}", flush=True)

    keep = ~((times.month == 2) & (times.day == 29))
    ds = xr.Dataset(
        data_vars={
            "temperature": (("time", "level"), temperature[keep]),
            "geopotential_height": (("time", "level"), height[keep]),
        },
        coords={
            "time": times[keep].values.astype("datetime64[ns]"),
            "level": LEVELS,
        },
        attrs={
            "source_file": str(source),
            "area": f"{LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
            "spatial_average": "cosine-latitude weighted mean over valid grid cells",
            "calendar": "no-leap",
        },
    )
    ds.temperature.attrs.update({"long_name": "Area-mean daily temperature", "units": "K"})
    ds.geopotential_height.attrs.update(
        {"long_name": "Area-mean daily geopotential height", "units": "gpm", "conversion": "z / 9.80665"}
    )
    validate_year_dataset(ds, year, str(source))
    return ds


def days_for_month(year: int, month: int) -> list[str]:
    return [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)]


def download_chunk_path(year: int, months: tuple[int, ...]) -> Path:
    month_tag = f"m{months[0]:02d}-{months[-1]:02d}"
    return DOWNLOAD_ROOT / str(year) / (
        f"era5_pl_t_z_12lev_1deg_4times_50N60N_180E200E_{year}_{month_tag}.nc"
    )


def validate_download(path: Path) -> None:
    with xr.open_dataset(path) as ds:
        names = set(ds.data_vars)
        if not ({"t", "z"} <= names or {"temperature", "geopotential"} <= names):
            raise ValueError(f"{path}: expected t/z variables, found {sorted(names)}")


def download_missing_chunk(year: int, months: tuple[int, ...], retries: int = 5) -> Path:
    """Download one regional month below the CDS cost limit."""
    path = download_chunk_path(year, months)
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy_month_path = path.parent / (
        f"era5_pl_t_z_12lev_1deg_4times_50N60N_180E200E_{year}{months[0]:02d}.nc"
    )
    if len(months) == 1 and legacy_month_path.exists():
        validate_download(legacy_month_path)
        print(f"[DOWNLOAD CACHE] {legacy_month_path}", flush=True)
        return legacy_month_path
    if path.exists():
        validate_download(path)
        print(f"[DOWNLOAD CACHE] {path}", flush=True)
        return path

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("cdsapi is required to download missing ERA5 years") from exc

    request = {
        "product_type": ["reanalysis"],
        "variable": ["temperature", "geopotential"],
        "pressure_level": [str(x) for x in LEVELS],
        "year": [str(year)],
        "month": [f"{month:02d}" for month in months],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
        "grid": [1.0, 1.0],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    client = cdsapi.Client()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[DOWNLOAD {attempt}/{retries}] {year} months {months[0]:02d}-{months[-1]:02d}",
                flush=True,
            )
            client.retrieve("reanalysis-era5-pressure-levels", request, str(tmp))
            validate_download(tmp)
            tmp.replace(path)
            print(f"[DOWNLOADED] {path}", flush=True)
            return path
        except Exception as exc:
            last_error = exc
            print(
                f"[DOWNLOAD RETRY] {year} months {months[0]:02d}-{months[-1]:02d}: {exc!r}",
                flush=True,
            )
            if tmp.exists():
                tmp.unlink()
            if "cost limits exceeded" in str(exc).lower():
                raise
            if attempt < retries:
                time.sleep(30)
    raise RuntimeError(
        f"Failed to download {year} months {months[0]:02d}-{months[-1]:02d}: {last_error!r}"
    ) from last_error


def normalize_downloaded_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in {
        "valid_time": "time",
        "pressure_level": "level",
        "lev": "level",
        "latitude": "lat",
        "longitude": "lon",
        "temperature": "t",
        "geopotential": "z",
    }.items():
        if old in ds.dims or old in ds.coords or old in ds.data_vars:
            if new not in ds.dims and new not in ds.coords and new not in ds.data_vars:
                rename[old] = new
    out = ds.rename(rename)
    if "t" not in out or "z" not in out:
        raise KeyError(f"Downloaded ERA5 data lacks t/z: {list(out.data_vars)}")
    if float(out.lat.values[0]) > float(out.lat.values[-1]):
        out = out.sortby("lat")
    if float(out.lon.min()) < 0 or float(out.lon.max()) <= 180:
        out = out.assign_coords(lon=out.lon % 360).sortby("lon")
    return out[["t", "z"]].sel(
        level=LEVELS,
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )


def extract_downloaded_year(year: int) -> xr.Dataset:
    chunks = [(month,) for month in range(1, 13)]
    paths: dict[tuple[int, ...], Path] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_missing_chunk, year, months): months for months in chunks
        }
        for future in as_completed(futures):
            months = futures[future]
            paths[months] = future.result()

    quarterly = []
    for months in chunks:
        path = paths[months]
        print(
            f"[EXTRACT DOWNLOADED] {year} months {months[0]:02d}-{months[-1]:02d}",
            flush=True,
        )
        with xr.open_dataset(path) as opened:
            ds = normalize_downloaded_dataset(opened)
            daily = ds.resample(time="1D").mean("time", skipna=True)
            weights = np.cos(np.deg2rad(daily.lat))
            quarterly.append(daily.weighted(weights).mean(("lat", "lon"), skipna=True).load())
    merged = xr.concat(quarterly, dim="time", coords="minimal", compat="override").sortby("time")
    times = pd.DatetimeIndex(merged.time.values).normalize()
    merged = merged.assign_coords(time=times.values.astype("datetime64[ns]"))
    keep = ~((times.month == 2) & (times.day == 29))
    merged = merged.isel(time=keep)
    out = xr.Dataset(
        data_vars={
            "temperature": merged.t.astype(np.float32),
            "geopotential_height": (merged.z / G).astype(np.float32),
        },
        coords={"time": merged.time, "level": LEVELS},
        attrs={
            "source_files": "\n".join(str(paths[months]) for months in chunks),
            "area": f"{LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
            "spatial_average": "cosine-latitude weighted mean over valid grid cells",
            "temporal_average": "mean of 00, 06, 12, and 18 UTC",
            "calendar": "no-leap",
        },
    )
    out.temperature.attrs.update({"long_name": "Area-mean daily temperature", "units": "K"})
    out.geopotential_height.attrs.update(
        {"long_name": "Area-mean daily geopotential height", "units": "gpm", "conversion": "z / 9.80665"}
    )
    validate_year_dataset(out, year, f"downloaded year {year}")
    for item in quarterly:
        item.close()
    merged.close()
    return out


def write_year_cache(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "temperature": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "geopotential_height": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    ds.to_netcdf(path, encoding=encoding)


def year_cache_path(year: int) -> Path:
    return YEARLY_CACHE_ROOT / (
        f"ERA5_T_Z_daily_area_mean_180E200E_50N60N_1000_100hPa_{year}_no_leap.nc"
    )


def build_year_cache(year: int, force: bool = False) -> Path:
    path = year_cache_path(year)
    if path.exists() and not force:
        with xr.open_dataset(path) as ds:
            validate_year_dataset(ds.load(), year, str(path))
        print(f"[YEAR CACHE] {path}", flush=True)
        return path
    if path.exists():
        path.unlink()

    local = ERA5_ROOT / f"{year}.nc"
    if local.exists():
        ds = extract_local_annual_file(year, local)
    elif year in MISSING_ANNUAL_YEARS:
        ds = extract_downloaded_year(year)
    else:
        raise FileNotFoundError(f"No ERA5 annual source for {year}: {local}")
    write_year_cache(ds, path)
    ds.close()
    print(f"[YEAR SAVED] {path}", flush=True)
    gc.collect()
    return path


def build_combined_raw_cache(force: bool = False, force_years: bool = False) -> xr.Dataset:
    if DAILY_RAW_CACHE.exists() and not force:
        print(f"[RAW CACHE] {DAILY_RAW_CACHE}", flush=True)
        return xr.open_dataset(DAILY_RAW_CACHE)

    paths = [build_year_cache(year, force=force_years) for year in YEARS]
    parts = [xr.open_dataset(path) for path in paths]
    try:
        merged = xr.concat(parts, dim="time", coords="minimal", compat="override").sortby("time").load()
    finally:
        for part in parts:
            part.close()
    times = pd.DatetimeIndex(merged.time.values)
    expected = 44 * 365
    if len(times) != expected:
        raise ValueError(f"Combined raw cache should have {expected} days, found {len(times)}")
    expected_index = pd.date_range("1981-01-01", "2024-12-31", freq="D")
    expected_index = expected_index[~((expected_index.month == 2) & (expected_index.day == 29))]
    if not np.array_equal(times.values, expected_index.values):
        missing = expected_index.difference(times)
        extra = times.difference(expected_index)
        raise ValueError(f"Combined dates mismatch; missing={missing[:5]}, extra={extra[:5]}")
    DAILY_RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.attrs.update(
        {
            "title": "ERA5 daily regional T and geopotential-height means",
            "period": "1981-2024",
            "area": "50-60N, 180-200E",
            "calendar": "no-leap",
            "processing": "cosine-latitude area mean; z divided by 9.80665",
        }
    )
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in merged.data_vars}
    merged.to_netcdf(DAILY_RAW_CACHE, encoding=encoding)
    merged.close()
    print(f"[RAW SAVED] {DAILY_RAW_CACHE}", flush=True)
    return xr.open_dataset(DAILY_RAW_CACHE)


def linear_detrend(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(values.shape[0], dtype=np.float64)
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)
    mean = values.mean(axis=0)
    slope = np.sum(x_centered[:, None] * (values - mean[None, :]), axis=0) / denom
    intercept = mean - slope * x.mean()
    residual = values - (x[:, None] * slope[None, :] + intercept[None, :])
    return residual.astype(np.float32), slope.astype(np.float64), intercept.astype(np.float64)


def build_anomaly_cache(raw: xr.Dataset, force: bool = False) -> xr.Dataset:
    if DAILY_ANOM_CACHE.exists() and not force:
        print(f"[ANOM CACHE] {DAILY_ANOM_CACHE}", flush=True)
        return xr.open_dataset(DAILY_ANOM_CACHE)

    times = pd.DatetimeIndex(raw.time.values)
    doy = np.array([calc_doy_nl(ts) for ts in times], dtype=np.int16)
    baseline = (times >= pd.Timestamp("1981-01-01")) & (times <= pd.Timestamp("2020-12-31"))
    data_vars = {}
    trend_vars = {}
    clim_vars = {}
    for source_name, short in [("temperature", "temperature"), ("geopotential_height", "geopotential_height")]:
        values = raw[source_name].values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Raw {source_name} contains NaN/Inf")
        residual, slope, intercept = linear_detrend(values)
        climatology = np.full((365, len(LEVELS)), np.nan, dtype=np.float32)
        counts = np.zeros(365, dtype=np.int16)
        for day in range(1, 366):
            select = baseline & (doy == day)
            counts[day - 1] = int(select.sum())
            climatology[day - 1] = residual[select].mean(axis=0)
        if not np.all(counts == 40):
            raise ValueError(f"Unexpected climatology counts for {source_name}: {np.unique(counts)}")
        anomaly = residual - climatology[doy - 1]
        data_vars[f"{short}_detrended"] = (("time", "level"), residual)
        data_vars[f"{short}_anomaly"] = (("time", "level"), anomaly.astype(np.float32))
        trend_vars[f"{short}_linear_slope_per_day"] = (("level",), slope)
        trend_vars[f"{short}_linear_intercept"] = (("level",), intercept)
        clim_vars[f"{short}_climatology"] = (("doy_nl", "level"), climatology)

    out = xr.Dataset(
        data_vars={
            **data_vars,
            **trend_vars,
            **clim_vars,
            "climatology_sample_count": (("doy_nl",), np.full(365, 40, dtype=np.int16)),
        },
        coords={
            "time": raw.time.values,
            "level": LEVELS,
            "doy_nl": np.arange(1, 366, dtype=np.int16),
            "time_doy_nl": (("time",), doy),
        },
        attrs={
            "title": "Detrended ERA5 regional T and geopotential-height daily anomalies",
            "source": str(DAILY_RAW_CACHE),
            "period": "1981-2024",
            "climatology_period": "1981-2020",
            "detrending": "linear trend removed independently for each variable and pressure level over 1981-2024",
            "calendar": "no-leap",
            "area": "50-60N, 180-200E",
        },
    )
    out.temperature_anomaly.attrs.update({"units": "K", "long_name": "Daily temperature anomaly"})
    out.geopotential_height_anomaly.attrs.update(
        {"units": "gpm", "long_name": "Daily geopotential-height anomaly"}
    )
    encoding = {}
    for name, variable in out.data_vars.items():
        if np.issubdtype(variable.dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        else:
            encoding[name] = {"zlib": True, "complevel": 4}
    DAILY_ANOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(DAILY_ANOM_CACHE, encoding=encoding)
    out.close()
    print(f"[ANOM SAVED] {DAILY_ANOM_CACHE}", flush=True)
    return xr.open_dataset(DAILY_ANOM_CACHE)


def build_event_dataset(anom: xr.Dataset) -> xr.Dataset:
    time_index = pd.DatetimeIndex(anom.time.values)
    time_to_index = {pd.Timestamp(ts): i for i, ts in enumerate(time_index)}
    n_event = len(PEAK_DATES)
    n_day = len(RELATIVE_DAYS)
    n_level = len(LEVELS)
    temperature = np.full((n_event, n_day, n_level), np.nan, dtype=np.float32)
    height = np.full_like(temperature, np.nan)
    target_date = np.empty((n_event, n_day), dtype="datetime64[ns]")
    window_start = np.empty_like(target_date)
    window_end = np.empty_like(target_date)

    t_daily = anom.temperature_anomaly.values
    z_daily = anom.geopotential_height_anomaly.values
    for event, peak in enumerate(PEAK_DATES):
        for day_i, rel_day in enumerate(RELATIVE_DAYS):
            center = pd.Timestamp(peak) + pd.Timedelta(days=int(rel_day))
            dates = [center + pd.Timedelta(days=int(offset)) for offset in ROLL_OFFSETS]
            missing = [str(date.date()) for date in dates if date not in time_to_index]
            if missing:
                raise ValueError(f"Missing event window dates for {peak.date()} day {rel_day}: {missing}")
            indices = [time_to_index[date] for date in dates]
            temperature[event, day_i] = t_daily[indices].mean(axis=0)
            height[event, day_i] = z_daily[indices].mean(axis=0)
            target_date[event, day_i] = center.to_datetime64()
            window_start[event, day_i] = dates[0].to_datetime64()
            window_end[event, day_i] = dates[-1].to_datetime64()

    groups = np.array(["group13_N13", "special2004_N1"], dtype=str)
    group_t = np.stack([temperature[REGULAR_INDICES].mean(axis=0), temperature[SPECIAL_INDEX]], axis=0)
    group_z = np.stack([height[REGULAR_INDICES].mean(axis=0), height[SPECIAL_INDEX]], axis=0)
    event_group = np.array(
        ["special2004_N1" if i == SPECIAL_INDEX else "group13_N13" for i in range(n_event)], dtype=str
    )
    out = xr.Dataset(
        data_vars={
            "temperature_anomaly_5dmean_event": (("event", "relative_day", "level"), temperature),
            "geopotential_height_anomaly_5dmean_event": (("event", "relative_day", "level"), height),
            "temperature_anomaly_5dmean_group": (("group", "relative_day", "level"), group_t),
            "geopotential_height_anomaly_5dmean_group": (("group", "relative_day", "level"), group_z),
            "sample_count": (("group",), np.array([13, 1], dtype=np.int16)),
            "target_date": (("event", "relative_day"), target_date),
            "window_start": (("event", "relative_day"), window_start),
            "window_end": (("event", "relative_day"), window_end),
        },
        coords={
            "event": np.arange(1, n_event + 1, dtype=np.int16),
            "peak_date": (("event",), PEAK_DATES.values.astype("datetime64[ns]")),
            "event_group": (("event",), event_group),
            "group": groups,
            "relative_day": RELATIVE_DAYS,
            "level": LEVELS,
        },
        attrs={
            "title": "ERA5 temperature and geopotential-height anomalies around MHW peaks",
            "area": "50-60N, 180-200E",
            "spatial_average": "cosine-latitude weighted mean",
            "temporal_average": "centered 5-day mean of daily anomalies",
            "detrending_period": "1981-2024",
            "climatology_period": "1981-2020",
            "calendar": "no-leap",
            "group13_definition": "all 14 events except 2004-12-24; includes 2020-11-13",
            "special_definition": "2004-12-24 only",
            "significance": "not calculated or plotted",
            "source_anomaly_cache": str(DAILY_ANOM_CACHE),
        },
    )
    out.temperature_anomaly_5dmean_event.attrs.update({"units": "K"})
    out.temperature_anomaly_5dmean_group.attrs.update({"units": "K"})
    out.geopotential_height_anomaly_5dmean_event.attrs.update({"units": "gpm"})
    out.geopotential_height_anomaly_5dmean_group.attrs.update({"units": "gpm"})
    return out


def save_event_dataset(ds: xr.Dataset) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, variable in ds.data_vars.items():
        if np.issubdtype(variable.dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        elif np.issubdtype(variable.dtype, np.integer):
            encoding[name] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(OUT_NC, encoding=encoding)
    print(f"[NC] {OUT_NC}", flush=True)


def rounded_temperature_limit(values: np.ndarray) -> float:
    robust = float(np.nanpercentile(np.abs(values), 98.0))
    return max(1.0, float(np.ceil(robust * 2.0) / 2.0))


def geopotential_contour_levels(values: np.ndarray) -> np.ndarray:
    robust = float(np.nanpercentile(np.abs(values), 98.0))
    vmax = max(20.0, float(np.ceil(robust / 20.0) * 20.0))
    levels = np.arange(-vmax, vmax + 20.0, 20.0)
    return levels[~np.isclose(levels, 0.0)]


def plot_sections(ds: xr.Dataset) -> tuple[float, np.ndarray]:
    group_t = ds.temperature_anomaly_5dmean_group.values
    group_z = ds.geopotential_height_anomaly_5dmean_group.values
    t_limit = rounded_temperature_limit(group_t)
    t_levels = np.arange(-t_limit, t_limit + 0.25, 0.5)
    if t_levels[-1] < t_limit - 1.0e-9:
        t_levels = np.append(t_levels, t_limit)
    z_levels = geopotential_contour_levels(group_z)

    fig, axes = plt.subplots(2, 1, figsize=(9.1, 7.3), sharex=True, sharey=True)
    panel_titles = ["(a) N = 13 events", "(b) 2004-12-24"]
    last_fill = None
    for group_i, ax in enumerate(axes):
        t_field = group_t[group_i].T
        z_field = group_z[group_i].T
        last_fill = ax.contourf(
            RELATIVE_DAYS,
            LEVELS,
            t_field,
            levels=t_levels,
            cmap="RdBu_r",
            extend="both",
        )
        negative = z_levels[z_levels < 0]
        positive = z_levels[z_levels > 0]
        contour_sets = []
        if len(negative):
            contour_sets.append(
                ax.contour(
                    RELATIVE_DAYS,
                    LEVELS,
                    z_field,
                    levels=negative,
                    colors="black",
                    linewidths=0.8,
                    linestyles="dashed",
                )
            )
        if len(positive):
            contour_sets.append(
                ax.contour(
                    RELATIVE_DAYS,
                    LEVELS,
                    z_field,
                    levels=positive,
                    colors="black",
                    linewidths=0.8,
                    linestyles="solid",
                )
            )
        zero = ax.contour(
            RELATIVE_DAYS,
            LEVELS,
            z_field,
            levels=[0.0],
            colors="black",
            linewidths=1.45,
            linestyles="solid",
        )
        for contours in contour_sets:
            ax.clabel(contours, inline=True, inline_spacing=2, fmt="%d", fontsize=7.0)
        ax.clabel(zero, inline=True, inline_spacing=2, fmt="%d", fontsize=7.0)
        ax.axvline(0, color="0.25", linewidth=1.0, linestyle="-", alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylim(1000, 100)
        ax.set_xlim(-10, 10)
        ax.set_title(panel_titles[group_i], loc="left", fontsize=11.5, fontweight="bold", pad=5)
        ax.set_title("50-60N, 180-160W", loc="right", fontsize=10.0, pad=5)
        ax.set_ylabel("Pressure (hPa)", fontsize=10.5)
        ax.yaxis.set_major_locator(FixedLocator(LEVELS.astype(float)))
        ax.yaxis.set_major_formatter(FixedFormatter([str(int(x)) for x in LEVELS]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="x", color="0.78", linewidth=0.45, linestyle=":", alpha=0.7)
        ax.tick_params(axis="both", labelsize=8.7, length=3.5, width=0.8)
        for level, label in zip(LEVELS, ax.get_yticklabels()):
            if level in (850, 925, 1000):
                label.set_fontsize(7.5)
            vertical_shift = {850: 3.0, 1000: -3.0}.get(int(level), 0.0)
            if vertical_shift:
                label.set_transform(
                    label.get_transform()
                    + ScaledTranslation(
                        0.0, vertical_shift / 72.0, fig.dpi_scale_trans
                    )
                )

    axes[-1].xaxis.set_major_locator(FixedLocator([-10, -5, 0, 5, 10]))
    axes[-1].xaxis.set_major_formatter(
        FixedFormatter(["lead 10", "lead 5", "peak", "lag 5", "lag 10"])
    )
    axes[-1].set_xlabel("Time relative to MHW peak (days)", fontsize=10.5, labelpad=7)
    fig.suptitle(
        "Temperature and geopotential height anomalies around MHW peak",
        fontsize=14.0,
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        "Centered 5-day mean; shading: temperature (K); contours: geopotential height (gpm)",
        ha="center",
        va="top",
        fontsize=9.2,
        color="0.25",
    )
    fig.subplots_adjust(left=0.105, right=0.965, top=0.91, bottom=0.16, hspace=0.19)
    if last_fill is None:
        raise RuntimeError("No temperature field was plotted")
    cbar = fig.colorbar(
        last_fill,
        ax=axes,
        orientation="horizontal",
        fraction=0.045,
        pad=0.095,
        aspect=40,
    )
    cbar.set_label("Temperature anomaly (K)", fontsize=10.0, labelpad=4)
    cbar.ax.tick_params(labelsize=8.5, length=3)
    cbar.outline.set_linewidth(0.7)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PNG] {OUT_PNG}", flush=True)
    print(f"[PDF] {OUT_PDF}", flush=True)
    return t_limit, z_levels


def compare_z500(anom: xr.Dataset) -> list[dict[str, object]]:
    checks = []
    if not Z500_ANOM_FILE.exists():
        return checks
    dates = pd.to_datetime(["1985-11-29", "2004-12-24", "2015-11-06", "2020-11-13"])
    with xr.open_dataset(Z500_ANOM_FILE) as ds:
        z500 = ds["z"].sel(time=dates, lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        weights = np.cos(np.deg2rad(z500.lat))
        reference = z500.weighted(weights).mean(("lat", "lon"), skipna=True).load()
    current = anom.geopotential_height_anomaly.sel(time=dates, level=500).load()
    for date, ref, new in zip(dates, reference.values, current.values):
        checks.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "existing_z500_gpm": float(ref),
                "new_multilevel_z500_gpm": float(new),
                "difference_gpm": float(new - ref),
            }
        )
    return checks


def write_check(ds: xr.Dataset, anom: xr.Dataset, t_limit: float, z_levels: np.ndarray, z500_checks: list[dict[str, object]]) -> None:
    lines = [
        "ERA5 temperature-geopotential-height time-pressure section check",
        "=" * 78,
        f"Output NetCDF: {OUT_NC}",
        f"Output PNG: {OUT_PNG}",
        f"Output PDF: {OUT_PDF}",
        f"Raw daily cache: {DAILY_RAW_CACHE}",
        f"Anomaly cache: {DAILY_ANOM_CACHE}",
        "",
        "Configuration",
        f"  Region: {LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
        f"  Levels (hPa): {LEVELS.tolist()}",
        f"  Relative days: {RELATIVE_DAYS.tolist()}",
        f"  Running mean offsets: {ROLL_OFFSETS.tolist()}",
        "  Detrending: linear, independently by variable and pressure level, 1981-2024",
        "  Climatology: 1981-2020 no-leap daily climatology",
        "  Significance: not calculated or plotted",
        "",
        "Dimensions and counts",
        f"  Daily anomaly days: {anom.sizes['time']}",
        f"  Climatology days: {anom.sizes['doy_nl']}",
        f"  Climatology sample count unique: {np.unique(anom.climatology_sample_count.values).tolist()}",
        f"  Events: {ds.sizes['event']}",
        f"  Groups: {ds.group.values.tolist()}",
        f"  Sample count: {ds.sample_count.values.tolist()}",
        "",
        "Plot levels",
        f"  Temperature symmetric limit: +/- {t_limit:.2f} K",
        f"  Geopotential-height contour levels: {z_levels.tolist()} gpm",
        "",
        "Variable statistics",
    ]
    for name in [
        "temperature_anomaly_5dmean_event",
        "geopotential_height_anomaly_5dmean_event",
        "temperature_anomaly_5dmean_group",
        "geopotential_height_anomaly_5dmean_group",
    ]:
        values = ds[name].values
        lines.append(
            f"  {name}: shape={values.shape}, min={np.nanmin(values):.6f}, "
            f"max={np.nanmax(values):.6f}, mean={np.nanmean(values):.6f}, "
            f"nan={int(np.isnan(values).sum())}, inf={int(np.isinf(values).sum())}"
        )
    lines.extend(["", "500 hPa comparison against existing detrended Z500 anomaly"])
    if z500_checks:
        for row in z500_checks:
            lines.append(
                f"  {row['date']}: existing={row['existing_z500_gpm']:.6f}, "
                f"new={row['new_multilevel_z500_gpm']:.6f}, "
                f"difference={row['difference_gpm']:.6f} gpm"
            )
    else:
        lines.append("  Existing Z500 anomaly file was unavailable; comparison skipped.")
    lines.extend(
        [
            "",
            "Event windows",
            "  Source anomaly dates required by final plot: relative day -12 to +12.",
            "  Final plotted centers: relative day -10 to +10.",
        ]
    )
    for event in ds.event.values:
        selection = ds.sel(event=event)
        lines.append(
            f"  event={int(event):02d}, peak={pd.Timestamp(selection.peak_date.item()).strftime('%Y-%m-%d')}, "
            f"group={selection.event_group.item()}, "
            f"first_window={pd.Timestamp(selection.window_start.isel(relative_day=0).item()).strftime('%Y-%m-%d')}.."
            f"{pd.Timestamp(selection.window_end.isel(relative_day=0).item()).strftime('%Y-%m-%d')}, "
            f"last_window={pd.Timestamp(selection.window_start.isel(relative_day=-1).item()).strftime('%Y-%m-%d')}.."
            f"{pd.Timestamp(selection.window_end.isel(relative_day=-1).item()).strftime('%Y-%m-%d')}"
        )
    OUT_CHECK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[CHECK] {OUT_CHECK}", flush=True)


def validate_final(ds: xr.Dataset) -> None:
    if not np.array_equal(ds.relative_day.values.astype(np.int16), RELATIVE_DAYS):
        raise ValueError("Final relative_day coordinate is incorrect")
    if not np.array_equal(ds.level.values.astype(np.int16), LEVELS):
        raise ValueError("Final pressure levels are incorrect")
    if ds.sample_count.values.tolist() != [13, 1]:
        raise ValueError(f"Final sample counts are incorrect: {ds.sample_count.values}")
    for name in [
        "temperature_anomaly_5dmean_event",
        "geopotential_height_anomaly_5dmean_event",
        "temperature_anomaly_5dmean_group",
        "geopotential_height_anomaly_5dmean_group",
    ]:
        if not np.isfinite(ds[name].values).all():
            raise ValueError(f"Final variable contains NaN/Inf: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-year-cache", action="store_true", help="Rebuild all yearly regional caches")
    parser.add_argument("--force-combined-cache", action="store_true", help="Rebuild combined raw/anomaly caches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 100, flush=True)
    print("ERA5 T/Z TIME-PRESSURE SECTION: N=13 PLUS 2004", flush=True)
    print("=" * 100, flush=True)

    raw = build_combined_raw_cache(
        force=args.force_combined_cache,
        force_years=args.force_year_cache,
    )
    try:
        anomaly = build_anomaly_cache(raw, force=args.force_combined_cache)
    finally:
        raw.close()

    try:
        event_ds = build_event_dataset(anomaly)
        validate_final(event_ds)
        save_event_dataset(event_ds)
        t_limit, z_levels = plot_sections(event_ds)
        z500_checks = compare_z500(anomaly)
        write_check(event_ds, anomaly, t_limit, z_levels, z500_checks)
        event_ds.close()
    finally:
        anomaly.close()

    print("[DONE] All outputs passed numerical checks.", flush=True)


if __name__ == "__main__":
    main()
