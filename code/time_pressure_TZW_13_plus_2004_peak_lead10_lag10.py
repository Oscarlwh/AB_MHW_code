#!/usr/bin/env python3
"""Build professional ERA5 T-Z-omega time-pressure sections for N=13 plus 2004.

Temperature and geopotential-height anomalies are reused from the established
T/Z workflow. Pressure vertical velocity (omega) is processed with the same
1981-2024 detrending, 1981-2020 no-leap climatology, and centered 5-day mean.
The native 21-day by 12-level values are saved; shape-preserving interpolation
is used only to render smoother shading and contours.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import xarray as xr

import time_pressure_TZ_13_plus_2004_peak_lead10_lag10 as base


ERA5_ROOT = Path("/path/to/data/ERA5")
W_SOURCE_ROOT = Path("/path/to/data/ERA5_raw/precipitation/23-24_w")
W_GRIB_SOURCE = W_SOURCE_ROOT / "c4fbf7fe473f129712aaa4245ee025a6.grib"
LOCAL_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "time_pressure_TZW"
W_GRIB_AREA_CACHE = (
    LOCAL_CACHE_ROOT / "ERA5_w_area_mean_2023_2024_180E200E_50N60N.nc"
)

OUT_ROOT = Path(
    "/path/to/data/ABH_170E240E_1981_2024_5N90N/composite/"
    "time_pressure_TZW_13_plus_2004_lead10_lag10"
)
CACHE_ROOT = OUT_ROOT / "cache"
YEARLY_W_CACHE_ROOT = CACHE_ROOT / "yearly_w_area_mean"
W_DAILY_RAW_CACHE = CACHE_ROOT / (
    "ERA5_w_daily_area_mean_1981_2024_180E200E_50N60N_"
    "1000_100hPa_no_leap.nc"
)
W_DAILY_ANOM_CACHE = CACHE_ROOT / (
    "ERA5_w_daily_area_mean_anomaly_1981_2024_vs1981_2020_"
    "180E200E_50N60N_1000_100hPa_detrended_no_leap.nc"
)

BASE_NAME = (
    "ERA5_TZW_anomaly_time_pressure_group13_plus_2004_peak_"
    "lead10_lag10_5dmean_180E200E_50N60N"
)
OUT_NC = OUT_ROOT / f"{BASE_NAME}.nc"
OUT_PNG = OUT_ROOT / f"{BASE_NAME}.png"
OUT_PDF = OUT_ROOT / f"{BASE_NAME}.pdf"
OUT_CHECK = OUT_ROOT / (
    "check_ERA5_TZW_anomaly_time_pressure_group13_plus_2004.txt"
)

YEARS = list(range(1981, 2025))
LEVELS = base.LEVELS
RELATIVE_DAYS = base.RELATIVE_DAYS
ROLL_OFFSETS = base.ROLL_OFFSETS
PEAK_DATES = base.PEAK_DATES
REGULAR_INDICES = base.REGULAR_INDICES
SPECIAL_INDEX = base.SPECIAL_INDEX
LAT_MIN, LAT_MAX = base.LAT_MIN, base.LAT_MAX
LON_MIN, LON_MAX = base.LON_MIN, base.LON_MAX

# Optional publication-layout overrides. Region-specific entry points can
# change these without altering the default appearance of earlier products.
PANEL_RIGHT_LABEL: str | None = None
SHOW_FIGURE_HEADER = True
SHOW_ENCODING_LEGEND = True
FIGURE_SIZE = (9.4, 8.7)


def format_longitude(value: float) -> str:
    """Format a 0-360 longitude for compact plot annotations."""
    longitude = value % 360.0
    if np.isclose(longitude, 0.0):
        return "0°"
    if np.isclose(longitude, 180.0):
        return "180°"
    if longitude < 180.0:
        return f"{longitude:g}°E"
    return f"{360.0 - longitude:g}°W"


def format_longitude_range(start: float, end: float) -> str:
    """Format a longitude range without repeating a shared hemisphere."""
    start_lon = start % 360.0
    end_lon = end % 360.0
    if start_lon < 180.0 and end_lon < 180.0:
        return f"{start_lon:g}°–{end_lon:g}°E"
    if start_lon > 180.0 and end_lon > 180.0:
        return f"{360.0 - start_lon:g}°–{360.0 - end_lon:g}°W"
    return f"{format_longitude(start)}–{format_longitude(end)}"


def expected_no_leap_index(year: int) -> pd.DatetimeIndex:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return dates[~((dates.month == 2) & (dates.day == 29))]


def validate_w_year(ds: xr.Dataset, year: int, context: str) -> None:
    if "vertical_velocity" not in ds:
        raise ValueError(f"{context}: missing vertical_velocity")
    if not np.array_equal(ds.level.values.astype(np.int16), LEVELS):
        raise ValueError(f"{context}: unexpected levels {ds.level.values}")
    times = pd.DatetimeIndex(ds.time.values).normalize()
    expected = expected_no_leap_index(year)
    if not np.array_equal(times.values, expected.values):
        missing = expected.difference(times)
        extra = times.difference(expected)
        raise ValueError(
            f"{context}: date mismatch; missing={missing[:8].tolist()}, "
            f"extra={extra[:8].tolist()}"
        )
    values = ds.vertical_velocity.values
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: vertical_velocity contains NaN/Inf")


def extract_local_w_year(year: int, source: Path) -> xr.Dataset:
    print(f"[EXTRACT W] {year}: {source}", flush=True)
    with nc.Dataset(source) as root:
        for variable in ["w", "time", "lev", "lat", "lon"]:
            if variable not in root.variables:
                raise KeyError(f"{source}: missing {variable}")
        units = str(getattr(root.variables["w"], "units", ""))
        if "Pa" not in units:
            raise ValueError(f"{source}: unexpected w units {units!r}")

        times = base.date_strings_from_nc(root.variables["time"])
        lev_idx = base.level_indices(root.variables["lev"][:])
        lat_all = np.asarray(root.variables["lat"][:], dtype=float)
        lon_all = np.asarray(root.variables["lon"][:], dtype=float) % 360.0
        lat_idx = base.coordinate_indices(lat_all, LAT_MIN, LAT_MAX)
        lon_idx = base.coordinate_indices(lon_all, LON_MIN, LON_MAX)
        if not len(lat_idx) or not len(lon_idx):
            raise ValueError(f"{source}: empty regional selection")
        if not np.all(np.diff(lat_idx) == 1) or not np.all(np.diff(lon_idx) == 1):
            raise ValueError(f"{source}: regional indices are not contiguous")

        lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
        lon_slice = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
        lat_region = lat_all[lat_idx]
        daily_w = np.full((len(times), len(LEVELS)), np.nan, dtype=np.float32)
        for start in range(0, len(times), 45):
            end = min(len(times), start + 45)
            block = root.variables["w"][
                start:end, lev_idx, lat_slice, lon_slice
            ]
            daily_w[start:end] = base.weighted_spatial_mean(block, lat_region)
            print(f"  {year}: days {start:03d}-{end - 1:03d}", flush=True)

    keep = ~((times.month == 2) & (times.day == 29))
    out = xr.Dataset(
        data_vars={
            "vertical_velocity": (
                ("time", "level"),
                daily_w[keep],
                {
                    "long_name": "Area-mean daily pressure vertical velocity",
                    "units": "Pa s-1",
                    "sign_convention": "negative ascent; positive descent",
                },
            )
        },
        coords={
            "time": times[keep].values.astype("datetime64[ns]"),
            "level": LEVELS,
        },
        attrs={
            "source_file": str(source),
            "area": f"{LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
            "spatial_average": "cosine-latitude weighted mean",
            "temporal_average": "source daily mean",
            "calendar": "no-leap",
        },
    )
    validate_w_year(out, year, str(source))
    return out


def run_cdo_area_mean(force: bool = False) -> Path:
    if W_GRIB_AREA_CACHE.exists() and not force:
        print(f"[W GRIB AREA CACHE] {W_GRIB_AREA_CACHE}", flush=True)
        return W_GRIB_AREA_CACHE
    if not W_GRIB_SOURCE.exists():
        raise FileNotFoundError(f"Missing 2023-2024 omega GRIB: {W_GRIB_SOURCE}")
    cdo = Path("/opt/homebrew/bin/cdo")
    if not cdo.exists():
        raise FileNotFoundError("CDO is required at /opt/homebrew/bin/cdo")

    LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = W_GRIB_AREA_CACHE.with_suffix(".nc.part")
    if tmp.exists():
        tmp.unlink()
    command = [
        str(cdo),
        "-L",
        "-f",
        "nc4",
        "-z",
        "zip_4",
        "fldmean",
        f"-sellonlatbox,{LON_MIN:g},{LON_MAX:g},{LAT_MIN:g},{LAT_MAX:g}",
        str(W_GRIB_SOURCE),
        str(tmp),
    ]
    print("[CDO] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    tmp.replace(W_GRIB_AREA_CACHE)
    return W_GRIB_AREA_CACHE


def normalize_cdo_w(path: Path) -> xr.Dataset:
    opened = xr.open_dataset(path)
    rename = {}
    aliases = {
        "valid_time": "time",
        "pressure_level": "level",
        "isobaricInhPa": "level",
        "plev": "level",
    }
    for old, new in aliases.items():
        if old in opened.dims or old in opened.coords:
            if new not in opened.dims and new not in opened.coords:
                rename[old] = new
    if "var135" in opened.data_vars and "w" not in opened.data_vars:
        rename["var135"] = "w"
    ds = opened.rename(rename)
    if "w" not in ds:
        opened.close()
        raise KeyError(f"{path}: expected w, found {list(ds.data_vars)}")
    if "level" not in ds.coords:
        opened.close()
        raise KeyError(f"{path}: no pressure-level coordinate")
    level_values = np.asarray(ds.level.values, dtype=float)
    if np.nanmax(level_values) > 2000.0:
        ds = ds.assign_coords(level=level_values / 100.0)

    field = ds.w
    for dim in list(field.dims):
        if dim not in {"time", "level"}:
            if field.sizes[dim] != 1:
                opened.close()
                raise ValueError(f"{path}: unexpected non-singleton dimension {dim}")
            field = field.isel({dim: 0}, drop=True)
    field = field.sel(level=LEVELS)
    result = field.to_dataset(name="w").load()
    opened.close()
    return result


def extract_grib_w_year(year: int, force_area_cache: bool = False) -> xr.Dataset:
    area_path = run_cdo_area_mean(force=force_area_cache)
    ds = normalize_cdo_w(area_path)
    if pd.DatetimeIndex(ds.time.values).has_duplicates:
        ds = ds.groupby("time").mean("time", skipna=True)
    times = pd.DatetimeIndex(ds.time.values)
    selected = ds.sel(time=str(year))
    daily = selected.resample(time="1D").mean("time", skipna=True)
    daily_times = pd.DatetimeIndex(daily.time.values).normalize()
    daily = daily.assign_coords(time=daily_times.values.astype("datetime64[ns]"))
    keep = ~((daily_times.month == 2) & (daily_times.day == 29))
    daily = daily.isel(time=keep)
    out = xr.Dataset(
        data_vars={
            "vertical_velocity": (
                ("time", "level"),
                daily.w.values.astype(np.float32),
                {
                    "long_name": "Area-mean daily pressure vertical velocity",
                    "units": "Pa s-1",
                    "sign_convention": "negative ascent; positive descent",
                },
            )
        },
        coords={"time": daily.time.values, "level": LEVELS},
        attrs={
            "source_file": str(W_GRIB_SOURCE),
            "area_mean_cache": str(area_path),
            "area": f"{LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
            "spatial_average": "CDO grid-cell-area weighted fldmean",
            "temporal_average": "mean of 00, 06, 12, and 18 UTC",
            "calendar": "no-leap",
        },
    )
    ds.close()
    daily.close()
    validate_w_year(out, year, f"{area_path} year {year}")
    return out


def w_year_cache_path(year: int) -> Path:
    return YEARLY_W_CACHE_ROOT / (
        f"ERA5_w_daily_area_mean_180E200E_50N60N_"
        f"1000_100hPa_{year}_no_leap.nc"
    )


def build_w_year_cache(
    year: int,
    force: bool = False,
    force_grib_area_cache: bool = False,
) -> Path:
    path = w_year_cache_path(year)
    if path.exists() and not force:
        with xr.open_dataset(path) as opened:
            validate_w_year(opened.load(), year, str(path))
        print(f"[W YEAR CACHE] {path}", flush=True)
        return path
    if path.exists():
        path.unlink()

    if year <= 2022:
        source = ERA5_ROOT / f"{year}.nc"
        if not source.exists():
            raise FileNotFoundError(f"Missing ERA5 annual source: {source}")
        ds = extract_local_w_year(year, source)
    else:
        ds = extract_grib_w_year(
            year,
            force_area_cache=force_grib_area_cache,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(
        path,
        encoding={
            "vertical_velocity": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
        },
    )
    ds.close()
    print(f"[W YEAR SAVED] {path}", flush=True)
    gc.collect()
    return path


def build_w_raw_cache(
    force: bool = False,
    force_years: bool = False,
    force_grib_area_cache: bool = False,
) -> xr.Dataset:
    if W_DAILY_RAW_CACHE.exists() and not force:
        print(f"[W RAW CACHE] {W_DAILY_RAW_CACHE}", flush=True)
        return xr.open_dataset(W_DAILY_RAW_CACHE)

    paths = [
        build_w_year_cache(
            year,
            force=force_years,
            force_grib_area_cache=force_grib_area_cache and year == 2023,
        )
        for year in YEARS
    ]
    parts = [xr.open_dataset(path) for path in paths]
    try:
        merged = xr.concat(
            parts, dim="time", coords="minimal", compat="override"
        ).sortby("time").load()
    finally:
        for part in parts:
            part.close()

    expected = pd.date_range("1981-01-01", "2024-12-31", freq="D")
    expected = expected[~((expected.month == 2) & (expected.day == 29))]
    actual = pd.DatetimeIndex(merged.time.values)
    if not np.array_equal(actual.values, expected.values):
        raise ValueError("Combined omega dates do not match the 16060-day calendar")
    merged.attrs.update(
        {
            "title": "ERA5 daily regional pressure vertical velocity mean",
            "period": "1981-2024",
            "area": f"{LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
            "calendar": "no-leap",
        }
    )
    W_DAILY_RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(
        W_DAILY_RAW_CACHE,
        encoding={
            "vertical_velocity": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
        },
    )
    merged.close()
    print(f"[W RAW SAVED] {W_DAILY_RAW_CACHE}", flush=True)
    return xr.open_dataset(W_DAILY_RAW_CACHE)


def build_w_anomaly_cache(raw: xr.Dataset, force: bool = False) -> xr.Dataset:
    if W_DAILY_ANOM_CACHE.exists() and not force:
        print(f"[W ANOM CACHE] {W_DAILY_ANOM_CACHE}", flush=True)
        return xr.open_dataset(W_DAILY_ANOM_CACHE)

    times = pd.DatetimeIndex(raw.time.values)
    doy = np.array([base.calc_doy_nl(ts) for ts in times], dtype=np.int16)
    baseline = (times >= "1981-01-01") & (times <= "2020-12-31")
    values = raw.vertical_velocity.values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Raw omega contains NaN/Inf")
    detrended, slope, intercept = base.linear_detrend(values)
    climatology = np.full((365, len(LEVELS)), np.nan, dtype=np.float32)
    counts = np.zeros(365, dtype=np.int16)
    for day in range(1, 366):
        select = baseline & (doy == day)
        counts[day - 1] = int(select.sum())
        climatology[day - 1] = detrended[select].mean(axis=0)
    if not np.all(counts == 40):
        raise ValueError(f"Unexpected omega climatology counts: {np.unique(counts)}")
    anomaly = detrended - climatology[doy - 1]

    out = xr.Dataset(
        data_vars={
            "vertical_velocity_detrended": (
                ("time", "level"),
                detrended,
                {"units": "Pa s-1"},
            ),
            "vertical_velocity_anomaly": (
                ("time", "level"),
                anomaly.astype(np.float32),
                {
                    "units": "Pa s-1",
                    "sign_convention": "negative ascent; positive descent",
                },
            ),
            "vertical_velocity_linear_slope_per_day": (("level",), slope),
            "vertical_velocity_linear_intercept": (("level",), intercept),
            "vertical_velocity_climatology": (
                ("doy_nl", "level"),
                climatology,
            ),
            "climatology_sample_count": (("doy_nl",), counts),
        },
        coords={
            "time": raw.time.values,
            "level": LEVELS,
            "doy_nl": np.arange(1, 366, dtype=np.int16),
            "time_doy_nl": (("time",), doy),
        },
        attrs={
            "title": "Detrended ERA5 regional pressure vertical velocity anomalies",
            "source": str(W_DAILY_RAW_CACHE),
            "period": "1981-2024",
            "climatology_period": "1981-2020",
            "detrending": "linear trend removed independently at each pressure level over 1981-2024",
            "calendar": "no-leap",
            "area": f"{LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
        },
    )
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        if np.issubdtype(var.dtype, np.floating)
        else {"zlib": True, "complevel": 4}
        for name, var in out.data_vars.items()
    }
    W_DAILY_ANOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(W_DAILY_ANOM_CACHE, encoding=encoding)
    out.close()
    print(f"[W ANOM SAVED] {W_DAILY_ANOM_CACHE}", flush=True)
    return xr.open_dataset(W_DAILY_ANOM_CACHE)


def add_vertical_velocity(
    event_ds: xr.Dataset,
    w_anom: xr.Dataset,
) -> xr.Dataset:
    time_index = pd.DatetimeIndex(w_anom.time.values)
    time_to_index = {pd.Timestamp(ts): i for i, ts in enumerate(time_index)}
    daily = w_anom.vertical_velocity_anomaly.values
    omega = np.full(
        (len(PEAK_DATES), len(RELATIVE_DAYS), len(LEVELS)),
        np.nan,
        dtype=np.float32,
    )
    for event, peak in enumerate(PEAK_DATES):
        for day_i, rel_day in enumerate(RELATIVE_DAYS):
            center = pd.Timestamp(peak) + pd.Timedelta(days=int(rel_day))
            dates = [
                center + pd.Timedelta(days=int(offset))
                for offset in ROLL_OFFSETS
            ]
            missing = [date for date in dates if date not in time_to_index]
            if missing:
                raise ValueError(
                    f"Missing omega event dates for {peak.date()} day {rel_day}: "
                    f"{missing}"
                )
            indices = [time_to_index[date] for date in dates]
            omega[event, day_i] = daily[indices].mean(axis=0)

    grouped = np.stack(
        [omega[REGULAR_INDICES].mean(axis=0), omega[SPECIAL_INDEX]],
        axis=0,
    )
    out = event_ds.copy()
    out["vertical_velocity_anomaly_5dmean_event"] = (
        ("event", "relative_day", "level"),
        omega,
    )
    out["vertical_velocity_anomaly_5dmean_group"] = (
        ("group", "relative_day", "level"),
        grouped,
    )
    for name in [
        "vertical_velocity_anomaly_5dmean_event",
        "vertical_velocity_anomaly_5dmean_group",
    ]:
        out[name].attrs.update(
            {
                "units": "Pa s-1",
                "long_name": "Centered 5-day mean pressure vertical velocity anomaly",
                "sign_convention": "negative ascent; positive descent",
            }
        )
    out.attrs.update(
        {
            "title": "ERA5 T, Z, and omega anomalies around MHW peaks",
            "area": f"{LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
            "omega_source": str(W_DAILY_ANOM_CACHE),
            "plot_interpolation": "PCHIP in relative time and log-pressure; display only",
        }
    )
    return out


def save_event_dataset(ds: xr.Dataset) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, variable in ds.data_vars.items():
        if np.issubdtype(variable.dtype, np.floating):
            encoding[name] = {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
        elif np.issubdtype(variable.dtype, np.integer):
            encoding[name] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(OUT_NC, encoding=encoding)
    print(f"[NC] {OUT_NC}", flush=True)


def pchip_display_grid(
    field_time_level: np.ndarray,
    n_time: int = 241,
    n_pressure: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(field_time_level, dtype=np.float64)
    dense_time = np.linspace(RELATIVE_DAYS[0], RELATIVE_DAYS[-1], n_time)
    time_interp = PchipInterpolator(
        RELATIVE_DAYS.astype(float),
        values,
        axis=0,
    )(dense_time)

    log_pressure = np.log(LEVELS.astype(float))
    order = np.argsort(log_pressure)
    dense_log_pressure = np.linspace(
        log_pressure[order][0],
        log_pressure[order][-1],
        n_pressure,
    )
    pressure_interp = PchipInterpolator(
        log_pressure[order],
        time_interp[:, order],
        axis=1,
    )(dense_log_pressure)
    dense_pressure = np.exp(dense_log_pressure)
    return dense_time, dense_pressure, pressure_interp.T


def geopotential_levels(
    values: np.ndarray,
    interval: float = 40.0,
) -> np.ndarray:
    robust = float(np.nanpercentile(np.abs(values), 98.0))
    vmax = max(interval, float(np.ceil(robust / interval) * interval))
    levels = np.arange(-vmax, vmax + interval, interval)
    return levels[~np.isclose(levels, 0.0)]


def temperature_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "temperature_anomaly",
        [
            "#274f73",
            "#4f81a7",
            "#8eb7cf",
            "#d4e2e7",
            "#f7f6f2",
            "#eed7cd",
            "#d99b88",
            "#b85e50",
            "#823b3b",
        ],
        N=256,
    )


ASCENT_HATCH_COLOR = "#00788a"
DESCENT_HATCH_COLOR = "#b46700"
ASCENT_HATCH = "/" * 6
DESCENT_HATCH = "\\" * 6


def add_omega_core_hatching(
    ax: plt.Axes,
    dense_time: np.ndarray,
    dense_pressure: np.ndarray,
    smooth_omega: np.ndarray,
    native_omega: np.ndarray,
) -> tuple[float, float]:
    negative = native_omega[native_omega < 0.0]
    positive = native_omega[native_omega > 0.0]
    if not len(negative) or not len(positive):
        raise ValueError("Omega field must contain both ascent and descent anomalies")
    ascent_threshold = float(np.nanpercentile(negative, 15.0))
    descent_threshold = float(np.nanpercentile(positive, 85.0))

    lower = float(np.nanmin(smooth_omega))
    upper = float(np.nanmax(smooth_omega))
    lower_margin = max(1.0e-9, abs(lower) * 1.0e-6)
    upper_margin = max(1.0e-9, abs(upper) * 1.0e-6)
    ascent = ax.contourf(
        dense_time,
        dense_pressure,
        smooth_omega,
        levels=[lower - lower_margin, ascent_threshold],
        colors=[to_rgba(ASCENT_HATCH_COLOR, 0.13)],
        hatches=[ASCENT_HATCH],
        zorder=3.0,
    )
    ascent.set_edgecolor(ASCENT_HATCH_COLOR)
    ascent.set_linewidth(0.0)
    ascent.set_hatch_linewidth(1.05)
    ax.contour(
        dense_time,
        dense_pressure,
        smooth_omega,
        levels=[ascent_threshold],
        colors=ASCENT_HATCH_COLOR,
        linewidths=1.0,
        zorder=3.2,
    )

    descent = ax.contourf(
        dense_time,
        dense_pressure,
        smooth_omega,
        levels=[descent_threshold, upper + upper_margin],
        colors=[to_rgba(DESCENT_HATCH_COLOR, 0.13)],
        hatches=[DESCENT_HATCH],
        zorder=3.0,
    )
    descent.set_edgecolor(DESCENT_HATCH_COLOR)
    descent.set_linewidth(0.0)
    descent.set_hatch_linewidth(1.05)
    ax.contour(
        dense_time,
        dense_pressure,
        smooth_omega,
        levels=[descent_threshold],
        colors=DESCENT_HATCH_COLOR,
        linewidths=1.0,
        zorder=3.2,
    )
    return ascent_threshold, descent_threshold


def plot_sections(
    ds: xr.Dataset,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    group_t = ds.temperature_anomaly_5dmean_group.values
    group_z = ds.geopotential_height_anomaly_5dmean_group.values
    group_w = ds.vertical_velocity_anomaly_5dmean_group.values
    t_limit = base.rounded_temperature_limit(group_t)
    t_levels = np.linspace(-t_limit, t_limit, 25)
    z_levels_top = geopotential_levels(group_z[0], interval=20.0)
    z_levels_bottom = geopotential_levels(group_z[1], interval=40.0)
    z_levels_by_group = [z_levels_top, z_levels_bottom]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#30343a",
            "xtick.color": "#30343a",
            "ytick.color": "#30343a",
            "text.color": "#22262b",
            "axes.labelcolor": "#22262b",
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
    )
    panel_titles = ["(a) N = 13 events", "(b) 2004-12-24"]
    last_fill = None
    smoothing_stats: dict[str, float] = {}

    for group_i, ax in enumerate(axes):
        td_t, pd_t, smooth_t = pchip_display_grid(group_t[group_i])
        td_z, pd_z, smooth_z = pchip_display_grid(group_z[group_i])
        td_w, pd_w, smooth_w = pchip_display_grid(group_w[group_i])
        if not np.array_equal(td_t, td_z) or not np.array_equal(td_t, td_w):
            raise RuntimeError("Display time grids differ")
        if not np.array_equal(pd_t, pd_z) or not np.array_equal(pd_t, pd_w):
            raise RuntimeError("Display pressure grids differ")

        smoothing_stats[f"group{group_i}_w_native_min"] = float(np.min(group_w[group_i]))
        smoothing_stats[f"group{group_i}_w_native_max"] = float(np.max(group_w[group_i]))
        smoothing_stats[f"group{group_i}_w_smooth_min"] = float(np.min(smooth_w))
        smoothing_stats[f"group{group_i}_w_smooth_max"] = float(np.max(smooth_w))

        last_fill = ax.contourf(
            td_t,
            pd_t,
            smooth_t,
            levels=t_levels,
            cmap=temperature_colormap(),
            extend="both",
            antialiased=True,
            zorder=1.0,
        )
        ascent_threshold, descent_threshold = add_omega_core_hatching(
            ax,
            td_w,
            pd_w,
            smooth_w,
            group_w[group_i],
        )
        smoothing_stats[f"group{group_i}_omega_ascent_core_threshold"] = (
            ascent_threshold
        )
        smoothing_stats[f"group{group_i}_omega_descent_core_threshold"] = (
            descent_threshold
        )

        z_levels = z_levels_by_group[group_i]
        z_label_interval = 40.0 if group_i == 0 else 80.0
        z_negative = z_levels[z_levels < 0]
        z_positive = z_levels[z_levels > 0]
        if len(z_negative):
            zc_neg = ax.contour(
                td_z,
                pd_z,
                smooth_z,
                levels=z_negative,
                colors="#252a30",
                linewidths=0.9,
                linestyles="dashed",
                zorder=4.0,
            )
            negative_labels = z_negative[
                np.isclose(
                    np.mod(np.abs(z_negative), z_label_interval),
                    0.0,
                )
            ]
            if len(negative_labels):
                ax.clabel(
                    zc_neg,
                    levels=negative_labels,
                    fmt="%d",
                    fontsize=6.8,
                    inline_spacing=3,
                )
        if len(z_positive):
            zc_pos = ax.contour(
                td_z,
                pd_z,
                smooth_z,
                levels=z_positive,
                colors="#252a30",
                linewidths=0.9,
                linestyles="solid",
                zorder=4.0,
            )
            positive_labels = z_positive[
                np.isclose(
                    np.mod(np.abs(z_positive), z_label_interval),
                    0.0,
                )
            ]
            if len(positive_labels):
                ax.clabel(
                    zc_pos,
                    levels=positive_labels,
                    fmt="%d",
                    fontsize=6.8,
                    inline_spacing=3,
                )
        ax.contour(
            td_z,
            pd_z,
            smooth_z,
            levels=[0.0],
            colors="#15191d",
            linewidths=1.35,
            linestyles="solid",
            zorder=4.0,
        )
        ax.axvline(0, color="#555b62", linewidth=0.9, alpha=0.9, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(1000, 100)
        ax.set_xlim(-10, 10)
        ax.set_title(
            panel_titles[group_i],
            loc="left",
            fontsize=11.0,
            fontweight="semibold",
            pad=7,
        )
        region_label = PANEL_RIGHT_LABEL or (
            f"{LAT_MIN:g}°–{LAT_MAX:g}°N, "
            f"{format_longitude_range(LON_MIN, LON_MAX)}"
        )
        ax.set_title(
            region_label,
            loc="right",
            fontsize=9.2,
            fontweight="normal",
            color="#22262b",
            pad=7,
        )
        ax.set_ylabel("Pressure (hPa)", fontsize=9.8, labelpad=8)
        ax.yaxis.set_major_locator(FixedLocator(LEVELS.astype(float)))
        ax.yaxis.set_major_formatter(
            FixedFormatter([str(int(level)) for level in LEVELS])
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(
            axis="x",
            color="#b7bcc2",
            linewidth=0.45,
            linestyle=":",
            alpha=0.65,
        )
        ax.tick_params(
            axis="both",
            labelsize=8.2,
            length=3.2,
            width=0.75,
            pad=4,
        )
        for spine in ax.spines.values():
            spine.set_color("#30343a")
            spine.set_linewidth(0.8)

    axes[-1].xaxis.set_major_locator(FixedLocator([-10, -5, 0, 5, 10]))
    axes[-1].xaxis.set_major_formatter(
        FixedFormatter(["lead 10", "lead 5", "peak", "lag 5", "lag 10"])
    )
    axes[-1].set_xlabel(
        "Time relative to MHW peak (days)",
        fontsize=9.8,
        labelpad=8,
    )
    if SHOW_FIGURE_HEADER:
        fig.suptitle(
            "Atmospheric anomalies around MHW peak",
            fontsize=13.5,
            fontweight="semibold",
            y=0.985,
        )
        fig.text(
            0.5,
            0.953,
            "Centered 5-day mean | temperature shading, geopotential-height contours, vertical-motion cores",
            ha="center",
            va="top",
            fontsize=8.9,
            color="#4f565e",
        )

    legend_handles = [
        Patch(
            facecolor=to_rgba(ASCENT_HATCH_COLOR, 0.13),
            edgecolor=ASCENT_HATCH_COLOR,
            hatch=ASCENT_HATCH,
            linewidth=0.8,
            label="Strong ascent core (ω′ < 0)",
        ),
        Patch(
            facecolor=to_rgba(DESCENT_HATCH_COLOR, 0.13),
            edgecolor=DESCENT_HATCH_COLOR,
            hatch=DESCENT_HATCH,
            linewidth=0.8,
            label="Strong descent core (ω′ > 0)",
        ),
        Line2D(
            [0],
            [0],
            color="#252a30",
            lw=1.2,
            ls="-",
            label="Z′ > 0 (gpm)",
        ),
        Line2D(
            [0],
            [0],
            color="#252a30",
            lw=1.2,
            ls="--",
            label="Z′ < 0 (gpm)",
        ),
    ]
    if SHOW_ENCODING_LEGEND:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.105),
            ncol=4,
            frameon=False,
            fontsize=7.9,
            handlelength=2.5,
            columnspacing=1.35,
        )

    if SHOW_FIGURE_HEADER:
        top = 0.91
    else:
        top = 0.965
    bottom = 0.235 if SHOW_ENCODING_LEGEND else 0.18
    fig.subplots_adjust(
        left=0.115,
        right=0.965,
        top=top,
        bottom=bottom,
        hspace=0.22,
    )
    if last_fill is None:
        raise RuntimeError("No temperature field was plotted")
    cbar_y = 0.055 if SHOW_ENCODING_LEGEND else 0.045
    cbar_ax = fig.add_axes([0.115, cbar_y, 0.85, 0.022])
    cbar = fig.colorbar(
        last_fill,
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.set_label(
        "Temperature anomaly, T′ (K)",
        fontsize=9.4,
        labelpad=5,
    )
    cbar.ax.tick_params(labelsize=8.0, length=3)
    cbar.outline.set_linewidth(0.7)

    fig.savefig(
        OUT_PNG,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PNG] {OUT_PNG}", flush=True)
    print(f"[PDF] {OUT_PDF}", flush=True)
    return t_limit, z_levels_top, z_levels_bottom, smoothing_stats


def validate_final(ds: xr.Dataset) -> None:
    if not np.array_equal(ds.relative_day.values.astype(np.int16), RELATIVE_DAYS):
        raise ValueError("Final relative_day coordinate is incorrect")
    if not np.array_equal(ds.level.values.astype(np.int16), LEVELS):
        raise ValueError("Final pressure levels are incorrect")
    if ds.sample_count.values.tolist() != [13, 1]:
        raise ValueError(f"Final sample counts are incorrect: {ds.sample_count.values}")
    names = [
        "temperature_anomaly_5dmean_event",
        "geopotential_height_anomaly_5dmean_event",
        "vertical_velocity_anomaly_5dmean_event",
        "temperature_anomaly_5dmean_group",
        "geopotential_height_anomaly_5dmean_group",
        "vertical_velocity_anomaly_5dmean_group",
    ]
    for name in names:
        if not np.isfinite(ds[name].values).all():
            raise ValueError(f"Final variable contains NaN/Inf: {name}")


def write_check(
    ds: xr.Dataset,
    w_anom: xr.Dataset,
    t_limit: float,
    z_levels_top: np.ndarray,
    z_levels_bottom: np.ndarray,
    smoothing_stats: dict[str, float],
) -> None:
    lines = [
        "ERA5 T-Z-omega time-pressure section check",
        "=" * 78,
        f"Output NetCDF: {OUT_NC}",
        f"Output PNG: {OUT_PNG}",
        f"Output PDF: {OUT_PDF}",
        f"T/Z anomaly cache: {base.DAILY_ANOM_CACHE}",
        f"Omega source GRIB: {W_GRIB_SOURCE}",
        f"Omega GRIB area-mean cache: {W_GRIB_AREA_CACHE}",
        f"Omega daily raw cache: {W_DAILY_RAW_CACHE}",
        f"Omega anomaly cache: {W_DAILY_ANOM_CACHE}",
        "",
        "Configuration",
        f"  Region: {LAT_MIN:g}-{LAT_MAX:g}N, {LON_MIN:g}-{LON_MAX:g}E",
        f"  Levels (hPa): {LEVELS.tolist()}",
        f"  Relative days: {RELATIVE_DAYS.tolist()}",
        f"  Running mean offsets: {ROLL_OFFSETS.tolist()}",
        "  Detrending: linear by variable and level, 1981-2024",
        "  Climatology: 1981-2020 no-leap daily climatology",
        "  Omega sign: negative ascent; positive descent",
        "  Significance: not calculated or plotted",
        "  Display smoothing: shape-preserving PCHIP in time and log-pressure only",
        "",
        "Dimensions and counts",
        f"  Omega daily anomaly days: {w_anom.sizes['time']}",
        f"  Omega climatology sample counts: {np.unique(w_anom.climatology_sample_count.values).tolist()}",
        f"  Events: {ds.sizes['event']}",
        f"  Groups: {ds.group.values.tolist()}",
        f"  Sample count: {ds.sample_count.values.tolist()}",
        "",
        "Plot levels",
        f"  Temperature shading symmetric limit: +/- {t_limit:.3f} K",
        "  Omega ascent core: values at or below the panel-specific 15th percentile of negative anomalies",
        "  Omega descent core: values at or above the panel-specific 85th percentile of positive anomalies",
        f"  N=13 geopotential-height contours: {z_levels_top.tolist()} gpm",
        f"  2004 geopotential-height contours: {z_levels_bottom.tolist()} gpm",
        "",
        "Variable statistics",
    ]
    for name in [
        "temperature_anomaly_5dmean_event",
        "geopotential_height_anomaly_5dmean_event",
        "vertical_velocity_anomaly_5dmean_event",
        "temperature_anomaly_5dmean_group",
        "geopotential_height_anomaly_5dmean_group",
        "vertical_velocity_anomaly_5dmean_group",
    ]:
        values = ds[name].values
        lines.append(
            f"  {name}: shape={values.shape}, min={np.nanmin(values):.7f}, "
            f"max={np.nanmax(values):.7f}, mean={np.nanmean(values):.7f}, "
            f"nan={int(np.isnan(values).sum())}, "
            f"inf={int(np.isinf(values).sum())}"
        )
    lines.extend(["", "Display interpolation bounds"])
    for key, value in smoothing_stats.items():
        lines.append(f"  {key}: {value:.8f}")
    OUT_CHECK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[CHECK] {OUT_CHECK}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-w-year-cache",
        action="store_true",
        help="Rebuild all yearly omega area-mean caches",
    )
    parser.add_argument(
        "--force-w-combined-cache",
        action="store_true",
        help="Rebuild combined omega raw and anomaly caches",
    )
    parser.add_argument(
        "--force-grib-area-cache",
        action="store_true",
        help="Rebuild the CDO 2023-2024 GRIB area-mean cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 100, flush=True)
    print("ERA5 T/Z/OMEGA TIME-PRESSURE SECTION: N=13 PLUS 2004", flush=True)
    print("=" * 100, flush=True)

    if not base.DAILY_ANOM_CACHE.exists():
        raise FileNotFoundError(
            "Existing T/Z anomaly cache is required: "
            f"{base.DAILY_ANOM_CACHE}"
        )
    tz_anom = xr.open_dataset(base.DAILY_ANOM_CACHE)
    w_raw = build_w_raw_cache(
        force=args.force_w_combined_cache,
        force_years=args.force_w_year_cache,
        force_grib_area_cache=args.force_grib_area_cache,
    )
    try:
        w_anom = build_w_anomaly_cache(
            w_raw,
            force=args.force_w_combined_cache,
        )
    finally:
        w_raw.close()

    try:
        base_event = base.build_event_dataset(tz_anom)
        event_ds = add_vertical_velocity(base_event, w_anom)
        base_event.close()
        validate_final(event_ds)
        save_event_dataset(event_ds)
        (
            t_limit,
            z_levels_top,
            z_levels_bottom,
            smoothing_stats,
        ) = plot_sections(event_ds)
        write_check(
            event_ds,
            w_anom,
            t_limit,
            z_levels_top,
            z_levels_bottom,
            smoothing_stats,
        )
        event_ds.close()
    finally:
        tz_anom.close()
        w_anom.close()

    print("[DONE] All T/Z/omega outputs passed numerical checks.", flush=True)


if __name__ == "__main__":
    main()
