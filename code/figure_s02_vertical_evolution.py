from __future__ import annotations

import gc
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

import time_pressure_TZW_13_plus_2004_peak_lead10_lag10 as core


LAT_MIN = 40.0
LAT_MAX = 50.0
LON_MIN = 200.0
LON_MAX = 225.0
AREA_TAG = "MHW_40N50N_200E225E"
AREA_ATTR = "40-50N, 200-225E"

OUT_ROOT = Path(
    "/path/to/data/ABH_170E240E_1981_2024_5N90N/composite/"
    "time_pressure_TZW_13_plus_2004_lead10_lag10_MHW_40N50N_200E225E"
)
TZ_CACHE_ROOT = OUT_ROOT / "cache" / "tz"
W_CACHE_ROOT = OUT_ROOT / "cache" / "omega"
LOCAL_W_CACHE_ROOT = (
    Path(__file__).resolve().parent / ".cache" / "time_pressure_TZW"
)
USER_TZ_2024_GRIB = Path(
    "/path/to/data/ERA5_raw/2024-T_Z/"
    "1c293ebe8380e8cae1d797045469e8cb.grib"
)
CFGRIB_INDEX = LOCAL_W_CACHE_ROOT / (
    "ERA5_T_Z_2024_cfgrib.idx"
)
TZ_2024_REGIONAL_DAILY = LOCAL_W_CACHE_ROOT / (
    "ERA5_T_Z_2024_daily_cfgrib_200E225E_40N50N_1deg.nc"
)

BASE_NAME = (
    "ERA5_TZW_anomaly_time_pressure_group13_plus_2004_peak_"
    "lead10_lag10_5dmean_200E225E_40N50N_MHWregion"
)


def build_2024_year_cache_from_grib(force: bool = False) -> Path:
    base = core.base
    path = base.year_cache_path(2024)
    if path.exists() and not force:
        with xr.open_dataset(path) as opened:
            base.validate_year_dataset(opened.load(), 2024, str(path))
        print(f"[YEAR CACHE] {path}", flush=True)
        return path
    if not USER_TZ_2024_GRIB.exists():
        raise FileNotFoundError(f"Missing user-supplied 2024 T/Z GRIB: {USER_TZ_2024_GRIB}")

    if not TZ_2024_REGIONAL_DAILY.exists() or force:
        TZ_2024_REGIONAL_DAILY.parent.mkdir(parents=True, exist_ok=True)
        temporary = TZ_2024_REGIONAL_DAILY.with_suffix(".nc.part")
        if temporary.exists():
            temporary.unlink()
        print(f"[CFGRIB 2024 T/Z] {USER_TZ_2024_GRIB}", flush=True)
        with xr.open_dataset(
            USER_TZ_2024_GRIB,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                "indexpath": str(CFGRIB_INDEX),
            },
        ) as opened:
            times = pd.DatetimeIndex(opened.time.values)
            expected_times = pd.date_range(
                "2024-01-01 00:00",
                "2024-12-31 18:00",
                freq="6h",
            )
            if not np.array_equal(times.values, expected_times.values):
                raise ValueError(
                    "2024 GRIB must contain the complete 1464-step 6-hourly "
                    "time axis"
                )
            target_lat = np.arange(LAT_MIN, LAT_MAX + 0.1, 1.0)
            target_lon = np.arange(LON_MIN, LON_MAX + 0.1, 1.0)
            monthly_daily = []
            for month in range(1, 13):
                start = pd.Timestamp(2024, month, 1)
                end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=18)
                print(
                    f"[CFGRIB MONTH] {start:%Y-%m}: {start} to {end}",
                    flush=True,
                )
                regional_month = (
                    opened[["t", "z"]]
                    .sel(
                        time=slice(start, end),
                        isobaricInhPa=base.LEVELS.astype(float),
                        latitude=target_lat,
                        longitude=target_lon,
                    )
                    .rename(
                        {
                            "isobaricInhPa": "level",
                            "latitude": "lat",
                            "longitude": "lon",
                        }
                    )
                    .drop_vars(
                        ["valid_time", "number", "step"],
                        errors="ignore",
                    )
                    .load()
                )
                daily_month = (
                    regional_month.resample(time="1D")
                    .mean("time", skipna=False)
                    .load()
                )
                monthly_daily.append(daily_month)
                regional_month.close()
                gc.collect()
                print(
                    f"[CFGRIB MONTH DONE] {start:%Y-%m}: "
                    f"{daily_month.sizes['time']} days",
                    flush=True,
                )
        daily = xr.concat(
            monthly_daily,
            dim="time",
            coords="minimal",
            compat="override",
        ).sortby("time")
        if not np.isfinite(daily[["t", "z"]].to_array().values).all():
            raise ValueError("2024 regional T/Z extraction contains NaN/Inf")
        daily.attrs.update(
            {
                "source_file": str(USER_TZ_2024_GRIB),
                "source_grid": "global 0.25-degree regular latitude-longitude",
                "processing_grid": "1-degree, 200-225E and 40-50N",
                "temporal_average": "mean of 00, 06, 12, and 18 UTC",
            }
        )
        daily.to_netcdf(
            temporary,
            encoding={
                "t": {"zlib": True, "complevel": 4, "dtype": "float32"},
                "z": {"zlib": True, "complevel": 4, "dtype": "float32"},
            },
        )
        daily.close()
        for monthly in monthly_daily:
            monthly.close()
        temporary.replace(TZ_2024_REGIONAL_DAILY)

    with xr.open_dataset(TZ_2024_REGIONAL_DAILY) as opened:
        selected = base.normalize_downloaded_dataset(opened)
        weights = np.cos(np.deg2rad(selected.lat))
        regional = selected.weighted(weights).mean(("lat", "lon"), skipna=True).load()

    times = pd.DatetimeIndex(regional.time.values).normalize()
    if len(times) != 366 or times[0] != pd.Timestamp("2024-01-01") or times[-1] != pd.Timestamp("2024-12-31"):
        raise ValueError(
            "2024 GRIB should yield all 366 daily means from 2024-01-01 "
            f"to 2024-12-31; found {len(times)} days, {times[0]} to {times[-1]}"
        )
    regional = regional.assign_coords(time=times.values.astype("datetime64[ns]"))
    keep = ~((times.month == 2) & (times.day == 29))
    regional = regional.isel(time=keep)
    out = xr.Dataset(
        data_vars={
            "temperature": regional.t.astype(np.float32),
            "geopotential_height": (regional.z / base.G).astype(np.float32),
        },
        coords={"time": regional.time, "level": base.LEVELS},
        attrs={
            "source_file": str(USER_TZ_2024_GRIB),
            "regional_daily_intermediate": str(TZ_2024_REGIONAL_DAILY),
            "area": AREA_ATTR,
            "source_grid": "global 0.25-degree regular latitude-longitude",
            "processing_grid": "1-degree, 200-225E and 40-50N",
            "regridding": "exact selection of the 1-degree subset from 0.25-degree grid",
            "spatial_average": "cosine-latitude weighted mean over valid grid cells",
            "temporal_average": "mean of 00, 06, 12, and 18 UTC",
            "calendar": "no-leap",
        },
    )
    out.temperature.attrs.update(
        {"long_name": "Area-mean daily temperature", "units": "K"}
    )
    out.geopotential_height.attrs.update(
        {
            "long_name": "Area-mean daily geopotential height",
            "units": "gpm",
            "conversion": "z / 9.80665",
        }
    )
    base.validate_year_dataset(out, 2024, str(USER_TZ_2024_GRIB))
    base.write_year_cache(out, path)
    out.close()
    regional.close()
    print(f"[YEAR SAVED] {path}", flush=True)
    return path


def configure_base_module() -> None:
    base = core.base
    original_build_year_cache = base.build_year_cache
    base.LAT_MIN = LAT_MIN
    base.LAT_MAX = LAT_MAX
    base.LON_MIN = LON_MIN
    base.LON_MAX = LON_MAX
    base.OUT_ROOT = OUT_ROOT
    base.CACHE_ROOT = TZ_CACHE_ROOT
    base.YEARLY_CACHE_ROOT = TZ_CACHE_ROOT / "yearly_area_mean"
    base.DOWNLOAD_ROOT = TZ_CACHE_ROOT / "era5_missing_years_regional"
    base.DAILY_RAW_CACHE = TZ_CACHE_ROOT / (
        "ERA5_T_Z_daily_area_mean_1981_2024_200E225E_40N50N_"
        "1000_100hPa_no_leap.nc"
    )
    base.DAILY_ANOM_CACHE = TZ_CACHE_ROOT / (
        "ERA5_T_Z_daily_area_mean_anomaly_1981_2024_vs1981_2020_"
        "200E225E_40N50N_1000_100hPa_detrended_no_leap.nc"
    )

    def year_cache_path(year: int) -> Path:
        return base.YEARLY_CACHE_ROOT / (
            "ERA5_T_Z_daily_area_mean_200E225E_40N50N_"
            f"1000_100hPa_{year}_no_leap.nc"
        )

    def download_chunk_path(year: int, months: tuple[int, ...]) -> Path:
        month_tag = f"m{months[0]:02d}-{months[-1]:02d}"
        return base.DOWNLOAD_ROOT / str(year) / (
            "era5_pl_t_z_12lev_1deg_4times_40N50N_200E225E_"
            f"{year}_{month_tag}.nc"
        )

    base.year_cache_path = year_cache_path
    base.download_chunk_path = download_chunk_path

    def build_year_cache(year: int, force: bool = False) -> Path:
        if year == 2024:
            return build_2024_year_cache_from_grib(force=force)
        return original_build_year_cache(year, force=force)

    base.build_year_cache = build_year_cache


def configure_core_module() -> None:
    core.LAT_MIN = LAT_MIN
    core.LAT_MAX = LAT_MAX
    core.LON_MIN = LON_MIN
    core.LON_MAX = LON_MAX
    core.OUT_ROOT = OUT_ROOT
    core.CACHE_ROOT = W_CACHE_ROOT
    core.YEARLY_W_CACHE_ROOT = W_CACHE_ROOT / "yearly_w_area_mean"
    core.W_DAILY_RAW_CACHE = W_CACHE_ROOT / (
        "ERA5_w_daily_area_mean_1981_2024_200E225E_40N50N_"
        "1000_100hPa_no_leap.nc"
    )
    core.W_DAILY_ANOM_CACHE = W_CACHE_ROOT / (
        "ERA5_w_daily_area_mean_anomaly_1981_2024_vs1981_2020_"
        "200E225E_40N50N_1000_100hPa_detrended_no_leap.nc"
    )
    core.W_GRIB_AREA_CACHE = LOCAL_W_CACHE_ROOT / (
        "ERA5_w_area_mean_2023_2024_200E225E_40N50N.nc"
    )
    core.BASE_NAME = BASE_NAME
    core.OUT_NC = OUT_ROOT / f"{BASE_NAME}.nc"
    core.OUT_PNG = OUT_ROOT / f"{BASE_NAME}.png"
    core.OUT_PDF = OUT_ROOT / f"{BASE_NAME}.pdf"
    core.OUT_CHECK = OUT_ROOT / (
        "check_ERA5_TZW_anomaly_time_pressure_"
        "group13_plus_2004_MHW_40N50N_200E225E.txt"
    )
    core.PANEL_RIGHT_LABEL = "MHW region"
    core.SHOW_FIGURE_HEADER = False
    core.SHOW_ENCODING_LEGEND = False
    core.FIGURE_SIZE = (9.4, 7.4)

    def w_year_cache_path(year: int) -> Path:
        return core.YEARLY_W_CACHE_ROOT / (
            "ERA5_w_daily_area_mean_200E225E_40N50N_"
            f"1000_100hPa_{year}_no_leap.nc"
        )

    core.w_year_cache_path = w_year_cache_path


def correct_cache_area(path: Path) -> None:
    with nc.Dataset(path, "r+") as dataset:
        dataset.setncattr("area", AREA_ATTR)


def ensure_tz_cache() -> None:
    base = core.base
    raw = base.build_combined_raw_cache()
    try:
        anomaly = base.build_anomaly_cache(raw)
    finally:
        raw.close()
    anomaly.close()
    correct_cache_area(base.DAILY_RAW_CACHE)
    correct_cache_area(base.DAILY_ANOM_CACHE)


def main() -> None:
    configure_base_module()
    configure_core_module()
    ensure_tz_cache()
    core.main()
    correct_cache_area(core.W_DAILY_RAW_CACHE)
    correct_cache_area(core.W_DAILY_ANOM_CACHE)


if __name__ == "__main__":
    main()
