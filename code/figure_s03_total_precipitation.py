#!/usr/bin/env python3
"""Compute and plot ERA5 total-precipitation anomalies for two MHW groups."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


ERA5_ROOT = Path("/path/to/data/ERA5")
PRECIP_2023 = Path(
    "/path/to/data/非绝热加热异常_ERA5反推/"
    "era5_convective_precip_peak_lag0_7/ERA5_precipitation_rates_2023_ND_daily_1deg.nc"
)
OUT_ROOT = Path(
    "/path/to/data/非绝热加热异常_ERA5反推/era5_total_precip_peak_lag0_7"
)

BASELINE_YEARS = list(range(1981, 2021))
LAGS = list(range(8))
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
        "key": "mean_13_events",
        "title": "13-event mean",
        "event_indices": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13],
    },
    {
        "panel": "B",
        "key": "event_2004_12_24",
        "title": "2004-12-24",
        "event_indices": [6],
    },
]

EXTENT = [150.0, 250.0, 20.0, 80.0]
FIGURE_REGION_TAG = "20_80N"
BOX = {"lon_min": 200.0, "lon_max": 225.0, "lat_min": 40.0, "lat_max": 50.0}
RATE_TO_MM_DAY = 86400.0
NEGATIVE_NOISE_LIMIT = -1.0e-8
P_THRESHOLD = 0.05
STIPPLE_STEP = 2
STIPPLE_SIZE = 7.0


def calc_doy_nl(ts: pd.Timestamp) -> int:
    if ts.month == 2 and ts.day == 29:
        raise ValueError("Feb 29 is excluded from the no-leap climatology.")
    doy = int(ts.dayofyear)
    if ts.is_leap_year and doy > 60:
        doy -= 1
    return doy


def date_from_doy_nl(year: int, doy_nl: int) -> pd.Timestamp:
    base = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(doy_nl) - 1)
    return pd.Timestamp(year=year, month=base.month, day=base.day)


def build_target_table() -> pd.DataFrame:
    rows = []
    for event, peak in enumerate(PEAK_DATES):
        peak_date = pd.Timestamp(peak)
        for lag in LAGS:
            target_date = peak_date + pd.Timedelta(days=lag)
            rows.append(
                {
                    "event": event,
                    "peak_date": peak_date,
                    "lag": lag,
                    "target_date": target_date,
                    "year": target_date.year,
                    "doy_nl": calc_doy_nl(target_date),
                }
            )
    return pd.DataFrame(rows)


def normalize_daily_rate(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for old, new in {
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
    }.items():
        if old in da.coords or old in da.dims:
            rename[old] = new
    da = da.rename(rename)
    if float(da["lat"].values[0]) > float(da["lat"].values[-1]):
        da = da.sortby("lat")
    if float(da["lon"].max()) <= 180.0:
        da = da.assign_coords(lon=da["lon"] % 360).sortby("lon")
    return da.transpose("time", "lat", "lon")


def validate_project_grid(da: xr.DataArray, source: str) -> None:
    if da.sizes.get("lat") != 181 or da.sizes.get("lon") != 360:
        raise ValueError(f"{source}: unexpected grid {dict(da.sizes)}")
    if not np.allclose(da["lat"].values, np.arange(-90.0, 91.0, 1.0)):
        raise ValueError(f"{source}: latitude is not -90..90 at 1 degree.")
    if not np.allclose(da["lon"].values, np.arange(0.0, 360.0, 1.0)):
        raise ValueError(f"{source}: longitude is not 0..359 at 1 degree.")


def clip_packing_noise(da: xr.DataArray, source: str) -> xr.DataArray:
    minimum = float(da.min(skipna=True))
    if minimum < NEGATIVE_NOISE_LIMIT:
        raise ValueError(
            f"{source}: physically invalid total precipitation minimum {minimum}"
        )
    if minimum < 0.0:
        da = da.clip(min=0.0)
    return da


def read_dates_from_year(
    year: int, dates: list[pd.Timestamp], variable: str = "mtpr"
) -> xr.DataArray:
    path = ERA5_ROOT / f"{year}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    requested = pd.DatetimeIndex(sorted({pd.Timestamp(d).normalize() for d in dates}))
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if variable not in ds:
            raise KeyError(f"{variable} not found in {path}")
        da = normalize_daily_rate(ds[variable])
        validate_project_grid(da, str(path))
        available = pd.DatetimeIndex(pd.to_datetime(da["time"].values).normalize())
        missing = requested.difference(available)
        if len(missing):
            raise KeyError(f"{path} missing dates: {missing.strftime('%Y-%m-%d').tolist()}")
        out = da.sel(time=requested).load().astype("float32")
        out = clip_packing_noise(out, str(path))
    return out


def read_2023_dates(
    path: Path, dates: list[pd.Timestamp], variable: str = "mtpr"
) -> xr.DataArray:
    requested = pd.DatetimeIndex(sorted({pd.Timestamp(d).normalize() for d in dates}))
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if variable not in ds:
            raise KeyError(f"{variable} not found in {path}")
        da = normalize_daily_rate(ds[variable])
        validate_project_grid(da, str(path))
        available = pd.DatetimeIndex(pd.to_datetime(da["time"].values).normalize())
        missing = requested.difference(available)
        if len(missing):
            raise KeyError(f"{path} missing dates: {missing.strftime('%Y-%m-%d').tolist()}")
        out = da.sel(time=requested).load().astype("float32")
        out = clip_packing_noise(out, str(path))
    return out


def build_climatology(
    target_df: pd.DataFrame,
) -> tuple[xr.DataArray, xr.DataArray, list[dict]]:
    unique_doys = sorted(target_df["doy_nl"].unique().astype(int).tolist())
    date_by_year = {
        year: [date_from_doy_nl(year, doy) for doy in unique_doys]
        for year in BASELINE_YEARS
    }
    clim_sum = None
    count = np.zeros(len(unique_doys), dtype=np.int16)
    check_rows = []
    lat = lon = None

    for year in BASELINE_YEARS:
        print(f"[CLIM] {year}: {len(unique_doys)} target DOYs", flush=True)
        da = read_dates_from_year(year, date_by_year[year])
        values = da.values.astype("float64")
        if clim_sum is None:
            clim_sum = np.zeros_like(values, dtype="float64")
            lat = da["lat"].values.astype("float32")
            lon = da["lon"].values.astype("float32")
        clim_sum += values
        count += 1
        check_rows.append(
            {
                "stage": "climatology_input",
                "year": year,
                "n_dates": len(unique_doys),
                "minimum_kg_m2_s": float(np.nanmin(values)),
                "maximum_kg_m2_s": float(np.nanmax(values)),
                "mean_kg_m2_s": float(np.nanmean(values)),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
            }
        )
        da.close()

    if clim_sum is None or lat is None or lon is None:
        raise RuntimeError("No climatology data were read.")
    if not np.all(count == len(BASELINE_YEARS)):
        raise ValueError(f"Unexpected climatology counts: {count}")

    climatology = xr.DataArray(
        (clim_sum / count[:, None, None]).astype("float32"),
        dims=("doy_nl", "lat", "lon"),
        coords={
            "doy_nl": np.array(unique_doys, dtype=np.int16),
            "lat": lat,
            "lon": lon,
        },
        name="mtpr_climatology_doy",
        attrs={
            "long_name": "ERA5 mean total precipitation rate no-leap daily climatology",
            "units": "kg m-2 s-1",
            "baseline_period": "1981-2020",
        },
    )
    sample_count = xr.DataArray(
        count,
        dims=("doy_nl",),
        coords={"doy_nl": climatology["doy_nl"]},
        name="climatology_sample_count",
        attrs={"long_name": "Number of years in each daily climatology", "units": "1"},
    )
    return climatology, sample_count, check_rows


def build_event_fields(
    target_df: pd.DataFrame, climatology: xr.DataArray, precip_2023: Path
) -> tuple[xr.DataArray, xr.DataArray, list[dict]]:
    events = len(PEAK_DATES)
    event_rate = np.full(
        (events, len(LAGS), climatology.sizes["lat"], climatology.sizes["lon"]),
        np.nan,
        dtype="float32",
    )
    check_rows = []

    for year, rows in target_df.groupby("year"):
        dates = pd.DatetimeIndex(rows["target_date"].sort_values().unique())
        print(f"[EVENT] {year}: {len(dates)} dates", flush=True)
        if int(year) == 2023:
            da = read_2023_dates(precip_2023, list(dates))
            source = str(precip_2023)
        else:
            da = read_dates_from_year(int(year), list(dates))
            source = str(ERA5_ROOT / f"{int(year)}.nc")
        index = {
            pd.Timestamp(t).normalize(): i
            for i, t in enumerate(pd.to_datetime(da["time"].values))
        }
        for row in rows.itertuples(index=False):
            event_rate[int(row.event), int(row.lag)] = da.isel(
                time=index[pd.Timestamp(row.target_date).normalize()]
            ).values
        values = da.values
        check_rows.append(
            {
                "stage": "event_input",
                "year": int(year),
                "n_dates": len(dates),
                "source_file": source,
                "minimum_kg_m2_s": float(np.nanmin(values)),
                "maximum_kg_m2_s": float(np.nanmax(values)),
                "mean_kg_m2_s": float(np.nanmean(values)),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
            }
        )
        da.close()

    event_da = xr.DataArray(
        event_rate,
        dims=("event", "lag", "lat", "lon"),
        coords={
            "event": np.arange(events, dtype=np.int16),
            "lag": np.array(LAGS, dtype=np.int16),
            "lat": climatology["lat"],
            "lon": climatology["lon"],
        },
        name="mtpr_event_lag",
        attrs={
            "long_name": "ERA5 mean total precipitation rate for each event and lag",
            "units": "kg m-2 s-1",
            "lag_definition": "lag=0 is peak day; lag=7 is peak day plus 7 days",
        },
    )

    anomaly = np.empty_like(event_rate)
    for row in target_df.itertuples(index=False):
        anomaly[int(row.event), int(row.lag)] = (
            event_rate[int(row.event), int(row.lag)]
            - climatology.sel(doy_nl=int(row.doy_nl)).values
        )
    anomaly_da = xr.DataArray(
        anomaly,
        dims=event_da.dims,
        coords=event_da.coords,
        name="mtpr_anomaly_event_lag",
        attrs={
            "long_name": "ERA5 total precipitation rate anomaly for each event and lag",
            "units": "kg m-2 s-1",
            "climatology": "1981-2020 no-leap daily climatology",
        },
    )
    return event_da, anomaly_da, check_rows


def box_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    subset = da.sel(
        lat=slice(BOX["lat_min"], BOX["lat_max"]),
        lon=slice(BOX["lon_min"], BOX["lon_max"]),
    )
    weights = np.cos(np.deg2rad(subset["lat"]))
    return subset.weighted(weights).mean(dim=("lat", "lon"), skipna=True)


def assemble_output(
    target_df: pd.DataFrame,
    climatology: xr.DataArray,
    sample_count: xr.DataArray,
    event_rate: xr.DataArray,
    anomaly: xr.DataArray,
    precip_2023: Path,
) -> tuple[xr.Dataset, pd.DataFrame]:
    peak_date = np.array(pd.to_datetime(PEAK_DATES), dtype="datetime64[ns]")
    target_date = (
        target_df.pivot(index="event", columns="lag", values="target_date")
        .sort_index()
        .sort_index(axis=1)
        .values.astype("datetime64[ns]")
    )
    doy_nl = (
        target_df.pivot(index="event", columns="lag", values="doy_nl")
        .sort_index()
        .sort_index(axis=1)
        .values.astype("int16")
    )

    event_lag_mean = anomaly.mean(dim="lag", skipna=True)
    n13_indices = PANEL_SPECS[0]["event_indices"]
    n13_samples = event_lag_mean.isel(event=n13_indices).values.astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        _, p_values = stats.ttest_1samp(
            n13_samples,
            popmean=0.0,
            axis=0,
            nan_policy="omit",
        )
    p_values = np.where(np.isfinite(p_values), p_values, 1.0).astype("float32")
    p_value_n13 = xr.DataArray(
        p_values,
        dims=("lat", "lon"),
        coords={"lat": climatology["lat"], "lon": climatology["lon"]},
        name="mtpr_anomaly_p_value_N13_lag0_7",
        attrs={
            "long_name": "P value for the N13 lag0-7 mean total precipitation anomaly",
            "test": "two-sided one-sample Student t-test across 13 event-level lag0-7 means",
            "null_hypothesis": "population mean anomaly equals zero",
            "units": "1",
        },
    )
    significant_n13 = (p_value_n13 < P_THRESHOLD).rename(
        "mtpr_anomaly_significant_p005_N13_lag0_7"
    )
    significant_n13.attrs.update(
        {
            "long_name": "N13 total precipitation anomaly significant at p < 0.05",
            "threshold": P_THRESHOLD,
        }
    )
    panel_keys = [spec["key"] for spec in PANEL_SPECS]
    composites = xr.concat(
        [
            event_lag_mean.isel(event=spec["event_indices"]).mean(dim="event", skipna=True)
            for spec in PANEL_SPECS
        ],
        dim=pd.Index(panel_keys, name="panel"),
    ).rename("mtpr_anomaly_composite_lag0_7")

    ds = xr.Dataset(
        {
            "mtpr_event_lag": event_rate,
            "mtpr_climatology_doy": climatology,
            "climatology_sample_count": sample_count,
            "mtpr_anomaly_event_lag": anomaly,
            "mtpr_anomaly_event_lag0_7_mean": event_lag_mean,
            "mtpr_anomaly_composite_lag0_7": composites,
            "mtpr_anomaly_p_value_N13_lag0_7": p_value_n13,
            "mtpr_anomaly_significant_p005_N13_lag0_7": significant_n13,
            "peak_date": (("event",), peak_date),
            "target_date": (("event", "lag"), target_date),
            "target_doy_nl": (("event", "lag"), doy_nl),
            "panel_event_count": (
                ("panel",),
                np.array([len(spec["event_indices"]) for spec in PANEL_SPECS], dtype=np.int16),
            ),
        }
    )
    ds["mtpr_anomaly_composite_lag0_7"].attrs.update(
        {
            "long_name": "ERA5 total precipitation rate anomaly composite, lag0-7 mean",
            "units": "kg m-2 s-1",
        }
    )
    ds.attrs.update(
        {
            "title": "ERA5 total precipitation anomalies for two MHW event groups",
            "baseline_period": "1981-2020",
            "climatology": "no-leap daily climatology",
            "detrending": "none",
            "lag_average": "lag0-7 mean",
            "grouping": "13-event mean excludes only 2004-12-24; 2004-12-24 is a separate case",
            "plot_extent": "150-250E, 20-80N",
            "box": "200-225E, 40-50N",
            "historical_source": str(ERA5_ROOT),
            "2023_source": str(precip_2023),
            "rate_to_mm_day": "multiply kg m-2 s-1 by 86400",
            "negative_rate_handling": (
                "Values between -1e-8 and 0 kg m-2 s-1 are treated as packing noise "
                "and clipped to zero."
            ),
        }
    )

    rows = []
    event_box = box_weighted_mean(event_rate) * RATE_TO_MM_DAY
    clim_for_events = np.empty((len(PEAK_DATES), len(LAGS)), dtype="float64")
    for row in target_df.itertuples(index=False):
        clim_for_events[int(row.event), int(row.lag)] = float(
            box_weighted_mean(climatology.sel(doy_nl=int(row.doy_nl))) * RATE_TO_MM_DAY
        )
    anomaly_box = box_weighted_mean(anomaly) * RATE_TO_MM_DAY

    for event, peak in enumerate(PEAK_DATES):
        for lag in LAGS:
            rows.append(
                {
                    "record_type": "event_lag",
                    "panel": "",
                    "event": event,
                    "peak_date": peak,
                    "lag": lag,
                    "target_date": str(pd.Timestamp(target_date[event, lag]).date()),
                    "doy_nl": int(doy_nl[event, lag]),
                    "n_events": 1,
                    "mtpr_event_box_mm_day": float(event_box.sel(event=event, lag=lag)),
                    "mtpr_climatology_box_mm_day": float(clim_for_events[event, lag]),
                    "mtpr_anomaly_box_mm_day": float(anomaly_box.sel(event=event, lag=lag)),
                }
            )

    for spec in PANEL_SPECS:
        values = anomaly_box.isel(event=spec["event_indices"]).mean(dim=("event", "lag"))
        rows.append(
            {
                "record_type": "panel_lag0_7_mean",
                "panel": spec["key"],
                "event": "",
                "peak_date": "",
                "lag": "0-7",
                "target_date": "",
                "doy_nl": "",
                "n_events": len(spec["event_indices"]),
                "mtpr_event_box_mm_day": "",
                "mtpr_climatology_box_mm_day": "",
                "mtpr_anomaly_box_mm_day": float(values),
            }
        )
    return ds, pd.DataFrame(rows)


def build_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "reference_brown_teal",
        [
            "#744000",
            "#9a6517",
            "#bf9140",
            "#dec176",
            "#f0dea8",
            "#f7f2df",
            "#dff1ee",
            "#b7dfd8",
            "#82c9bf",
            "#47aa9f",
            "#197f76",
            "#00594f",
        ],
        N=256,
    )


def lon_label(x: float) -> str:
    if x == 180:
        return "180°"
    if x < 180:
        return f"{int(x)}°E"
    return f"{int(360 - x)}°W"


def add_box(ax, transform) -> None:
    x0, x1 = BOX["lon_min"], BOX["lon_max"]
    y0, y1 = BOX["lat_min"], BOX["lat_max"]
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        color="#17823b",
        linewidth=1.8,
        transform=transform,
        zorder=8,
    )


def nice_vmax(values: np.ndarray) -> float:
    max_abs = float(np.nanmax(np.abs(values)))
    candidates = np.array(
        [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    )
    target = max_abs * 1.04
    bigger = candidates[candidates >= target]
    if len(bigger):
        return float(bigger[0])
    return float(np.ceil(target / 2.0) * 2.0)


def colorbar_ticks(vmax: float) -> np.ndarray:
    step = 2.0 if vmax <= 6.0 else 4.0
    start = np.ceil(-vmax / step) * step
    return np.arange(start, vmax + 0.01, step)


def plot_panel(ds: xr.Dataset, out_root: Path, spec: dict) -> tuple[Path, Path]:
    data = ds["mtpr_anomaly_composite_lag0_7"] * RATE_TO_MM_DAY
    field = data.sel(panel=spec["key"]).sel(
        lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3])
    )
    vmax = nice_vmax(field.values)
    levels = np.linspace(-vmax, vmax, 13)

    projection = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6.4, 4.75))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    im = ax.contourf(
        field["lon"],
        field["lat"],
        field,
        levels=levels,
        cmap=build_cmap(),
        extend="both",
        transform=data_crs,
        zorder=1,
    )
    ax.coastlines(resolution="110m", linewidth=0.85, color="#222222", zorder=7)
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linewidth=0.28,
        edgecolor="#333333",
        zorder=7,
    )
    ax.set_extent(EXTENT, crs=data_crs)
    add_box(ax, data_crs)

    if spec["key"] == "mean_13_events":
        p_value = ds["mtpr_anomaly_p_value_N13_lag0_7"].sel(
            lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3])
        )
        significant = (p_value < P_THRESHOLD).isel(
            lat=slice(None, None, STIPPLE_STEP),
            lon=slice(None, None, STIPPLE_STEP),
        )
        lon2d, lat2d = np.meshgrid(significant["lon"].values, significant["lat"].values)
        mask = significant.values.astype(bool)
        ax.scatter(
            lon2d[mask],
            lat2d[mask],
            s=STIPPLE_SIZE,
            c="black",
            marker="o",
            linewidths=0,
            alpha=0.92,
            transform=data_crs,
            zorder=9,
        )

    xticks = [150, 170, 190, 210, 230, 250]
    yticks = [20, 30, 40, 50, 60, 70, 80]
    ax.set_xticks(xticks, crs=data_crs)
    ax.set_yticks(yticks, crs=data_crs)
    ax.set_xticklabels([lon_label(x) for x in xticks], fontsize=9)
    ax.set_yticklabels([f"{y}°N" for y in yticks], fontsize=9)
    ax.tick_params(length=3, width=0.8, pad=2)
    ax.text(
        0.01,
        1.015,
        f"({spec['panel'].lower()}) {spec['title']}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
    )
    ax.text(
        0.99,
        1.015,
        "lag0-7 mean",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
    )

    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.10)
    fig.canvas.draw()
    map_bottom = ax.get_position().y0
    cax = fig.add_axes([0.24, map_bottom - 0.105, 0.52, 0.027])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Total precipitation anomaly (mm day$^{-1}$)", fontsize=10.5)
    cbar.set_ticks(colorbar_ticks(vmax))
    cbar.ax.tick_params(labelsize=9)

    suffix = "N13_p005" if spec["key"] == "mean_13_events" else "2004_12_24"
    png = out_root / f"ERA5_mtpr_anomaly_lag0_7_{suffix}_{FIGURE_REGION_TAG}.png"
    pdf = out_root / f"ERA5_mtpr_anomaly_lag0_7_{suffix}_{FIGURE_REGION_TAG}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_combined(ds: xr.Dataset, out_root: Path) -> Path:
    """Plot the N=13 composite and 2004 event side by side."""
    data = ds["mtpr_anomaly_composite_lag0_7"] * RATE_TO_MM_DAY
    projection = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()
    cmap = build_cmap()

    fig = plt.figure(figsize=(12.6, 5.25), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        left=0.055,
        right=0.955,
        bottom=0.205,
        top=0.91,
        wspace=0.10,
    )

    panel_headers = {
        "mean_13_events": ("a", "N = 13"),
        "event_2004_12_24": ("b", "2004-12-24"),
    }
    xticks = [150, 170, 190, 210, 230, 250]
    yticks = [20, 30, 40, 50, 60, 70, 80]
    contour_sets = []
    axes = []

    for column, spec in enumerate(PANEL_SPECS):
        ax = fig.add_subplot(grid[0, column], projection=projection)
        axes.append(ax)
        field = data.sel(panel=spec["key"]).sel(
            lon=slice(EXTENT[0], EXTENT[1]),
            lat=slice(EXTENT[2], EXTENT[3]),
        )
        vmax = nice_vmax(field.values)
        levels = np.linspace(-vmax, vmax, 13)
        contour = ax.contourf(
            field["lon"],
            field["lat"],
            field,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=data_crs,
            zorder=1,
        )
        contour_sets.append((contour, vmax))

        ax.coastlines(
            resolution="110m",
            linewidth=0.75,
            color="#222222",
            zorder=7,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            linewidth=0.25,
            edgecolor="#333333",
            zorder=7,
        )
        ax.set_extent(EXTENT, crs=data_crs)
        add_box(ax, data_crs)

        if spec["key"] == "mean_13_events":
            p_value = ds["mtpr_anomaly_p_value_N13_lag0_7"].sel(
                lon=slice(EXTENT[0], EXTENT[1]),
                lat=slice(EXTENT[2], EXTENT[3]),
            )
            significant = (p_value < P_THRESHOLD).isel(
                lat=slice(None, None, STIPPLE_STEP),
                lon=slice(None, None, STIPPLE_STEP),
            )
            lon2d, lat2d = np.meshgrid(
                significant["lon"].values,
                significant["lat"].values,
            )
            mask = significant.values.astype(bool)
            ax.scatter(
                lon2d[mask],
                lat2d[mask],
                s=5.2,
                c="black",
                marker="o",
                linewidths=0,
                alpha=0.88,
                transform=data_crs,
                zorder=9,
            )

        ax.set_xticks(xticks, crs=data_crs)
        ax.set_yticks(yticks, crs=data_crs)
        ax.set_xticklabels([lon_label(x) for x in xticks], fontsize=8.5)
        ax.set_yticklabels([f"{y}°N" for y in yticks], fontsize=8.5)
        if column == 0:
            ax.tick_params(
                labelleft=True,
                labelright=False,
                left=True,
                right=False,
            )
        else:
            ax.yaxis.tick_right()
            ax.tick_params(
                labelleft=False,
                labelright=True,
                left=False,
                right=True,
            )
        ax.tick_params(length=2.8, width=0.7, pad=1.8)

        panel, group_label = panel_headers[spec["key"]]
        ax.text(
            0.0,
            1.018,
            f"({panel})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.2,
            fontweight="semibold",
            color="#20242a",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.018,
            group_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="semibold",
            color="#20242a",
            clip_on=False,
        )
        ax.text(
            1.0,
            1.018,
            "lag 0–7 mean",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.7,
            fontweight="semibold",
            color="#20242a",
            clip_on=False,
        )

    fig.canvas.draw()
    for ax, (contour, vmax) in zip(axes, contour_sets):
        position = ax.get_position()
        cax = fig.add_axes(
            [
                position.x0 + 0.12 * position.width,
                position.y0 - 0.105,
                0.76 * position.width,
                0.025,
            ]
        )
        colorbar = fig.colorbar(
            contour,
            cax=cax,
            orientation="horizontal",
        )
        colorbar.set_label(
            "Total precipitation anomaly (mm day$^{-1}$)",
            fontsize=9.0,
            labelpad=3,
        )
        colorbar.set_ticks(colorbar_ticks(vmax))
        colorbar.ax.tick_params(
            labelsize=8.0,
            length=2.6,
            width=0.65,
        )
        colorbar.outline.set_linewidth(0.6)

    out_root.mkdir(parents=True, exist_ok=True)
    output = out_root / (
        "ERA5_mtpr_anomaly_lag0_7_N13_vs_2004_"
        f"combined_{FIGURE_REGION_TAG}.png"
    )
    fig.savefig(output, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_composites(ds: xr.Dataset, out_root: Path) -> list[Path]:
    outputs = []
    for spec in PANEL_SPECS:
        outputs.extend(plot_panel(ds, out_root, spec))
    outputs.append(plot_combined(ds, out_root))
    return outputs


def build_quality_rows(
    ds: xr.Dataset, stage_rows: list[dict], standardized_2023: Path
) -> pd.DataFrame:
    rows = list(stage_rows)
    for name in [
        "mtpr_event_lag",
        "mtpr_climatology_doy",
        "mtpr_anomaly_event_lag",
        "mtpr_anomaly_composite_lag0_7",
        "mtpr_anomaly_p_value_N13_lag0_7",
    ]:
        values = ds[name].values
        rows.append(
            {
                "stage": "output",
                "variable": name,
                "minimum_kg_m2_s": float(np.nanmin(values)),
                "maximum_kg_m2_s": float(np.nanmax(values)),
                "mean_kg_m2_s": float(np.nanmean(values)),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
            }
        )
    p_value = ds["mtpr_anomaly_p_value_N13_lag0_7"]
    p_regions = {
        "full_grid": p_value,
        "plot_domain": p_value.sel(
            lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3])
        ),
        "green_box": p_value.sel(
            lon=slice(BOX["lon_min"], BOX["lon_max"]),
            lat=slice(BOX["lat_min"], BOX["lat_max"]),
        ),
    }
    for region_name, p_region in p_regions.items():
        rows.append(
            {
                "stage": "N13_significance",
                "variable": "mtpr_anomaly_p_value_N13_lag0_7",
                "region": region_name,
                "p_threshold": P_THRESHOLD,
                "significant_fraction": float((p_region < P_THRESHOLD).mean()),
            }
        )
    with xr.open_dataset(standardized_2023, engine="netcdf4") as precip:
        variables = set(precip.data_vars)
        rows.append(
            {
                "stage": "2023_source_variable_check",
                "variable": "mtpr",
                "status": "available" if "mtpr" in variables else "missing",
                "available_variables": ",".join(sorted(variables)),
            }
        )
    return pd.DataFrame(rows)


def validate_output(ds: xr.Dataset) -> None:
    expected = {
        "event": 14,
        "lag": 8,
        "lat": 181,
        "lon": 360,
        "panel": 2,
    }
    for dim, size in expected.items():
        if int(ds.sizes.get(dim, -1)) != size:
            raise ValueError(f"Unexpected {dim}: {ds.sizes.get(dim)} != {size}")
    if not np.all(ds["climatology_sample_count"].values == 40):
        raise ValueError("Not every daily climatology has 40 samples.")
    if ds["panel_event_count"].values.tolist() != [13, 1]:
        raise ValueError(f"Unexpected panel event counts: {ds['panel_event_count'].values}")
    for name in [
        "mtpr_event_lag",
        "mtpr_climatology_doy",
        "mtpr_anomaly_event_lag",
        "mtpr_anomaly_composite_lag0_7",
        "mtpr_anomaly_p_value_N13_lag0_7",
    ]:
        if not np.isfinite(ds[name].values).all():
            raise ValueError(f"{name} contains NaN or Inf.")
    p_value = ds["mtpr_anomaly_p_value_N13_lag0_7"]
    if float(p_value.min()) < 0.0 or float(p_value.max()) > 1.0:
        raise ValueError("N13 p values fall outside [0, 1].")
    expected_sig = p_value < P_THRESHOLD
    if not np.array_equal(
        ds["mtpr_anomaly_significant_p005_N13_lag0_7"].values,
        expected_sig.values,
    ):
        raise ValueError("Stored N13 significance mask does not match p < 0.05.")
    if float(ds["mtpr_event_lag"].min()) < -1e-12:
        raise ValueError("Event total precipitation contains negative values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precip-2023", type=Path, default=PRECIP_2023)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate PNG/PDF from the existing composite NetCDF.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="With --plot-only, generate only the combined two-panel PNG.",
    )
    parser.add_argument(
        "--figure-output-root",
        type=Path,
        default=None,
        help="Optional figure destination separate from the data/output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    nc_file = args.out_root / "ERA5_mtpr_anomaly_lag0_7_two_groups_13plus2004.nc"
    if args.plot_only:
        figure_root = args.figure_output_root or args.out_root
        figure_root.mkdir(parents=True, exist_ok=True)
        with xr.open_dataset(nc_file, engine="netcdf4") as saved:
            validate_output(saved)
            if args.combined_only:
                figure_paths = [plot_combined(saved, figure_root)]
            else:
                figure_paths = plot_composites(saved, figure_root)
        for path in figure_paths:
            print(f"[FIG] {path}")
        return

    if not args.precip_2023.exists():
        raise FileNotFoundError(
            f"Standardized 2023 precipitation file not found: {args.precip_2023}"
        )

    target_df = build_target_table()
    target_csv = args.out_root / "ERA5_mtpr_peak_lag0_7_target_dates.csv"
    target_df.to_csv(target_csv, index=False, encoding="utf-8-sig")

    climatology, sample_count, clim_checks = build_climatology(target_df)
    event_rate, anomaly, event_checks = build_event_fields(
        target_df, climatology, args.precip_2023
    )
    ds, box_df = assemble_output(
        target_df, climatology, sample_count, event_rate, anomaly, args.precip_2023
    )
    validate_output(ds)

    box_csv = args.out_root / "ERA5_mtpr_box_lag0_7_two_groups_13plus2004.csv"
    check_csv = args.out_root / "ERA5_mtpr_lag0_7_processing_check.csv"
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.floating)
    }
    encoding["mtpr_anomaly_significant_p005_N13_lag0_7"] = {
        "zlib": True,
        "complevel": 4,
        "dtype": "int8",
    }
    ds.to_netcdf(nc_file, engine="netcdf4", encoding=encoding)
    box_df.to_csv(box_csv, index=False, encoding="utf-8-sig")
    checks = build_quality_rows(ds, clim_checks + event_checks, args.precip_2023)
    checks.to_csv(check_csv, index=False, encoding="utf-8-sig")
    figure_paths = plot_composites(ds, args.out_root)

    with xr.open_dataset(nc_file, engine="netcdf4") as saved:
        validate_output(saved)
        values = saved["mtpr_anomaly_composite_lag0_7"] * RATE_TO_MM_DAY
        print(
            "[RANGE] composite anomaly "
            f"{float(values.min()):.3f}..{float(values.max()):.3f} mm/day"
        )
    panel_rows = box_df[box_df["record_type"] == "panel_lag0_7_mean"]
    print(panel_rows[["panel", "n_events", "mtpr_anomaly_box_mm_day"]].to_string(index=False))
    print(f"[NC] {nc_file}")
    print(f"[CSV] {box_csv}")
    print(f"[CHECK] {check_csv}")
    for path in figure_paths:
        print(f"[FIG] {path}")


if __name__ == "__main__":
    main()
