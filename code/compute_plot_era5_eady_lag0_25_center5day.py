from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


ERA5_ROOT = Path("/path/to/data/ERA5")
MONTHLY_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly_12lev")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_eady_peak_lag0_25_center5day_N13_plus_2004")
DATA_FILE = OUT_ROOT / "ERA5_Eady_peak_lagm2_27_N13_plus_2004_8lev_1981_2020clim.nc"
CHECK_FILE = OUT_ROOT / "ERA5_Eady_peak_lag0_25_center5day_check.csv"

LEVELS_12 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
PLOT_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]
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
LAGS = list(range(-2, 28))
PANEL_LAGS = [0, 5, 10, 15, 20, 25]
BASELINE_YEARS = list(range(1981, 2021))
N13_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
CASE_2004_INDEX = 6

LAT_MIN, LAT_MAX = 10.0, 85.0
LON_MIN, LON_MAX = 120.0, 260.0
DISPLAY_LAT_MIN, DISPLAY_LAT_MAX = 20.0, 80.0
DISPLAY_LON_MIN, DISPLAY_LON_MAX = 150.0, 240.0
BOX_LON_MIN, BOX_LON_MAX = 200.0, 225.0
BOX_LAT_MIN, BOX_LAT_MAX = 40.0, 50.0

R_D = 287.0
CP = 1004.0
G = 9.80665
P0_HPA = 1000.0
SECONDS_PER_DAY = 86400.0
N2_FLOOR = 1.0e-6


def calc_doy_nl(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    if ts.month == 2 and ts.day == 29:
        raise ValueError("Feb 29 is excluded from no-leap climatology")
    doy = int(ts.dayofyear)
    if ts.is_leap_year and doy > 60:
        doy -= 1
    return doy


def date_from_doy_nl(year: int, doy_nl: int) -> pd.Timestamp:
    ref = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=int(doy_nl) - 1)
    return pd.Timestamp(year=year, month=ref.month, day=ref.day)


def target_table() -> pd.DataFrame:
    rows = []
    for event, peak in enumerate(PEAK_DATES):
        peak_ts = pd.Timestamp(peak)
        for lag in LAGS:
            target = peak_ts + pd.Timedelta(days=lag)
            rows.append(
                {
                    "event": event,
                    "peak_date": peak_ts,
                    "lag": lag,
                    "target_date": target,
                    "doy_nl": calc_doy_nl(target),
                }
            )
    return pd.DataFrame(rows)


def normalize_temperature(ds: xr.Dataset) -> xr.DataArray:
    rename = {}
    for old, new in {
        "valid_time": "time",
        "pressure_level": "lev",
        "level": "lev",
        "latitude": "lat",
        "longitude": "lon",
    }.items():
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    ds = ds.rename(rename)
    if "t" not in ds:
        raise KeyError("ERA5 temperature variable 't' is missing")
    da = ds["t"].sel(lev=LEVELS_12)
    if float(da.lat.values[0]) > float(da.lat.values[-1]):
        da = da.sortby("lat")
    if float(da.lon.max()) <= 180.0:
        da = da.assign_coords(lon=da.lon % 360).sortby("lon")
    return da.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX)).transpose("time", "lev", "lat", "lon")


def monthly_file(year: int, month: int) -> Path:
    return MONTHLY_ROOT / str(year) / f"{year}{month:02d}.nc"


def load_dates(dates: list[pd.Timestamp]) -> xr.DataArray:
    unique = sorted({pd.Timestamp(value).normalize() for value in dates})
    by_year: dict[int, list[pd.Timestamp]] = defaultdict(list)
    for date in unique:
        by_year[date.year].append(date)

    parts: list[xr.DataArray] = []
    for year, year_dates in sorted(by_year.items()):
        months = sorted({date.month for date in year_dates})
        use_monthly = all(monthly_file(year, month).exists() for month in months)
        print(f"[LOAD] {year}: {len(year_dates)} days, source={'monthly12' if use_monthly else 'annual'}", flush=True)
        if use_monthly:
            month_parts = []
            for month in months:
                with xr.open_dataset(monthly_file(year, month)) as ds:
                    month_parts.append(normalize_temperature(ds).load())
            try:
                annual = xr.concat(month_parts, dim="time", coords="minimal", compat="override").sortby("time")
                index = np.array(year_dates, dtype="datetime64[ns]")
                parts.append(annual.sel(time=index).load())
            finally:
                for part in month_parts:
                    part.close()
        else:
            path = ERA5_ROOT / f"{year}.nc"
            if not path.exists():
                raise FileNotFoundError(path)
            with xr.open_dataset(path) as ds:
                da = normalize_temperature(ds)
                index = np.array(year_dates, dtype="datetime64[ns]")
                parts.append(da.sel(time=index).load())

    try:
        out = xr.concat(parts, dim="time", coords="minimal", compat="override").sortby("time").load()
    finally:
        for part in parts:
            part.close()
    expected = np.array(unique, dtype="datetime64[ns]")
    if not np.array_equal(out.time.values.astype("datetime64[ns]"), expected):
        missing = expected[~np.isin(expected, out.time.values.astype("datetime64[ns]"))]
        raise KeyError(f"Missing requested ERA5 dates: {missing}")
    return out


def pressure_gradient(values: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    """Nonuniform pressure derivative along axis 1."""
    return np.gradient(values, pressure_pa, axis=1, edge_order=2)


def compute_eady(temperature: xr.DataArray) -> tuple[xr.DataArray, pd.DataFrame]:
    """Return Eady growth rate in s-1 and stability-floor diagnostics."""
    temperature = temperature.sortby("lev", ascending=False)
    t = temperature.values.astype(np.float64)
    lev_hpa = temperature.lev.values.astype(np.float64)
    pressure_pa = lev_hpa * 100.0
    theta = t * (P0_HPA / lev_hpa[None, :, None, None]) ** (R_D / CP)

    dtheta_dp = pressure_gradient(theta, pressure_pa)
    n2_raw = -(G * G) * pressure_pa[None, :, None, None] * dtheta_dp / (R_D * t * theta)
    invalid = (~np.isfinite(n2_raw)) | (n2_raw < N2_FLOOR)
    n2 = n2_raw.copy()
    # Boundary-layer instability makes the Eady formula singular, especially
    # at 1000 hPa. Use the nearest stable level above before applying the
    # numerical floor; this preserves a finite local stratification without
    # manufacturing extreme near-surface growth rates.
    for ilev in range(n2.shape[1] - 1):
        replacement = n2[:, ilev + 1]
        replacement = np.where(np.isfinite(replacement) & (replacement >= N2_FLOOR), replacement, N2_FLOOR)
        n2[:, ilev] = np.where(invalid[:, ilev], replacement, n2[:, ilev])
    n2[:, -1] = np.where(invalid[:, -1], N2_FLOOR, n2[:, -1])
    nfreq = np.sqrt(n2)

    lat_rad = np.deg2rad(temperature.lat.values.astype(np.float64))
    dtheta_dphi = np.gradient(theta, lat_rad, axis=2, edge_order=2)
    dtheta_dy = dtheta_dphi / 6_371_000.0
    eady = 0.31 * G * np.abs(dtheta_dy) / (nfreq * theta)

    target_index = [int(np.where(lev_hpa == level)[0][0]) for level in PLOT_LEVELS]
    eady_target = eady[:, target_index, :, :].astype(np.float32)
    invalid_target = invalid[:, target_index, :, :]
    raw_target = n2_raw[:, target_index, :, :]
    diagnostics = []
    for itime, time in enumerate(pd.to_datetime(temperature.time.values)):
        for ilev, level in enumerate(PLOT_LEVELS):
            diagnostics.append(
                {
                    "record_type": "stability",
                    "date": str(pd.Timestamp(time).date()),
                    "level_hPa": level,
                    "n2_floor_fraction": float(invalid_target[itime, ilev].mean()),
                    "n2_raw_min_s-2": float(np.nanmin(raw_target[itime, ilev])),
                    "n2_raw_max_s-2": float(np.nanmax(raw_target[itime, ilev])),
                }
            )
    out = xr.DataArray(
        eady_target,
        dims=("time", "lev", "lat", "lon"),
        coords={
            "time": temperature.time.values,
            "lev": np.array(PLOT_LEVELS, dtype=np.int16),
            "lat": temperature.lat.values,
            "lon": temperature.lon.values,
        },
        name="eady_growth_rate",
        attrs={"units": "s-1", "long_name": "Eady baroclinic growth rate"},
    )
    return out, pd.DataFrame(diagnostics)


def build_data(overwrite: bool = False) -> Path:
    if DATA_FILE.exists() and not overwrite:
        print(f"[CACHE] {DATA_FILE}", flush=True)
        return DATA_FILE
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = target_table()
    target_doys = np.array(sorted(targets.doy_nl.unique()), dtype=np.int16)

    clim_sum = None
    clim_count = np.zeros(len(target_doys), dtype=np.int16)
    diagnostics = []
    coords = None
    doy_to_index = {int(doy): i for i, doy in enumerate(target_doys)}
    for year in BASELINE_YEARS:
        print(f"[CLIM] {year}", flush=True)
        dates = [date_from_doy_nl(year, int(doy)) for doy in target_doys]
        temp = load_dates(dates)
        try:
            eady, diag = compute_eady(temp)
            diagnostics.append(diag)
            values = eady.values.astype(np.float64)
            if clim_sum is None:
                clim_sum = np.zeros((len(target_doys),) + values.shape[1:], dtype=np.float64)
                coords = {"lev": eady.lev.values, "lat": eady.lat.values, "lon": eady.lon.values}
            for i, date in enumerate(pd.to_datetime(eady.time.values)):
                j = doy_to_index[calc_doy_nl(date)]
                clim_sum[j] += values[i]
                clim_count[j] += 1
        finally:
            temp.close()
    if clim_sum is None or coords is None:
        raise RuntimeError("No climatology fields were accumulated")
    if not np.all(clim_count == len(BASELINE_YEARS)):
        raise ValueError(f"Unexpected climatology counts: {dict(zip(target_doys, clim_count))}")
    climatology = (clim_sum / clim_count[:, None, None, None]).astype(np.float32)

    event_dates = [pd.Timestamp(value) for value in sorted(targets.target_date.unique())]
    event_temp = load_dates(event_dates)
    try:
        event_eady, event_diag = compute_eady(event_temp)
        diagnostics.append(event_diag)
    finally:
        event_temp.close()
    event_lookup = {pd.Timestamp(time).normalize(): i for i, time in enumerate(pd.to_datetime(event_eady.time.values))}

    shape = (len(PEAK_DATES), len(LAGS), len(PLOT_LEVELS), len(coords["lat"]), len(coords["lon"]))
    anomalies = np.empty(shape, dtype=np.float32)
    target_date_grid = np.empty((len(PEAK_DATES), len(LAGS)), dtype="datetime64[ns]")
    target_doy_grid = np.empty((len(PEAK_DATES), len(LAGS)), dtype=np.int16)
    for row in targets.itertuples(index=False):
        date = pd.Timestamp(row.target_date).normalize()
        field = event_eady.values[event_lookup[date]]
        clim = climatology[doy_to_index[int(row.doy_nl)]]
        anomalies[int(row.event), LAGS.index(int(row.lag))] = field - clim
        target_date_grid[int(row.event), LAGS.index(int(row.lag))] = np.datetime64(date)
        target_doy_grid[int(row.event), LAGS.index(int(row.lag))] = int(row.doy_nl)

    panel_n13 = []
    panel_2004 = []
    panel_p = []
    for center in PANEL_LAGS:
        lag_indices = [LAGS.index(lag) for lag in range(center - 2, center + 3)]
        event_windows = anomalies[:, lag_indices].mean(axis=1)
        n13_samples = event_windows[N13_INDICES]
        panel_n13.append(n13_samples.mean(axis=0))
        panel_2004.append(event_windows[CASE_2004_INDEX])
        result = stats.ttest_1samp(n13_samples, popmean=0.0, axis=0, nan_policy="omit")
        panel_p.append(result.pvalue.astype(np.float32))

    ds_out = xr.Dataset(
        {
            "eady_climatology": (("doy_nl", "lev", "lat", "lon"), climatology),
            "eady_event_anomaly": (("event", "lag", "lev", "lat", "lon"), anomalies),
            "eady_N13_center5day": (("panel_lag", "lev", "lat", "lon"), np.stack(panel_n13).astype(np.float32)),
            "eady_2004_center5day": (("panel_lag", "lev", "lat", "lon"), np.stack(panel_2004).astype(np.float32)),
            "p_value_N13_center5day": (("panel_lag", "lev", "lat", "lon"), np.stack(panel_p).astype(np.float32)),
            "target_date": (("event", "lag"), target_date_grid),
            "target_doy_nl": (("event", "lag"), target_doy_grid),
            "climatology_sample_count": (("doy_nl",), clim_count),
            "peak_date": (("event",), np.array(PEAK_DATES, dtype="datetime64[ns]")),
        },
        coords={
            "event": np.arange(len(PEAK_DATES), dtype=np.int16),
            "lag": np.array(LAGS, dtype=np.int16),
            "panel_lag": np.array(PANEL_LAGS, dtype=np.int16),
            "doy_nl": target_doys,
            "lev": coords["lev"],
            "lat": coords["lat"],
            "lon": coords["lon"],
        },
        attrs={
            "title": "ERA5 Eady growth-rate anomalies for N13 and 2004 peak event",
            "formula": "sigma_BI=0.31*g*abs(dtheta/dy)/(N*theta)",
            "theta": "T*(1000hPa/p)^(R/Cp)",
            "N2": "-g^2*p/(R*T*theta)*dtheta/dp; unstable values use nearest stable upper level, then 1e-6 s-2 floor",
            "climatology": "1981-2020 no-leap daily climatology of Eady rate",
            "N13_event_indices": ",".join(str(i) for i in N13_INDICES),
            "centered_windows": ";".join(f"{lag}:{lag-2}..{lag+2}" for lag in PANEL_LAGS),
        },
    )
    for name in ["eady_climatology", "eady_event_anomaly", "eady_N13_center5day", "eady_2004_center5day"]:
        ds_out[name].attrs["units"] = "s-1"
    ds_out["p_value_N13_center5day"].attrs["test"] = "two-sided Student t-test against zero, N=13"
    ds_out["significant_p005_N13_center5day"] = ds_out["p_value_N13_center5day"] < 0.05

    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in ["eady_climatology", "eady_event_anomaly", "eady_N13_center5day", "eady_2004_center5day", "p_value_N13_center5day"]
    }
    encoding["significant_p005_N13_center5day"] = {"zlib": True, "complevel": 4, "dtype": "int8"}
    ds_out.to_netcdf(DATA_FILE, encoding=encoding)
    ds_out.close()

    diag_all = pd.concat(diagnostics, ignore_index=True)
    stability_summary = (
        diag_all.groupby("level_hPa", as_index=False)
        .agg(
            n2_floor_fraction=("n2_floor_fraction", "mean"),
            n2_raw_min_s_2=("n2_raw_min_s-2", "min"),
            n2_raw_max_s_2=("n2_raw_max_s-2", "max"),
        )
        .assign(record_type="stability_summary", climatology_year_count=len(BASELINE_YEARS))
    )
    stability_summary.to_csv(CHECK_FILE, index=False, encoding="utf-8-sig")
    print(f"[DATA] {DATA_FILE}", flush=True)
    return DATA_FILE


def paper_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_blue_red",
        ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff", "#fff7bc", "#fee391", "#fdae61", "#f46d43", "#b30000"],
        N=256,
    )


def nice_vmax(value: float) -> float:
    for candidate in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 64, 80, 100]:
        if value <= candidate:
            return float(candidate)
    return float(np.ceil(value / 25.0) * 25.0)


def lon_label(value: float) -> str:
    value = int(round(float(value)))
    if value == 180:
        return "180°"
    return f"{value}°E" if value < 180 else f"{360 - value}°W"


def add_box(ax, transform=None) -> None:
    xs = [BOX_LON_MIN, BOX_LON_MAX, BOX_LON_MAX, BOX_LON_MIN, BOX_LON_MIN]
    ys = [BOX_LAT_MIN, BOX_LAT_MIN, BOX_LAT_MAX, BOX_LAT_MAX, BOX_LAT_MIN]
    kwargs = {"color": "green", "linewidth": 1.5, "zorder": 9}
    if transform is not None:
        kwargs["transform"] = transform
    ax.plot(xs, ys, **kwargs)


def plot_one(ds: xr.Dataset, level: int, member: str) -> tuple[Path, list[dict[str, object]]]:
    if member == "N13":
        data = ds["eady_N13_center5day"].sel(lev=level) * SECONDS_PER_DAY * 100.0
        pval = ds["p_value_N13_center5day"].sel(lev=level)
        member_label = "N=13"
    else:
        data = ds["eady_2004_center5day"].sel(lev=level) * SECONDS_PER_DAY * 100.0
        pval = None
        member_label = "2004-12-24"
    data = data.sel(lat=slice(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX), lon=slice(DISPLAY_LON_MIN, DISPLAY_LON_MAX))
    if pval is not None:
        pval = pval.sel(lat=slice(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX), lon=slice(DISPLAY_LON_MIN, DISPLAY_LON_MAX))
    vmax = nice_vmax(float(np.nanpercentile(np.abs(data.values), 98.0)))
    levels = np.linspace(-vmax, vmax, 17)

    projection = ccrs.PlateCarree(central_longitude=180) if HAS_CARTOPY else None
    data_crs = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig = plt.figure(figsize=(7.60, 7.35))
    records = []
    last = None
    for i, lag in enumerate(PANEL_LAGS):
        ax = fig.add_subplot(3, 2, i + 1, projection=projection) if HAS_CARTOPY else fig.add_subplot(3, 2, i + 1)
        field = data.sel(panel_lag=lag)
        lon = field.lon.values
        lat = field.lat.values
        if HAS_CARTOPY:
            ax.set_extent([DISPLAY_LON_MIN, DISPLAY_LON_MAX, DISPLAY_LAT_MIN, DISPLAY_LAT_MAX], crs=data_crs)
            last = ax.contourf(lon, lat, field.values, levels=levels, cmap=paper_cmap(), extend="both", transform=data_crs)
            ax.coastlines(resolution="110m", linewidth=0.65, color="black", zorder=8)
            ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25, edgecolor="black", zorder=8)
            ax.set_xticks([150, 180, 210, 240], crs=data_crs)
            ax.set_yticks([20, 40, 60, 80], crs=data_crs)
            add_box(ax, data_crs)
        else:
            ax.set_xlim(DISPLAY_LON_MIN, DISPLAY_LON_MAX)
            ax.set_ylim(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX)
            last = ax.contourf(lon, lat, field.values, levels=levels, cmap=paper_cmap(), extend="both")
            ax.set_xticks([150, 180, 210, 240])
            ax.set_yticks([20, 40, 60, 80])
            add_box(ax)
        if pval is not None:
            sig = pval.sel(panel_lag=lag).values[::2, ::2] < 0.05
            yy, xx = np.meshgrid(lat[::2], lon[::2], indexing="ij")
            scatter_kwargs = {"s": 6.0, "color": "black", "marker": "o", "linewidths": 0, "zorder": 10}
            if HAS_CARTOPY:
                scatter_kwargs["transform"] = data_crs
            ax.scatter(xx[sig], yy[sig], **scatter_kwargs)
        ax.set_xticklabels([lon_label(x) for x in [150, 180, 210, 240]] if i >= 4 else [""] * 4, fontsize=8)
        ax.set_yticklabels([f"{y}°N" for y in [20, 40, 60, 80]] if i % 2 == 0 else [""] * 4, fontsize=8)
        ax.tick_params(length=3, width=0.8, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.set_title(f"({chr(97 + i)}) lag {lag} day", loc="left", fontsize=8.2, pad=3)
        ax.set_title("EGR", loc="center", fontsize=8.0, pad=3)
        ax.set_title(member_label, loc="right", fontsize=8.0, pad=3)
        values = field.values
        records.append(
            {
                "record_type": "plot",
                "member": member_label,
                "level_hPa": level,
                "panel_lag": lag,
                "window_lags": f"{lag-2},{lag-1},{lag},{lag+1},{lag+2}",
                "vmax_1e-2_day-1": vmax,
                "min_1e-2_day-1": float(np.nanmin(values)),
                "max_1e-2_day-1": float(np.nanmax(values)),
                "mean_1e-2_day-1": float(np.nanmean(values)),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "significant_fraction_p005": float((pval.sel(panel_lag=lag).values < 0.05).mean()) if pval is not None else np.nan,
            }
        )
    if last is None:
        raise RuntimeError("No Eady panels were plotted")
    cax = fig.add_axes([0.22, 0.086, 0.56, 0.025])
    cb = fig.colorbar(last, cax=cax, orientation="horizontal", ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.ax.tick_params(labelsize=8, length=2.5, pad=1)
    cb.set_label(r"$10^{-2}$ day$^{-1}$" + f"\n{level} hPa", fontsize=8.5, labelpad=2)
    cb.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.073, right=0.988, bottom=0.175, top=0.955, hspace=0.20, wspace=0.005)
    figure_dir = OUT_ROOT / "figures" / member
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"ERA5_Eady_{level}hPa_{member}_lag0_25_center5day_3x2.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[FIG] {path}", flush=True)
    return path, records


def plot_all(data_file: Path) -> list[Path]:
    with xr.open_dataset(data_file) as ds:
        ds = ds.load()
    paths = []
    records = []
    for level in PLOT_LEVELS:
        for member in ["N13", "2004-12-24"]:
            path, rows = plot_one(ds, level, member)
            paths.append(path)
            records.extend(rows)
    ds.close()

    plot_df = pd.DataFrame(records)
    if CHECK_FILE.exists():
        old = pd.read_csv(CHECK_FILE)
        old = old[old.get("record_type", "") != "plot"] if "record_type" in old else old
        check = pd.concat([old, plot_df], ignore_index=True, sort=False)
    else:
        check = plot_df
    check.to_csv(CHECK_FILE, index=False, encoding="utf-8-sig")
    print(f"[CHECK] {CHECK_FILE}", flush=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Plot from the existing NetCDF cache")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite this workflow's NetCDF cache")
    args = parser.parse_args()
    if args.plot_only:
        if not DATA_FILE.exists():
            raise FileNotFoundError(DATA_FILE)
        data_file = DATA_FILE
    else:
        data_file = build_data(overwrite=args.overwrite)
    paths = plot_all(data_file)
    if len(paths) != 16:
        raise RuntimeError(f"Expected 16 figures, generated {len(paths)}")
    print("[DONE] Generated 16 Eady growth-rate figures", flush=True)


if __name__ == "__main__":
    main()
