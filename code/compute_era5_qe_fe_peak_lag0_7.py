from __future__ import annotations

import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


ERA5_ROOT = Path("/path/to/data/ERA5")
ERA5_FILLED_8LEV_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_qe_fe_peak_lag0_7")
CACHE_TAG = "8lev"
MEMBERS_FILE = OUT_ROOT / f"ERA5_Qe_Fe_peak_lag0_7_members_12plus1plus1_{CACHE_TAG}.nc"

VARS = ["t", "u", "v", "w"]
LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]
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
LAGS = list(range(8))
BASELINE_YEARS = list(range(1981, 2021))
GROUP1_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
PLOT_MEMBERS = {
    "group1_N12": GROUP1_INDICES,
    "event06_2004-12-24": [6],
    "event11_2020-11-13": [11],
}
PLOT_LABELS = {
    "group1_N12": "N=12",
    "event06_2004-12-24": "2004-12-24",
    "event11_2020-11-13": "2020-11-13",
}
FILE_LABELS = {
    "group1_N12": "N12",
    "event06_2004-12-24": "2004-12-24",
    "event11_2020-11-13": "2020-11-13",
}

LAT_MIN, LAT_MAX = 10.0, 85.0
LON_MIN, LON_MAX = 120.0, 260.0
DISPLAY_LAT_MIN, DISPLAY_LAT_MAX = 20.0, 80.0
DISPLAY_LON_MIN, DISPLAY_LON_MAX = 150.0, 250.0
BOX_LON_MIN, BOX_LON_MAX = 200.0, 225.0
BOX_LAT_MIN, BOX_LAT_MAX = 40.0, 50.0
PLOT_LEVEL = 925

EARTH_RADIUS = 6_371_000.0
R_D = 287.0
CP = 1004.0
SEASON_DATES = pd.date_range("2001-10-01", "2002-02-28", freq="D")
SEASON_DAY_COUNT = len(SEASON_DATES)


def lag_tag() -> str:
    if LAGS == list(range(int(LAGS[0]), int(LAGS[-1]) + 1)):
        return f"lag{int(LAGS[0])}_{int(LAGS[-1])}"
    return "lag" + "-".join(str(int(x)) for x in LAGS)


def calc_doy_nl(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    if ts.month == 2 and ts.day == 29:
        raise ValueError("Feb 29 is excluded from no-leap climatology.")
    doy = int(ts.dayofyear)
    if ts.is_leap_year and doy > 60:
        doy -= 1
    return doy


def season_day_for_date(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    dummy_year = 2001 if ts.month >= 10 else 2002
    dummy = pd.Timestamp(year=dummy_year, month=ts.month, day=ts.day)
    return int((dummy - SEASON_DATES[0]).days)


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
                    "season_year": int(peak_ts.year),
                }
            )
    return pd.DataFrame(rows)


def lon_label(x: float) -> str:
    x = int(round(float(x)))
    if x == 180:
        return "180°"
    return f"{x}°E" if x < 180 else f"{360 - x}°W"


def filled_month_file(year: int, month: int) -> Path:
    return ERA5_FILLED_8LEV_ROOT / str(year) / f"{year}{month:02d}.nc"


def normalize_era5(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in {"valid_time": "time", "pressure_level": "lev", "level": "lev", "latitude": "lat", "longitude": "lon"}.items():
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    ds = ds.rename(rename)
    missing = [name for name in VARS if name not in ds]
    if missing:
        raise KeyError(f"Missing ERA5 variables: {missing}")
    out = ds[VARS].sel(lev=LEVELS)
    if float(out.lat.values[0]) > float(out.lat.values[-1]):
        out = out.sortby("lat")
    if float(out.lon.max()) <= 180.0:
        out = out.assign_coords(lon=out.lon % 360).sortby("lon")
    out = out.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX)).transpose("time", "lev", "lat", "lon")
    return out.reset_coords(names=[name for name in out.coords if name not in out.dims], drop=True)


def load_month(year: int, month: int) -> xr.Dataset:
    filled = filled_month_file(year, month)
    if filled.exists():
        with xr.open_dataset(filled) as ds:
            return normalize_era5(ds).load()
    path = ERA5_ROOT / f"{year}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    with xr.open_dataset(path) as ds:
        return normalize_era5(ds).sel(time=slice(start, end)).load()


def season_months(season_year: int, truncate_2023: bool = False) -> list[tuple[int, int]]:
    months = [(season_year, 10), (season_year, 11), (season_year, 12)]
    if not (truncate_2023 and season_year == 2023):
        months.extend([(season_year + 1, 1), (season_year + 1, 2)])
    return months


def load_season(season_year: int, truncate_2023: bool = False) -> xr.Dataset:
    parts = [load_month(year, month) for year, month in season_months(season_year, truncate_2023=truncate_2023)]
    try:
        ds = xr.concat(parts, dim="time", coords="minimal", compat="override").sortby("time").load()
        if truncate_2023 and season_year == 2023:
            dates = pd.date_range("2023-10-01", "2023-12-31", freq="D")
        else:
            dates = pd.date_range(f"{season_year}-10-01", f"{season_year + 1}-02-28", freq="D")
        ds = ds.sel(time=dates)
        return ds.load()
    finally:
        for part in parts:
            part.close()


def build_variable_climatology(overwrite: bool) -> xr.Dataset:
    out_file = OUT_ROOT / f"ERA5_variable_daily_climatology_OctFeb_{CACHE_TAG}_1981_2020.nc"
    check_file = OUT_ROOT / f"ERA5_variable_daily_climatology_OctFeb_{CACHE_TAG}_1981_2020_check.csv"
    if out_file.exists() and not overwrite:
        print(f"[CACHE] {out_file}", flush=True)
        return xr.open_dataset(out_file)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sums = None
    counts = np.zeros(SEASON_DAY_COUNT, dtype=np.int16)
    records = []
    coords = None
    for year in BASELINE_YEARS:
        for month in [1, 2, 10, 11, 12]:
            print(f"[VAR-CLIM] {year}-{month:02d}", flush=True)
            ds = load_month(year, month)
            try:
                if sums is None:
                    coords = {
                        "season_day": np.arange(SEASON_DAY_COUNT, dtype=np.int16),
                        "lev": ds.lev.values,
                        "lat": ds.lat.values,
                        "lon": ds.lon.values,
                    }
                    sums = {name: np.zeros((SEASON_DAY_COUNT, len(coords["lev"]), len(coords["lat"]), len(coords["lon"])), dtype=np.float64) for name in VARS}
                for i, tval in enumerate(pd.to_datetime(ds.time.values)):
                    if tval.month == 2 and tval.day == 29:
                        continue
                    sd = season_day_for_date(tval)
                    for name in VARS:
                        sums[name][sd] += ds[name].isel(time=i).values.astype(np.float64)
                    counts[sd] += 1
                records.append({"year": year, "month": month, "n_time": int(ds.sizes["time"])})
            finally:
                ds.close()
                gc.collect()

    if sums is None or coords is None:
        raise RuntimeError("No climatology data loaded")
    if int(counts.min()) != len(BASELINE_YEARS):
        raise ValueError(f"Unexpected variable climatology counts: min={counts.min()} max={counts.max()}")

    data_vars = {}
    for name in VARS:
        data_vars[f"{name}_clim"] = (("season_day", "lev", "lat", "lon"), (sums[name] / counts[:, None, None, None]).astype("float32"))
    clim = xr.Dataset(
        data_vars=data_vars,
        coords={**coords, "count": ("season_day", counts.astype(np.int16))},
        attrs={
            "title": "ERA5 Oct-Feb variable daily climatology for project Qe/Fe",
            "baseline_period": "1981-2020",
            "levels_hPa": ",".join(str(x) for x in LEVELS),
            "domain": f"{LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E",
            "created_by": Path(__file__).name,
        },
    )
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in clim.data_vars if name.endswith("_clim")}
    clim.to_netcdf(out_file, encoding=encoding)
    pd.DataFrame(records).to_csv(check_file, index=False, encoding="utf-8-sig")
    print(f"[NC] {out_file}", flush=True)
    print(f"[CSV] {check_file}", flush=True)
    return xr.open_dataset(out_file)


def bandpass_3_8_days(arr: np.ndarray) -> np.ndarray:
    b, a = signal.butter(2, [1 / 8 / 0.5, 1 / 3 / 0.5], btype="bandpass")
    return signal.filtfilt(b, a, arr, axis=0).astype("float32")


def horizontal_gradients(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat.astype(np.float64))
    lon_rad = np.deg2rad(lon.astype(np.float64))
    d_dphi = np.gradient(field, lat_rad, axis=2, edge_order=1)
    d_dlambda = np.gradient(field, lon_rad, axis=3, edge_order=1)
    d_dy = d_dphi / EARTH_RADIUS
    coslat = np.cos(lat_rad)
    d_dx = np.zeros_like(d_dlambda, dtype=np.float64)
    np.divide(d_dlambda, EARTH_RADIUS * coslat[None, None, :, None], out=d_dx, where=(np.abs(coslat) > 1e-8)[None, None, :, None])
    return d_dx.astype("float32"), d_dy.astype("float32")


def anomaly_dataset(ds: xr.Dataset, var_clim: xr.Dataset) -> xr.Dataset:
    sdays = np.array([season_day_for_date(t) for t in pd.to_datetime(ds.time.values)], dtype=np.int16)
    data_vars = {}
    for name in VARS:
        clim_arr = var_clim[f"{name}_clim"].sel(season_day=sdays).values.astype("float32")
        data_vars[name] = (("time", "lev", "lat", "lon"), ds[name].values.astype("float32") - clim_arr)
    return xr.Dataset(data_vars, coords=ds.coords)


def compute_qe_fe(ds_anom: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    lat = ds_anom.lat.values.astype(np.float64)
    lon = ds_anom.lon.values.astype(np.float64)
    lev_pa = ds_anom.lev.values.astype(np.float64) * 100.0

    u = bandpass_3_8_days(ds_anom["u"].values.astype("float32"))
    v = bandpass_3_8_days(ds_anom["v"].values.astype("float32"))
    t = bandpass_3_8_days(ds_anom["t"].values.astype("float32"))
    omega = ds_anom["w"].values.astype("float32")

    ut = u * t
    vt = v * t
    wt = omega * t
    dut_dx, _ = horizontal_gradients(ut, lat, lon)
    _, dvt_dy = horizontal_gradients(vt, lat, lon)
    dwt_dp = np.gradient(wt, lev_pa, axis=1, edge_order=1).astype("float32")
    p4 = lev_pa[None, :, None, None].astype("float32")
    qe = -(dut_dx + dvt_dy) - dwt_dp + (R_D * omega * t) / (CP * p4)

    du_dx, du_dy = horizontal_gradients(u, lat, lon)
    dv_dx, _ = horizontal_gradients(v, lat, lon)
    zeta = dv_dx - du_dy
    duz_dx, _ = horizontal_gradients(u * zeta, lat, lon)
    _, dvz_dy = horizontal_gradients(v * zeta, lat, lon)
    fe = -(duz_dx + dvz_dy)

    coords = {"time": ds_anom.time, "lev": ds_anom.lev, "lat": ds_anom.lat, "lon": ds_anom.lon}
    return (
        xr.DataArray(qe.astype("float32"), dims=("time", "lev", "lat", "lon"), coords=coords, name="Qe", attrs={"units": "K s-1"}),
        xr.DataArray(fe.astype("float32"), dims=("time", "lev", "lat", "lon"), coords=coords, name="Fe", attrs={"units": "s-2"}),
    )


def build_qe_fe_daily_climatology(var_clim: xr.Dataset, target: pd.DataFrame, overwrite: bool) -> xr.Dataset:
    out_file = OUT_ROOT / f"ERA5_Qe_Fe_daily_climatology_target_doys_{CACHE_TAG}_1981_2020.nc"
    check_file = OUT_ROOT / f"ERA5_Qe_Fe_daily_climatology_target_doys_{CACHE_TAG}_1981_2020_check.csv"
    if out_file.exists() and not overwrite:
        print(f"[CACHE] {out_file}", flush=True)
        return xr.open_dataset(out_file)

    doys = sorted(int(x) for x in target["doy_nl"].unique())
    sum_qe = None
    sum_fe = None
    counts = {doy: 0 for doy in doys}
    records = []
    coords = None
    for season_year in BASELINE_YEARS:
        print(f"[QEFE-CLIM] season {season_year}", flush=True)
        ds = load_season(season_year)
        try:
            ds_anom = anomaly_dataset(ds, var_clim)
            qe, fe = compute_qe_fe(ds_anom)
            qe = qe.sel(lat=slice(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX), lon=slice(DISPLAY_LON_MIN, DISPLAY_LON_MAX))
            fe = fe.sel(lat=slice(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX), lon=slice(DISPLAY_LON_MIN, DISPLAY_LON_MAX))
            if sum_qe is None:
                coords = {"doy_nl": np.array(doys, dtype=np.int16), "lev": qe.lev.values, "lat": qe.lat.values, "lon": qe.lon.values}
                shape = (len(doys), qe.sizes["lev"], qe.sizes["lat"], qe.sizes["lon"])
                sum_qe = np.zeros(shape, dtype=np.float64)
                sum_fe = np.zeros(shape, dtype=np.float64)
            time_index = {calc_doy_nl(t): i for i, t in enumerate(pd.to_datetime(qe.time.values))}
            for di, doy in enumerate(doys):
                if doy not in time_index:
                    continue
                ti = time_index[doy]
                sum_qe[di] += qe.isel(time=ti).values.astype(np.float64)
                sum_fe[di] += fe.isel(time=ti).values.astype(np.float64)
                counts[doy] += 1
            records.append({"season_year": season_year, "n_time": int(ds.sizes["time"])})
        finally:
            ds.close()
            gc.collect()

    if sum_qe is None or sum_fe is None or coords is None:
        raise RuntimeError("No Qe/Fe climatology data loaded")
    count_arr = np.array([counts[doy] for doy in doys], dtype=np.int16)
    if int(count_arr.min()) != len(BASELINE_YEARS):
        raise ValueError(f"Unexpected Qe/Fe climatology counts: {counts}")

    out = xr.Dataset(
        {
            "Qe_climatology": (("doy_nl", "lev", "lat", "lon"), (sum_qe / count_arr[:, None, None, None]).astype("float32")),
            "Fe_climatology": (("doy_nl", "lev", "lat", "lon"), (sum_fe / count_arr[:, None, None, None]).astype("float32")),
            "count": ("doy_nl", count_arr),
        },
        coords=coords,
        attrs={
            "title": "ERA5 Qe/Fe daily climatology for project lag target DOYs",
            "baseline_period": "1981-2020 season years",
            "bandpass": "3-8 day Butterworth order 2 on daily anomaly fields",
            "created_by": Path(__file__).name,
        },
    )
    encoding = {"Qe_climatology": {"zlib": True, "complevel": 4, "dtype": "float32"}, "Fe_climatology": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    out.to_netcdf(out_file, encoding=encoding)
    pd.DataFrame(records).to_csv(check_file, index=False, encoding="utf-8-sig")
    print(f"[NC] {out_file}", flush=True)
    print(f"[CSV] {check_file}", flush=True)
    return xr.open_dataset(out_file)


def build_event_anomalies(var_clim: xr.Dataset, qefe_clim: xr.Dataset, target: pd.DataFrame, overwrite: bool) -> xr.Dataset:
    tag = lag_tag()
    out_file = OUT_ROOT / f"ERA5_Qe_Fe_peak_{tag}_event_anomalies_{CACHE_TAG}.nc"
    check_file = OUT_ROOT / f"ERA5_Qe_Fe_peak_{tag}_event_anomalies_{CACHE_TAG}_check.csv"
    if out_file.exists() and not overwrite:
        print(f"[CACHE] {out_file}", flush=True)
        return xr.open_dataset(out_file)

    target_by_season = {int(k): v.copy() for k, v in target.groupby("season_year")}
    n_event, n_lag = len(PEAK_DATES), len(LAGS)
    sample = qefe_clim["Qe_climatology"]
    qe_anom = np.full((n_event, n_lag, sample.sizes["lev"], sample.sizes["lat"], sample.sizes["lon"]), np.nan, dtype=np.float32)
    fe_anom = np.full_like(qe_anom, np.nan)
    records = []

    for season_year, sub in sorted(target_by_season.items()):
        print(f"[EVENT] season {season_year}", flush=True)
        ds = load_season(season_year, truncate_2023=True)
        try:
            ds_anom = anomaly_dataset(ds, var_clim)
            qe, fe = compute_qe_fe(ds_anom)
            qe = qe.sel(lat=slice(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX), lon=slice(DISPLAY_LON_MIN, DISPLAY_LON_MAX))
            fe = fe.sel(lat=slice(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX), lon=slice(DISPLAY_LON_MIN, DISPLAY_LON_MAX))
            time_lookup = {pd.Timestamp(t).normalize(): i for i, t in enumerate(pd.to_datetime(qe.time.values))}
            for row in sub.itertuples(index=False):
                target_date = pd.Timestamp(row.target_date).normalize()
                if target_date not in time_lookup:
                    raise ValueError(f"Missing target date {target_date.date()} in season {season_year}")
                ti = time_lookup[target_date]
                clim_qe = qefe_clim["Qe_climatology"].sel(doy_nl=int(row.doy_nl))
                clim_fe = qefe_clim["Fe_climatology"].sel(doy_nl=int(row.doy_nl))
                qe_field = qe.isel(time=ti) - clim_qe
                fe_field = fe.isel(time=ti) - clim_fe
                lag_index = LAGS.index(int(row.lag))
                qe_anom[int(row.event), lag_index] = qe_field.values.astype("float32")
                fe_anom[int(row.event), lag_index] = fe_field.values.astype("float32")
                records.append(
                    {
                        "event": int(row.event),
                        "peak_date": str(pd.Timestamp(row.peak_date).date()),
                        "lag": int(row.lag),
                        "target_date": str(target_date.date()),
                        "doy_nl": int(row.doy_nl),
                        "season_year": season_year,
                        "season_time_count": int(ds.sizes["time"]),
                        "qe_min_1e6": float((qe_field * 1e6).min(skipna=True)),
                        "qe_max_1e6": float((qe_field * 1e6).max(skipna=True)),
                        "fe_min_1e11": float((fe_field * 1e11).min(skipna=True)),
                        "fe_max_1e11": float((fe_field * 1e11).max(skipna=True)),
                        "nan_qe": int(np.isnan(qe_field.values).sum()),
                        "nan_fe": int(np.isnan(fe_field.values).sum()),
                    }
                )
        finally:
            ds.close()
            gc.collect()

    out = xr.Dataset(
        {
            "Qe_event_lag_anomaly": (("event", "lag", "lev", "lat", "lon"), qe_anom),
            "Fe_event_lag_anomaly": (("event", "lag", "lev", "lat", "lon"), fe_anom),
            "peak_date": ("event", np.array([np.datetime64(pd.Timestamp(d)) for d in PEAK_DATES])),
        },
        coords={"event": np.arange(n_event), "lag": LAGS, "lev": sample.lev.values, "lat": sample.lat.values, "lon": sample.lon.values},
        attrs={
            "title": f"ERA5 project peak {tag} transient eddy Qe/Fe anomalies",
            "baseline_period": "1981-2020",
            "members": "12+1+1: group1_N12, event06_2004-12-24, event11_2020-11-13",
            "created_by": Path(__file__).name,
        },
    )
    out["Qe_event_lag_anomaly"].attrs["units"] = "K s-1"
    out["Fe_event_lag_anomaly"].attrs["units"] = "s-2"
    encoding = {
        "Qe_event_lag_anomaly": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "Fe_event_lag_anomaly": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    out.to_netcdf(out_file, encoding=encoding)
    pd.DataFrame(records).to_csv(check_file, index=False, encoding="utf-8-sig")
    print(f"[NC] {out_file}", flush=True)
    print(f"[CSV] {check_file}", flush=True)
    return xr.open_dataset(out_file)


def build_composites(events: xr.Dataset, overwrite: bool) -> xr.Dataset:
    tag = lag_tag()
    out_file = OUT_ROOT / f"ERA5_Qe_Fe_peak_{tag}_members_12plus1plus1_{CACHE_TAG}.nc"
    check_file = OUT_ROOT / f"ERA5_Qe_Fe_peak_{tag}_members_12plus1plus1_{CACHE_TAG}_check.csv"
    if out_file.exists() and not overwrite:
        print(f"[CACHE] {out_file}", flush=True)
        return xr.open_dataset(out_file)

    qe_members = []
    fe_members = []
    qe_pvals = []
    fe_pvals = []
    records = []
    for member, indices in PLOT_MEMBERS.items():
        qe_event = events["Qe_event_lag_anomaly"].isel(event=indices)
        fe_event = events["Fe_event_lag_anomaly"].isel(event=indices)
        qe_comp = qe_event.mean("event", skipna=True)
        fe_comp = fe_event.mean("event", skipna=True)
        if len(indices) > 1:
            _, qe_p = stats.ttest_1samp(qe_event.values, popmean=0.0, axis=0, nan_policy="omit")
            _, fe_p = stats.ttest_1samp(fe_event.values, popmean=0.0, axis=0, nan_policy="omit")
        else:
            qe_p = np.full(qe_comp.shape, np.nan, dtype=np.float32)
            fe_p = np.full(fe_comp.shape, np.nan, dtype=np.float32)
        qe_p_da = xr.DataArray(qe_p.astype("float32"), dims=("lag", "lev", "lat", "lon"), coords=qe_comp.coords)
        fe_p_da = xr.DataArray(fe_p.astype("float32"), dims=("lag", "lev", "lat", "lon"), coords=fe_comp.coords)
        qe_members.append(qe_comp)
        fe_members.append(fe_comp)
        qe_pvals.append(qe_p_da)
        fe_pvals.append(fe_p_da)
        records.append(
            {
                "member": member,
                "n_events": len(indices),
                "event_indices": ",".join(str(x) for x in indices),
                "qe925_min_1e6": float((qe_comp.sel(lev=PLOT_LEVEL) * 1e6).min(skipna=True)),
                "qe925_max_1e6": float((qe_comp.sel(lev=PLOT_LEVEL) * 1e6).max(skipna=True)),
                "fe925_min_1e11": float((fe_comp.sel(lev=PLOT_LEVEL) * 1e11).min(skipna=True)),
                "fe925_max_1e11": float((fe_comp.sel(lev=PLOT_LEVEL) * 1e11).max(skipna=True)),
                "qe925_p005_fraction": float((qe_p_da.sel(lev=PLOT_LEVEL) < 0.05).mean(skipna=True)) if len(indices) > 1 else np.nan,
                "fe925_p005_fraction": float((fe_p_da.sel(lev=PLOT_LEVEL) < 0.05).mean(skipna=True)) if len(indices) > 1 else np.nan,
            }
        )

    members = list(PLOT_MEMBERS)
    out = xr.Dataset(
        {
            "Qe_member_lag_anomaly": xr.concat(qe_members, dim=xr.IndexVariable("member", members)).astype("float32"),
            "Fe_member_lag_anomaly": xr.concat(fe_members, dim=xr.IndexVariable("member", members)).astype("float32"),
            "Qe_p_value": xr.concat(qe_pvals, dim=xr.IndexVariable("member", members)).astype("float32"),
            "Fe_p_value": xr.concat(fe_pvals, dim=xr.IndexVariable("member", members)).astype("float32"),
        },
        attrs={"title": f"ERA5 project Qe/Fe {tag} 12+1+1 members", "significance": "p<0.05 only meaningful for group1_N12"},
    )
    out["Qe_significant_p010"] = out["Qe_p_value"] < 0.1
    out["Fe_significant_p010"] = out["Fe_p_value"] < 0.1
    out["Qe_significant_p005"] = out["Qe_p_value"] < 0.05
    out["Fe_significant_p005"] = out["Fe_p_value"] < 0.05
    out.to_netcdf(out_file)
    pd.DataFrame(records).to_csv(check_file, index=False, encoding="utf-8-sig")
    print(f"[NC] {out_file}", flush=True)
    print(f"[CSV] {check_file}", flush=True)
    return xr.open_dataset(out_file)


def paper_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_blue_red",
        ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff", "#fff7bc", "#fee391", "#fdae61", "#f46d43", "#b30000"],
        N=256,
    )


def add_box(ax, transform=None) -> None:
    xs = [BOX_LON_MIN, BOX_LON_MAX, BOX_LON_MAX, BOX_LON_MIN, BOX_LON_MIN]
    ys = [BOX_LAT_MIN, BOX_LAT_MIN, BOX_LAT_MAX, BOX_LAT_MAX, BOX_LAT_MIN]
    kwargs = {"color": "green", "linewidth": 1.5, "zorder": 9}
    if transform is not None:
        kwargs["transform"] = transform
    ax.plot(xs, ys, **kwargs)


def plot_member_maps(
    ds: xr.Dataset,
    var: str,
    member: str,
    plot_level: int,
    vmax: float,
    suffix: str = "",
) -> Path:
    data_name = f"{var}_member_lag_anomaly"
    p_name = f"{var}_p_value"
    scale = 1e6 if var == "Qe" else 1e11
    unit = r"$10^{-6}$ K s$^{-1}$" if var == "Qe" else r"$10^{-11}$ s$^{-2}$"
    da = ds[data_name].sel(member=member, lev=plot_level) * scale
    pval = ds[p_name].sel(member=member, lev=plot_level)
    levels = np.linspace(-vmax, vmax, 17)
    cmap = paper_cmap()
    data_crs = ccrs.PlateCarree() if HAS_CARTOPY else None
    proj = ccrs.PlateCarree(central_longitude=180) if HAS_CARTOPY else None
    fig = plt.figure(figsize=(7.45, 8.0))
    letters = list("abcdefgh")
    last = None
    for i, lag in enumerate(LAGS):
        ax = fig.add_subplot(4, 2, i + 1, projection=proj) if HAS_CARTOPY else fig.add_subplot(4, 2, i + 1)
        field = da.sel(lag=lag)
        lon = field.lon.values
        lat = field.lat.values
        if HAS_CARTOPY:
            ax.set_extent([DISPLAY_LON_MIN, DISPLAY_LON_MAX, DISPLAY_LAT_MIN, DISPLAY_LAT_MAX], crs=data_crs)
            last = ax.contourf(lon, lat, field.values, levels=levels, cmap=cmap, extend="both", transform=data_crs)
            ax.coastlines(resolution="110m", linewidth=0.65, color="black", zorder=8)
            ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25, edgecolor="black", zorder=8)
            ax.set_xticks([150, 180, 210, 240], crs=data_crs)
            ax.set_yticks([20, 40, 60, 80], crs=data_crs)
            add_box(ax, transform=data_crs)
            sig_transform = data_crs
        else:
            ax.set_xlim(DISPLAY_LON_MIN, DISPLAY_LON_MAX)
            ax.set_ylim(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX)
            last = ax.contourf(lon, lat, field.values, levels=levels, cmap=cmap, extend="both")
            ax.set_xticks([150, 180, 210, 240])
            ax.set_yticks([20, 40, 60, 80])
            add_box(ax)
            sig_transform = None
        if member == "group1_N12":
            sig = pval.sel(lag=lag).values < 0.05
            sig_sub = sig[::2, ::2]
            yy, xx = np.meshgrid(lat[::2], lon[::2], indexing="ij")
            ax.scatter(
                xx[sig_sub],
                yy[sig_sub],
                s=6.0,
                color="black",
                marker="o",
                linewidths=0,
                transform=sig_transform,
                zorder=10,
            )
        ax.set_xticklabels([lon_label(x) for x in [150, 180, 210, 240]] if i >= 6 else [""] * 4, fontsize=8)
        ax.set_yticklabels([f"{int(y)}°N" for y in [20, 40, 60, 80]] if i % 2 == 0 else [""] * 4, fontsize=8)
        ax.tick_params(length=3, width=0.8, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.set_title(f"({letters[i]}) lag {lag} days", loc="left", fontsize=10.5, pad=3)
        ax.text(0.50, 1.02, var, transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5)
        ax.text(0.98, 1.02, PLOT_LABELS[member], transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5)
    cax = fig.add_axes([0.22, 0.071, 0.56, 0.023])
    cb = fig.colorbar(last, cax=cax, orientation="horizontal", ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.ax.tick_params(labelsize=8, length=2.5, pad=1)
    cb.set_label(f"{unit}\n{plot_level} hPa", fontsize=8, labelpad=2)
    cb.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.145, top=0.965, hspace=0.20, wspace=0.005)
    fig_dir = OUT_ROOT / "figures" / var
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"ERA5_{var}_{plot_level}hPa_{FILE_LABELS[member]}_lag0_7_4x2{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[FIG] {out}", flush=True)
    return out


def nice_vmax(value: float) -> float:
    for candidate in [8, 15, 30, 50, 80, 120, 160, 250, 400, 800, 1200, 2000]:
        if value <= candidate:
            return float(candidate)
    return float(np.ceil(value / 500.0) * 500.0)


def plot_all(ds: xr.Dataset) -> None:
    records = []
    for var in ["Qe", "Fe"]:
        for plot_level in LEVELS:
            for member in PLOT_MEMBERS:
                data_name = f"{var}_member_lag_anomaly"
                scale = 1e6 if var == "Qe" else 1e11
                vals = np.abs((ds[data_name].sel(member=member, lev=plot_level) * scale).values)
                auto_vmax = nice_vmax(float(np.nanpercentile(vals, 98.0)))
                out_auto = plot_member_maps(
                    ds,
                    var,
                    member,
                    plot_level,
                    auto_vmax,
                    suffix=f"_autoscale_vmax{int(auto_vmax)}",
                )
                records.append(
                    {
                        "var": var,
                        "member": PLOT_LABELS[member],
                        "level": plot_level,
                        "vmax": auto_vmax,
                        "scale": "p98_autoscale",
                        "significance": "p<0.05 for N=12 only",
                        "file": str(out_auto),
                    }
                )
    pd.DataFrame(records).to_csv(OUT_ROOT / "ERA5_Qe_Fe_peak_lag0_7_plot_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-var-clim", action="store_true")
    parser.add_argument("--overwrite-qefe-clim", action="store_true")
    parser.add_argument("--overwrite-events", action="store_true")
    parser.add_argument("--overwrite-members", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Plot all eight levels from the existing member cache.")
    args = parser.parse_args()

    if args.plot_only:
        if not MEMBERS_FILE.exists():
            raise FileNotFoundError(f"Missing member cache: {MEMBERS_FILE}")
        with xr.open_dataset(MEMBERS_FILE) as members:
            plot_all(members)
        return

    target = target_table()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    target.to_csv(OUT_ROOT / "ERA5_Qe_Fe_peak_lag0_7_target_dates.csv", index=False, encoding="utf-8-sig")

    var_clim = build_variable_climatology(args.overwrite_var_clim)
    try:
        qefe_clim = build_qe_fe_daily_climatology(var_clim, target, args.overwrite_qefe_clim)
        try:
            events = build_event_anomalies(var_clim, qefe_clim, target, args.overwrite_events)
        finally:
            qefe_clim.close()
    finally:
        var_clim.close()
    try:
        members = build_composites(events, args.overwrite_members)
    finally:
        events.close()
    try:
        plot_all(members)
    finally:
        members.close()


if __name__ == "__main__":
    main()
