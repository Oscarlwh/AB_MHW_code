from __future__ import annotations

import argparse
import gc
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


AUTHOR_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTHOR_CODE))
from Laplace_SOR import Laplac_SOR  # noqa: E402


ERA5_ROOT = Path("/path/to/data/ERA5")
ERA5_FILLED_17LEV_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly_17lev")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_qd_peak_lag0_7_author_sor_17lev")

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
SOR_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
P_PA = np.array(LEVELS, dtype=np.float64) * 100.0
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
GROUPS = {
    "all_N14": list(range(14)),
    "group1_N12": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13],
    "group2_N2": [6, 11],
}
EVENT_MEMBERS = {
    "event06_2004-12-24": 6,
    "event11_2020-11-13": 11,
}
PLOT_MEMBERS = ["all_N14", "group1_N12", "group2_N2", "event06_2004-12-24", "event11_2020-11-13"]
PROJECT_PLOT_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]

R_D = 287.0
CP = 1004.0
EARTH_RADIUS = 6_371_000.0

# Inversion domain is larger than the displayed project domain.
LAT_MIN, LAT_MAX = 10.0, 85.0
LON_MIN, LON_MAX = 120.0, 260.0
DISPLAY_LAT_MIN, DISPLAY_LAT_MAX = 20.0, 80.0
DISPLAY_LON_MIN, DISPLAY_LON_MAX = 150.0, 250.0
BOX_LON_MIN, BOX_LON_MAX = 200.0, 225.0
BOX_LAT_MIN, BOX_LAT_MAX = 40.0, 50.0
VARS = ["t", "u", "v", "w"]


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


def date_from_doy_nl(year: int, doy_nl: int) -> pd.Timestamp:
    base = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=int(doy_nl) - 1)
    return pd.Timestamp(year=year, month=base.month, day=base.day)


def target_table() -> pd.DataFrame:
    rows = []
    for event, peak in enumerate(PEAK_DATES):
        peak_ts = pd.Timestamp(peak)
        for lag in LAGS:
            target = peak_ts + pd.Timedelta(days=lag)
            rows.append({"event": event, "peak_date": peak_ts, "lag": lag, "target_date": target, "doy_nl": calc_doy_nl(target)})
    return pd.DataFrame(rows)


def raw_dates_for_qd(center: pd.Timestamp) -> list[pd.Timestamp]:
    center = pd.Timestamp(center).normalize()
    return [center + pd.Timedelta(days=dd) for dd in (-2, -1, 0, 1)]


def month_starts_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS"))


def filled_month_file(year: int, month: int) -> Path:
    return ERA5_FILLED_17LEV_ROOT / str(year) / f"{year}{month:02d}.nc"


def monthly_files_available(year: int, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return all((filled_month_file(year, m.month).exists() for m in month_starts_between(start, end)))


def normalize_era5(ds: xr.Dataset, variables: list[str], lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.Dataset:
    rename = {}
    for old, new in {"valid_time": "time", "pressure_level": "lev", "level": "lev", "latitude": "lat", "longitude": "lon"}.items():
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    ds = ds.rename(rename)
    missing = [name for name in variables if name not in ds]
    if missing:
        raise KeyError(f"Missing ERA5 variables: {missing}")
    out = ds[variables].sel(lev=LEVELS)
    if float(out.lat.values[0]) > float(out.lat.values[-1]):
        out = out.sortby("lat")
    if float(out.lon.max()) <= 180.0:
        out = out.assign_coords(lon=out.lon % 360).sortby("lon")
    return out.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).transpose("time", "lev", "lat", "lon")


def open_year_window(year: int, start: pd.Timestamp, end: pd.Timestamp, variables: list[str]) -> xr.Dataset:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if monthly_files_available(year, start, end):
        opened, parts = [], []
        try:
            for ms in month_starts_between(start, end):
                ds = xr.open_dataset(filled_month_file(year, ms.month))
                opened.append(ds)
                parts.append(normalize_era5(ds, variables, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX))
            out = xr.concat(parts, dim="time").sortby("time").sel(time=slice(start, end)).load()
        finally:
            for ds in opened:
                ds.close()
        out.attrs["source_kind"] = "filled_17lev_monthly"
        return out

    path = ERA5_ROOT / f"{year}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        out = normalize_era5(ds, variables, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX).sel(time=slice(start, end)).load()
    out.attrs["source_kind"] = "annual_existing"
    return out


def load_raw_fields(dates: list[pd.Timestamp]) -> xr.Dataset:
    by_year: dict[int, list[pd.Timestamp]] = defaultdict(list)
    for d in sorted({pd.Timestamp(x).normalize() for x in dates}):
        by_year[d.year].append(d)
    parts = []
    for year, vals in sorted(by_year.items()):
        start, end = min(vals), max(vals)
        print(f"[LOAD] year={year} {start.date()}..{end.date()} days={len(vals)}", flush=True)
        year_file = ERA5_ROOT / f"{year}.nc"
        requested_months = sorted({pd.Timestamp(v).month for v in vals})
        has_requested_months = all(filled_month_file(year, month).exists() for month in requested_months)
        if (not year_file.exists()) and has_requested_months:
            for month in requested_months:
                month_vals = [v for v in vals if pd.Timestamp(v).month == month]
                month_start, month_end = min(month_vals), max(month_vals)
                ds = open_year_window(year, month_start, month_end, VARS)
                idx = np.array(sorted(set(month_vals)), dtype="datetime64[ns]")
                parts.append(ds.sel(time=idx).load())
                ds.close()
        else:
            ds = open_year_window(year, start, end, VARS)
            idx = np.array(sorted(set(vals)), dtype="datetime64[ns]")
            parts.append(ds.sel(time=idx).load())
            ds.close()
    out = xr.concat(parts, dim="time").sortby("time").load()
    for ds in parts:
        ds.close()
    return out


def compute_qd_from_smoothed_variables(ds: xr.Dataset, centers: list[pd.Timestamp]) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    ds = ds.sortby("time")
    fields = {pd.Timestamp(t).normalize(): ds.sel(time=t).drop_vars("time", errors="ignore") for t in ds.time.values}
    lat = ds.lat.values.astype(np.float64)
    lon = ds.lon.values.astype(np.float64)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    qd_list = []
    sigma0_list = []
    sigma1_list = []
    out_dates = []
    for center in centers:
        center = pd.Timestamp(center).normalize()
        raw = raw_dates_for_qd(center)
        missing = [d for d in raw if d not in fields]
        if missing:
            raise KeyError(f"Missing raw dates for {center.date()}: {missing}")
        cur = xr.concat([fields[center + pd.Timedelta(days=dd)] for dd in (-1, 0, 1)], dim="smooth_day").mean("smooth_day")
        prv = xr.concat([fields[center + pd.Timedelta(days=dd)] for dd in (-2, -1, 0)], dim="smooth_day").mean("smooth_day")

        t = cur["t"].values.astype(np.float64)
        u = cur["u"].values.astype(np.float64)
        v = cur["v"].values.astype(np.float64)
        omega = cur["w"].values.astype(np.float64)
        dtdt = (t - prv["t"].values.astype(np.float64)) / 86400.0
        dtdp = np.gradient(t, P_PA, axis=0, edge_order=1)
        dtdphi = np.gradient(t, lat_rad, axis=1, edge_order=1)
        dtdlambda = np.gradient(t, lon_rad, axis=2, edge_order=1)
        dtdy = dtdphi / EARTH_RADIUS
        coslat = np.cos(lat_rad)
        dtdx = np.zeros_like(dtdlambda)
        np.divide(dtdlambda, EARTH_RADIUS * coslat[None, :, None], out=dtdx, where=(np.abs(coslat) > 1e-8)[None, :, None])
        p_b = P_PA[:, None, None]
        sigma0 = R_D * t / (CP * p_b) - dtdp
        qd = dtdt + u * dtdx + v * dtdy - omega * sigma0
        alpha = R_D * t / p_b
        sigma1 = alpha / t * sigma0
        qd_list.append(qd.astype("float32"))
        sigma0_list.append(sigma0.astype("float32"))
        sigma1_list.append(sigma1.astype("float32"))
        out_dates.append(center)
    coords = {"date": np.array(out_dates, dtype="datetime64[ns]"), "lev": np.array(LEVELS, dtype=np.int32), "lat": ds.lat.values, "lon": ds.lon.values}
    qd_da = xr.DataArray(np.stack(qd_list), dims=("date", "lev", "lat", "lon"), coords=coords, name="Qd", attrs={"units": "K s-1"})
    sig0_da = xr.DataArray(np.stack(sigma0_list), dims=("date", "lev", "lat", "lon"), coords=coords, name="sigma0")
    sig1_da = xr.DataArray(np.stack(sigma1_list), dims=("date", "lev", "lat", "lon"), coords=coords, name="sigma1")
    return qd_da, sig0_da, sig1_da


def compute_qd_for_centers(centers: list[pd.Timestamp]) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    centers = sorted({pd.Timestamp(x).normalize() for x in centers})
    raw_dates = []
    for c in centers:
        raw_dates.extend(raw_dates_for_qd(c))
    ds = load_raw_fields(raw_dates)
    try:
        return compute_qd_from_smoothed_variables(ds, centers)
    finally:
        ds.close()
        gc.collect()


def build_forcing(out_root: Path, overwrite: bool) -> xr.Dataset:
    tag = lag_tag()
    out_file = out_root / f"ERA5_Qd_peak_{tag}_author_smoothvars_17lev_forcing.nc"
    check_file = out_root / f"ERA5_Qd_peak_{tag}_author_smoothvars_17lev_forcing_check.csv"
    if out_file.exists() and not overwrite:
        return xr.open_dataset(out_file)

    targets = target_table()
    unique_doys = sorted(targets.doy_nl.unique().astype(int).tolist())
    event_centers = sorted(pd.to_datetime(targets.target_date).unique())
    qd_event_days, _, _ = compute_qd_for_centers([pd.Timestamp(x) for x in event_centers])

    n_event, n_lag = len(PEAK_DATES), len(LAGS)
    nlev, nlat, nlon = len(LEVELS), qd_event_days.sizes["lat"], qd_event_days.sizes["lon"]
    event_lag = np.full((n_event, n_lag, nlev, nlat, nlon), np.nan, dtype="float32")
    for row in targets.itertuples(index=False):
        event_lag[int(row.event), LAGS.index(int(row.lag))] = qd_event_days.sel(date=np.datetime64(row.target_date)).values

    clim_sum = np.zeros((len(unique_doys), nlev, nlat, nlon), dtype=np.float64)
    sigma0_sum = np.zeros((len(unique_doys), nlev), dtype=np.float64)
    sigma1_sum = np.zeros((len(unique_doys), nlev), dtype=np.float64)
    clim_count = np.zeros(len(unique_doys), dtype=np.int16)
    doy_index = {d: i for i, d in enumerate(unique_doys)}
    rows = []
    for year in BASELINE_YEARS:
        centers = [date_from_doy_nl(year, d) for d in unique_doys]
        qd_year, sig0_year, sig1_year = compute_qd_for_centers(centers)
        for doy, center in zip(unique_doys, centers):
            i = doy_index[doy]
            clim_sum[i] += qd_year.sel(date=np.datetime64(center)).values.astype(np.float64)
            sigma0_sum[i] += sig0_year.sel(date=np.datetime64(center)).mean(("lat", "lon"), skipna=True).values.astype(np.float64)
            sigma1_sum[i] += sig1_year.sel(date=np.datetime64(center)).mean(("lat", "lon"), skipna=True).values.astype(np.float64)
            clim_count[i] += 1
        rows.append(
            {
                "stage": "baseline",
                "year": year,
                "doy_count": len(unique_doys),
                "qd_min_1e6": float((qd_year * 1e6).min(skipna=True)),
                "qd_max_1e6": float((qd_year * 1e6).max(skipna=True)),
                "nan_count": int(np.isnan(qd_year.values).sum()),
            }
        )
        qd_year.close(); sig0_year.close(); sig1_year.close()
        gc.collect()

    clim = (clim_sum / clim_count[:, None, None, None]).astype("float32")
    sigma0_area = (sigma0_sum / clim_count[:, None]).astype("float32")
    sigma1_area = (sigma1_sum / clim_count[:, None]).astype("float32")
    anom = np.full_like(event_lag, np.nan, dtype="float32")
    for row in targets.itertuples(index=False):
        anom[int(row.event), LAGS.index(int(row.lag))] = event_lag[int(row.event), LAGS.index(int(row.lag))] - clim[doy_index[int(row.doy_nl)]]
    group_comps = []
    group_counts = []
    for _, indices in GROUPS.items():
        group_anom = anom[indices]
        group_comps.append(np.nanmean(group_anom, axis=0).astype("float32"))
        group_counts.append(np.sum(np.isfinite(group_anom[:, :, 0, 0, 0]), axis=0).astype(np.int16))
    comp = np.stack(group_comps, axis=0).astype("float32")
    group_count = np.stack(group_counts, axis=0).astype(np.int16)

    target_date = np.empty((n_event, n_lag), dtype="datetime64[ns]")
    target_doy = np.empty((n_event, n_lag), dtype=np.int16)
    for row in targets.itertuples(index=False):
        target_date[int(row.event), LAGS.index(int(row.lag))] = np.datetime64(pd.Timestamp(row.target_date))
        target_doy[int(row.event), LAGS.index(int(row.lag))] = int(row.doy_nl)

    ds = xr.Dataset(
        {
            "Qd_event_lag": (("event", "lag", "lev", "lat", "lon"), event_lag),
            "Qd_climatology_doy": (("doy_nl", "lev", "lat", "lon"), clim),
            "Qd_anomaly_event_lag": (("event", "lag", "lev", "lat", "lon"), anom),
            "Qd_anomaly_composite_lag_mean": (("group", "lag", "lev", "lat", "lon"), comp),
            "Qd_anomaly_valid_event_count": (("group", "lag"), group_count),
            "sigma0_area_mean_doy": (("doy_nl", "lev"), sigma0_area),
            "sigma1_area_mean_doy": (("doy_nl", "lev"), sigma1_area),
            "climatology_valid_year_count": ("doy_nl", clim_count),
            "peak_date": ("event", np.array([np.datetime64(pd.Timestamp(d)) for d in PEAK_DATES])),
            "target_date": (("event", "lag"), target_date),
            "target_doy_nl": (("event", "lag"), target_doy),
        },
        coords={
            "event": np.arange(n_event, dtype=np.int16),
            "group": np.array(list(GROUPS), dtype=object),
            "lag": np.array(LAGS, dtype=np.int16),
            "lev": np.array(LEVELS, dtype=np.int32),
            "lat": qd_event_days.lat.values,
            "lon": qd_event_days.lon.values,
            "doy_nl": np.array(unique_doys, dtype=np.int16),
        },
        attrs={
            "title": f"ERA5 Qd forcing for project peak {tag}, author-style smoothed-variable residual",
            "baseline_period": "1981-2020",
            "smooth": "T/u/v/w are 3-day smoothed before Qd is computed",
            "dTdt": "backward difference of smoothed T, matching Qdiabatic.m logic",
            "domain": f"{LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E",
            "peak_dates": ",".join(PEAK_DATES),
            "groups": "; ".join(f"{name}:{indices}" for name, indices in GROUPS.items()),
        },
    )
    out_root.mkdir(parents=True, exist_ok=True)
    enc = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in ["Qd_event_lag", "Qd_climatology_doy", "Qd_anomaly_event_lag", "Qd_anomaly_composite_lag_mean"]}
    ds.to_netcdf(out_file, encoding=enc)
    rows.append(
        {
            "stage": "event_anomaly",
            "year": "events",
            "doy_count": len(unique_doys),
            "qd_min_1e6": float(np.nanmin(event_lag) * 1e6),
            "qd_max_1e6": float(np.nanmax(event_lag) * 1e6),
            "anom_min_1e6": float(np.nanmin(anom) * 1e6),
            "anom_max_1e6": float(np.nanmax(anom) * 1e6),
            "comp_min_1e6": float(np.nanmin(comp) * 1e6),
            "comp_max_1e6": float(np.nanmax(comp) * 1e6),
            "nan_count": int(np.isnan(anom).sum()),
        }
    )
    pd.DataFrame(rows).to_csv(check_file, index=False, encoding="utf-8-sig")
    qd_event_days.close()
    return xr.open_dataset(out_file)


def events_for_member(member: str) -> list[int]:
    if member in GROUPS:
        return GROUPS[member]
    if member in EVENT_MEMBERS:
        return [EVENT_MEMBERS[member]]
    raise ValueError(f"Unknown member: {member}")


def sigma_for_lag(forcing_ds: xr.Dataset, lag: int, mode: str, member: str) -> np.ndarray:
    var = "sigma0_area_mean_doy" if mode == "sigma0" else "sigma1_area_mean_doy"
    doys = [int(forcing_ds.target_doy_nl.sel(event=e, lag=lag).values) for e in events_for_member(member)]
    sigma = forcing_ds[var].sel(doy_nl=doys, lev=SOR_LEVELS).mean("doy_nl").values.astype(np.float64)
    return np.maximum(sigma, 1e-12 if mode == "sigma1" else 1e-7)


def qd_for_member_lag(forcing_ds: xr.Dataset, member: str, lag: int) -> np.ndarray:
    if member in GROUPS:
        return forcing_ds.Qd_anomaly_composite_lag_mean.sel(group=member, lag=lag, lev=SOR_LEVELS).values.astype(np.float64)
    event = EVENT_MEMBERS[member]
    return forcing_ds.Qd_anomaly_event_lag.sel(event=event, lag=lag, lev=SOR_LEVELS).values.astype(np.float64)


def invert_author_sor(forcing_ds: xr.Dataset, out_root: Path, mode: str, overwrite: bool, sc: float) -> xr.Dataset:
    sc_tag = str(sc).replace(".", "p")
    tag = lag_tag()
    out_file = out_root / f"ERA5_Qd_peak_{tag}_phi_tendency_author_SOR_1000-100hPa_{mode}_SC{sc_tag}.nc"
    check_file = out_root / f"ERA5_Qd_peak_{tag}_phi_tendency_author_SOR_1000-100hPa_{mode}_SC{sc_tag}_check.csv"
    if out_file.exists() and not overwrite:
        return xr.open_dataset(out_file)
    lat = forcing_ds.lat.values
    lon = forcing_ds.lon.values
    levels = np.array(SOR_LEVELS, dtype=np.float64)
    members = np.array(PLOT_MEMBERS, dtype=object)
    phi = np.full((len(members), len(LAGS), len(SOR_LEVELS), len(lat), len(lon)), np.nan, dtype=np.float32)
    rows = []
    for mi, member in enumerate(members):
        for li, lag in enumerate(LAGS):
            print(f"[SOR] {mode} member={member} lag={lag}", flush=True)
            q_lev_lat_lon = qd_for_member_lag(forcing_ds, str(member), int(lag))
            q = np.transpose(q_lev_lat_lon, (2, 1, 0))
            sigma = sigma_for_lag(forcing_ds, int(lag), mode, str(member))
            gt = Laplac_SOR(q, sigma, sc, 1e-12, np.pi / 180.0, np.pi / 180.0, levels * 100.0, lat, 1)
            arr = np.transpose(gt, (2, 1, 0)).astype(np.float32)
            phi[mi, li] = arr
            p925 = arr[SOR_LEVELS.index(925)] * 1e4
            rows.append(
                {
                    "member": str(member),
                    "lag": int(lag),
                    "mode": mode,
                    "event_indices": ",".join(str(x) for x in events_for_member(str(member))),
                    "phi925_min_1e4": float(np.nanmin(p925)),
                    "phi925_max_1e4": float(np.nanmax(p925)),
                    "phi925_mean_1e4": float(np.nanmean(p925)),
                    "nan_count": int(np.isnan(arr).sum()),
                    "inf_count": int(np.isinf(arr).sum()),
                }
            )
    ds = xr.Dataset(
        {"phi_tendency_qd": (("member", "lag", "lev", "lat", "lon"), phi)},
        coords={"member": members, "lag": np.array(LAGS, dtype=np.int16), "lev": np.array(SOR_LEVELS, dtype=np.int32), "lat": lat, "lon": lon},
        attrs={
            "title": f"ERA5 project peak {tag} Qd-induced geopotential tendency, author Laplace_SOR, {mode}",
            "units": "m2 s-3",
            "sigma_mode": mode,
            "horizontal_boundary": "fixed chi=0 in author Laplace_SOR.py",
            "vertical_interpolation": "author SOR internal 25 hPa grid from 1000 to 100 hPa",
            "input_levels": "1000-100 hPa only; source Qd cache keeps all 17 ERA5 levels",
            "relaxation_factor_SC": sc,
        },
    )
    for name in ["peak_date", "target_date", "target_doy_nl"]:
        ds[name] = forcing_ds[name]
    ds.to_netcdf(out_file)
    pd.DataFrame(rows).to_csv(check_file, index=False, encoding="utf-8-sig")
    return xr.open_dataset(out_file)


def lon_label(x: float) -> str:
    if int(x) == 180:
        return "180°"
    return f"{int(x)}°E" if x < 180 else f"{int(360 - x)}°W"


def cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_like",
        [
            "#053061",
            "#2166ac",
            "#67a9cf",
            "#d1e5f0",
            "#f7f7f7",
            "#fddbc7",
            "#ef8a62",
            "#b2182b",
            "#67001f",
        ],
        N=256,
    )


def add_box(ax, transform) -> None:
    xs = [BOX_LON_MIN, BOX_LON_MAX, BOX_LON_MAX, BOX_LON_MIN, BOX_LON_MIN]
    ys = [BOX_LAT_MIN, BOX_LAT_MIN, BOX_LAT_MAX, BOX_LAT_MAX, BOX_LAT_MIN]
    ax.plot(xs, ys, color="green", linewidth=1.8, transform=transform, zorder=9)


def member_title(member: str) -> str:
    if member == "all_N14":
        return "N=14"
    if member == "group1_N12":
        return "N=12"
    if member == "group2_N2":
        return "N=2"
    if member == "event06_2004-12-24":
        return "2004-12-24"
    if member == "event11_2020-11-13":
        return "2020-11-13"
    return member


def plot_member_level(ds: xr.Dataset, out_root: Path, member: str, level: int, mode: str, vmax: float) -> Path:
    da = (ds.phi_tendency_qd.sel(member=member, lev=level) * 1e4).sel(
        lat=slice(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX),
        lon=slice(DISPLAY_LON_MIN, DISPLAY_LON_MAX),
    )
    fig = plt.figure(figsize=(7.8, 9.2))
    levels = np.arange(-vmax, vmax + 3, 3)
    subplot_kw = {}
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree(central_longitude=180)
        data_crs = ccrs.PlateCarree()
        subplot_kw = {"projection": proj}
    else:
        data_crs = None
    im = None
    for i, lag in enumerate(LAGS):
        ax = fig.add_subplot(4, 2, i + 1, **subplot_kw)
        kwargs = {"transform": data_crs} if HAS_CARTOPY else {}
        field = da.sel(lag=lag)
        im = ax.contourf(field.lon, field.lat, field, levels=levels, cmap=cmap(), extend="both", **kwargs)
        if HAS_CARTOPY:
            ax.coastlines(resolution="110m", linewidth=0.55, color="black")
            ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.22)
            ax.set_extent([DISPLAY_LON_MIN, DISPLAY_LON_MAX, DISPLAY_LAT_MIN, DISPLAY_LAT_MAX], crs=data_crs)
            ax.set_xticks([150, 180, 210, 240], crs=data_crs)
            ax.set_yticks([20, 40, 60, 80], crs=data_crs)
        else:
            ax.set_xlim(DISPLAY_LON_MIN, DISPLAY_LON_MAX)
            ax.set_ylim(DISPLAY_LAT_MIN, DISPLAY_LAT_MAX)
            ax.set_xticks([150, 180, 210, 240])
            ax.set_yticks([20, 40, 60, 80])
        add_box(ax, data_crs if HAS_CARTOPY else ax.transData)
        ax.set_xticklabels([lon_label(x) for x in [150, 180, 210, 240]] if i >= 6 else [""] * 4, fontsize=8)
        ax.set_yticklabels([f"{int(y)}°N" for y in [20, 40, 60, 80]] if i % 2 == 0 else [""] * 4, fontsize=8)
        ax.set_title(f"({chr(97+i)}) lag {lag} days", loc="left", fontsize=10, pad=2)
        ax.set_title(member_title(member), loc="center", fontsize=10, pad=2)
        ax.set_title(f"{level} hPa", loc="right", fontsize=10, pad=2)
        ax.tick_params(length=2.5, width=0.7, pad=1.5)
    cax = fig.add_axes([0.20, 0.045, 0.60, 0.020])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks(np.arange(-vmax, vmax + 0.1, 6))
    cb.set_label(r"$10^{-4}$ m$^2$ s$^{-3}$", fontsize=10, labelpad=1)
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.970, hspace=0.22, wspace=0.035)
    fig_dir = out_root / "figures"
    level_dir = fig_dir / str(level)
    level_dir.mkdir(parents=True, exist_ok=True)
    member_safe = member.replace("/", "-")
    sc = str(ds.attrs.get("relaxation_factor_SC", "")).replace(".", "p")
    sc_tag = f"_SC{sc}" if sc else ""
    out = level_dir / f"ERA5_Qd_phi_tendency_{level}hPa_{member_safe}_lag0_7_author_SOR_1000-100hPa_{mode}{sc_tag}_lat20-80_papercolor_vmax{int(vmax)}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all(ds: xr.Dataset, out_root: Path, mode: str, vmax: float) -> pd.DataFrame:
    rows = []
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for level in PROJECT_PLOT_LEVELS:
        if level not in ds.lev.values:
            continue
        for member in PLOT_MEMBERS:
            out = plot_member_level(ds, out_root, member, level, mode, vmax)
            da = ds.phi_tendency_qd.sel(member=member, lev=level) * 1e4
            rows.append(
                {
                    "member": member,
                    "level": level,
                    "png": str(out),
                    "min_1e4": float(da.min(skipna=True)),
                    "max_1e4": float(da.max(skipna=True)),
                    "mean_1e4": float(da.mean(skipna=True)),
                }
            )
            print(f"[FIG] {out}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(fig_dir / "ERA5_Qd_peak_lag0_7_author_SOR_plot_summary_lat20-80_papercolor.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-inversion", action="store_true")
    p.add_argument("--modes", nargs="+", default=["sigma1"], choices=["sigma0", "sigma1"])
    p.add_argument("--vmax", type=float, default=18.0)
    p.add_argument("--sc", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    forcing = build_forcing(args.out_root, args.overwrite)
    if args.skip_inversion:
        print("[DONE] forcing only", flush=True)
        return
    for mode in args.modes:
        ds = invert_author_sor(forcing, args.out_root, mode, args.overwrite, args.sc)
        plot_all(ds, args.out_root, mode, args.vmax)
        ds.close()
    forcing.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
