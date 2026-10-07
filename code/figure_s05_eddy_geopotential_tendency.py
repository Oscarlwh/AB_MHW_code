#!/usr/bin/env python3
"""Build a corrected forcing-first lag0-30 mean GTeddy section.

The older lag-2..27 Qe/Fe event file stored fields by the numeric lag value
instead of by the lag coordinate position. Its first 28 array positions still
contain the correct physical lag0..27 fields, so this script decodes those
positions, computes lag28..30 explicitly, averages the 31-day forcing, and
then performs one SOR inversion per factor and member.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_era5_qd_peak_lag0_7_author_sor_17lev as qdgt  # noqa: E402
import compute_era5_qe_fe_geopotential_tendency as qefegt  # noqa: E402
import compute_era5_qe_fe_peak_lag0_7 as qefe  # noqa: E402
import compute_plot_hu2025_lag0_20_5day as controller  # noqa: E402
import plot_hu2025_lag3_qd_qe_fe_comparison as plotbase  # noqa: E402


DATA_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推")
OLD_QEFE_ROOT = DATA_ROOT / "era5_qe_fe_geopotential_tendency_lagm2_27"
OLD_FORCING_ROOT = OLD_QEFE_ROOT / "forcing_12lev"
OLD_EVENT_FILE = OLD_FORCING_ROOT / "ERA5_Qe_Fe_peak_lag-2_27_event_anomalies_12lev.nc"
OLD_VAR_CLIM_FILE = OLD_FORCING_ROOT / "ERA5_variable_daily_climatology_OctFeb_12lev_1981_2020.nc"
OLD_QEFE_CLIM_FILE = OLD_FORCING_ROOT / "ERA5_Qe_Fe_daily_climatology_target_doys_12lev_1981_2020.nc"
OLD_SIGMA_FILE = (
    DATA_ROOT
    / "era5_qd_peak_lagm2_27_author_sor_12lev"
    / "ERA5_Qd_peak_lag-2_27_author_smoothvars_17lev_forcing.nc"
)

OUT_ROOT = DATA_ROOT / "era5_qe_fe_geopotential_tendency_lag0_30_corrected"
MISSING_CLIM_FILE = OUT_ROOT / "ERA5_Qe_Fe_daily_climatology_missing_doy21_23_12lev_1981_2020.nc"
MISSING_SIGMA_FILE = OUT_ROOT / "ERA5_sigma1_missing_doy21_23_12lev_1981_2020.nc"
FORCING_FILE = OUT_ROOT / "ERA5_Qe_Fe_lag0_30_forcing_mean_corrected.nc"
GT_FILE = OUT_ROOT / "ERA5_GT_by_Qe_Fe_lag0_30_forcing_mean_corrected_author_SOR.nc"

FIG_ROOT = DATA_ROOT / "hu2025_GTeddy_N13_vs_2004_lag0_30_mean_corrected_lon180_225E"
FIG_FILE = FIG_ROOT / "ERA5_Hu2025_GTeddy_N13_vs_2004_lag0_30_mean_corrected_lat_pressure.png"
CHECK_FILE = FIG_ROOT / "ERA5_Hu2025_GTeddy_N13_vs_2004_lag0_30_mean_corrected_check.csv"

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
LAGS = list(range(31))
NEW_LAGS = [28, 29, 30]
MEMBERS = ["N=13", "2004-12-24"]
MEMBER_INDICES = {
    "N=13": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13],
    "2004-12-24": [6],
}
LON_MIN, LON_MAX = 180.0, 225.0
LAT_MIN, LAT_MAX = 20.0, 80.0


def configure_modules() -> None:
    qefe.ERA5_FILLED_8LEV_ROOT = controller.PREP12_ROOT
    qefe.LEVELS = LEVELS
    qefe.LAGS = LAGS
    qefe.LAT_MIN, qefe.LAT_MAX = 10.0, 85.0
    qefe.LON_MIN, qefe.LON_MAX = 120.0, 260.0
    qefe.DISPLAY_LAT_MIN, qefe.DISPLAY_LAT_MAX = 10.0, 85.0
    qefe.DISPLAY_LON_MIN, qefe.DISPLAY_LON_MAX = 120.0, 260.0
    qefe.load_season = controller.load_season_with_2024_jan

    qdgt.ERA5_FILLED_17LEV_ROOT = controller.PREP12_ROOT
    qdgt.LEVELS = LEVELS
    qdgt.SOR_LEVELS = LEVELS
    qdgt.P_PA = np.asarray(LEVELS, dtype=np.float64) * 100.0

    qefegt.SOR_LEVELS = LEVELS


def required_missing_doys() -> list[int]:
    target_doys = set(int(x) for x in qefe.target_table().doy_nl.unique())
    with xr.open_dataset(OLD_QEFE_CLIM_FILE) as old:
        old_doys = set(int(x) for x in old.doy_nl.values)
    missing = sorted(target_doys - old_doys)
    if missing != [21, 22, 23]:
        raise ValueError(f"Unexpected missing Qe/Fe climatology DOYs: {missing}")
    return missing


def build_missing_qefe_climatology(missing_doys: list[int]) -> xr.Dataset:
    if MISSING_CLIM_FILE.exists():
        print(f"[CACHE] {MISSING_CLIM_FILE}", flush=True)
        return xr.open_dataset(MISSING_CLIM_FILE)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sums: dict[str, np.ndarray] = {}
    counts = np.zeros(len(missing_doys), dtype=np.int16)
    coords: dict[str, np.ndarray] | None = None
    with xr.open_dataset(OLD_VAR_CLIM_FILE) as variable_clim:
        for season_year in qefe.BASELINE_YEARS:
            print(f"[MISSING-QEFE-CLIM] season={season_year}", flush=True)
            season = qefe.load_season(season_year)
            try:
                anomalies = qefe.anomaly_dataset(season, variable_clim)
                qe, fe = qefe.compute_qe_fe(anomalies)
                qe = qe.sel(lat=slice(10.0, 85.0), lon=slice(120.0, 260.0))
                fe = fe.sel(lat=slice(10.0, 85.0), lon=slice(120.0, 260.0))
                if coords is None:
                    coords = {
                        "doy_nl": np.asarray(missing_doys, dtype=np.int16),
                        "lev": qe.lev.values,
                        "lat": qe.lat.values,
                        "lon": qe.lon.values,
                    }
                    shape = (len(missing_doys), qe.sizes["lev"], qe.sizes["lat"], qe.sizes["lon"])
                    sums = {"Qe": np.zeros(shape, dtype=np.float64), "Fe": np.zeros(shape, dtype=np.float64)}
                lookup = {qefe.calc_doy_nl(t): i for i, t in enumerate(pd.to_datetime(qe.time.values))}
                for di, doy in enumerate(missing_doys):
                    ti = lookup[doy]
                    sums["Qe"][di] += qe.isel(time=ti).values.astype(np.float64)
                    sums["Fe"][di] += fe.isel(time=ti).values.astype(np.float64)
                    counts[di] += 1
            finally:
                season.close()
                gc.collect()

    if coords is None or int(counts.min()) != len(qefe.BASELINE_YEARS):
        raise RuntimeError(f"Incomplete missing Qe/Fe climatology counts: {counts.tolist()}")
    out = xr.Dataset(
        {
            "Qe_climatology": (("doy_nl", "lev", "lat", "lon"), (sums["Qe"] / counts[:, None, None, None]).astype(np.float32)),
            "Fe_climatology": (("doy_nl", "lev", "lat", "lon"), (sums["Fe"] / counts[:, None, None, None]).astype(np.float32)),
            "count": ("doy_nl", counts),
        },
        coords=coords,
        attrs={"baseline_period": "1981-2020", "purpose": "Exact missing DOYs for corrected lag0-30 forcing"},
    )
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in ["Qe_climatology", "Fe_climatology"]}
    out.to_netcdf(MISSING_CLIM_FILE, encoding=encoding)
    out.close()
    print(f"[NC] {MISSING_CLIM_FILE}", flush=True)
    return xr.open_dataset(MISSING_CLIM_FILE)


def build_missing_sigma(missing_doys: list[int]) -> xr.Dataset:
    if MISSING_SIGMA_FILE.exists():
        print(f"[CACHE] {MISSING_SIGMA_FILE}", flush=True)
        return xr.open_dataset(MISSING_SIGMA_FILE)

    sigma_sum = np.zeros((len(missing_doys), len(LEVELS)), dtype=np.float64)
    counts = np.zeros(len(missing_doys), dtype=np.int16)
    for year in qdgt.BASELINE_YEARS:
        centers = [qdgt.date_from_doy_nl(year, doy) for doy in missing_doys]
        print(f"[MISSING-SIGMA] year={year}", flush=True)
        qd, sigma0, sigma1 = qdgt.compute_qd_for_centers(centers)
        try:
            for di, center in enumerate(centers):
                sigma_sum[di] += sigma1.sel(date=np.datetime64(center)).mean(("lat", "lon"), skipna=True).values
                counts[di] += 1
        finally:
            qd.close()
            sigma0.close()
            sigma1.close()
            gc.collect()
    if int(counts.min()) != len(qdgt.BASELINE_YEARS):
        raise RuntimeError(f"Incomplete missing sigma counts: {counts.tolist()}")
    out = xr.Dataset(
        {"sigma1_area_mean_doy": (("doy_nl", "lev"), (sigma_sum / counts[:, None]).astype(np.float32))},
        coords={"doy_nl": np.asarray(missing_doys, dtype=np.int16), "lev": np.asarray(LEVELS, dtype=np.int16)},
        attrs={"baseline_period": "1981-2020", "purpose": "Exact missing DOYs for corrected lag0-30 inversion"},
    )
    out.to_netcdf(MISSING_SIGMA_FILE)
    out.close()
    print(f"[NC] {MISSING_SIGMA_FILE}", flush=True)
    return xr.open_dataset(MISSING_SIGMA_FILE)


def combined_climatology(old: xr.Dataset, missing: xr.Dataset, variable: str) -> xr.DataArray:
    result = xr.concat([old[variable], missing[variable]], dim="doy_nl").sortby("doy_nl")
    if len(np.unique(result.doy_nl.values)) != result.sizes["doy_nl"]:
        raise ValueError(f"Duplicate DOYs in {variable}")
    return result.load()


def build_forcing_mean(missing_clim: xr.Dataset) -> xr.Dataset:
    if FORCING_FILE.exists():
        print(f"[CACHE] {FORCING_FILE}", flush=True)
        return xr.open_dataset(FORCING_FILE)

    target = qefe.target_table()
    with xr.open_dataset(OLD_QEFE_CLIM_FILE) as old_clim:
        clim = {
            "Qe": combined_climatology(old_clim, missing_clim, "Qe_climatology"),
            "Fe": combined_climatology(old_clim, missing_clim, "Fe_climatology"),
        }

    with xr.open_dataset(OLD_EVENT_FILE) as old_events:
        # Positions 0..27 contain actual physical lag0..27 despite the old
        # coordinate labels -2..25.
        event_sums = {
            "Qe": old_events["Qe_event_lag_anomaly"].isel(lag=slice(0, 28)).sum("lag").values.astype(np.float64),
            "Fe": old_events["Fe_event_lag_anomaly"].isel(lag=slice(0, 28)).sum("lag").values.astype(np.float64),
        }
        lev = old_events.lev.values
        lat = old_events.lat.values
        lon = old_events.lon.values

    new_target = target[target.lag.isin(NEW_LAGS)]
    for season_year, rows in sorted(new_target.groupby("season_year")):
        print(f"[EVENT-LAG28-30] season={season_year}", flush=True)
        season = qefe.load_season(int(season_year))
        try:
            with xr.open_dataset(OLD_VAR_CLIM_FILE) as variable_clim:
                anomalies = qefe.anomaly_dataset(season, variable_clim)
            qe, fe = qefe.compute_qe_fe(anomalies)
            fields = {
                "Qe": qe.sel(lat=slice(10.0, 85.0), lon=slice(120.0, 260.0)),
                "Fe": fe.sel(lat=slice(10.0, 85.0), lon=slice(120.0, 260.0)),
            }
            lookup = {pd.Timestamp(t).normalize(): i for i, t in enumerate(pd.to_datetime(qe.time.values))}
            for row in rows.itertuples(index=False):
                target_date = pd.Timestamp(row.target_date).normalize()
                ti = lookup[target_date]
                for factor in ["Qe", "Fe"]:
                    anomaly = fields[factor].isel(time=ti) - clim[factor].sel(doy_nl=int(row.doy_nl))
                    event_sums[factor][int(row.event)] += anomaly.values.astype(np.float64)
        finally:
            season.close()
            gc.collect()

    forcing = np.empty((2, 2, len(lev), len(lat), len(lon)), dtype=np.float32)
    for fi, factor in enumerate(["Qe", "Fe"]):
        for mi, member in enumerate(MEMBERS):
            indices = MEMBER_INDICES[member]
            forcing[fi, mi] = (event_sums[factor][indices].sum(axis=0) / (len(indices) * len(LAGS))).astype(np.float32)

    out = xr.Dataset(
        {"forcing_interval_mean": (("factor", "member", "lev", "lat", "lon"), forcing)},
        coords={"factor": ["Qe", "Fe"], "member": MEMBERS, "lev": lev, "lat": lat, "lon": lon},
        attrs={
            "lag_window": "0-30 inclusive",
            "n_days": 31,
            "aggregation": "forcing-first interval mean before SOR inversion",
            "old_cache_decode": "old array positions 0..27 interpreted as physical lag0..27",
            "N13_indices": ",".join(str(x) for x in MEMBER_INDICES["N=13"]),
        },
    )
    out["forcing_interval_mean"].attrs["units"] = "Qe: K s-1; Fe: s-2"
    encoding = {"forcing_interval_mean": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    out.to_netcdf(FORCING_FILE, encoding=encoding)
    out.close()
    print(f"[NC] {FORCING_FILE}", flush=True)
    return xr.open_dataset(FORCING_FILE)


def combined_sigma(old: xr.Dataset, missing: xr.Dataset) -> xr.DataArray:
    sigma = xr.concat([old["sigma1_area_mean_doy"], missing["sigma1_area_mean_doy"]], dim="doy_nl").sortby("doy_nl")
    _, unique_index = np.unique(sigma.doy_nl.values, return_index=True)
    return sigma.isel(doy_nl=np.sort(unique_index)).load()


def build_gt(forcing: xr.Dataset, missing_sigma: xr.Dataset) -> xr.Dataset:
    if GT_FILE.exists():
        print(f"[CACHE] {GT_FILE}", flush=True)
        return xr.open_dataset(GT_FILE)

    target = qefe.target_table()
    with xr.open_dataset(OLD_SIGMA_FILE) as old_sigma:
        sigma_all = combined_sigma(old_sigma, missing_sigma)

    phi = np.empty((2, 2, len(LEVELS), forcing.sizes["lat"], forcing.sizes["lon"]), dtype=np.float32)
    sigma_used = np.empty((2, len(LEVELS)), dtype=np.float32)
    for mi, member in enumerate(MEMBERS):
        rows = target[target.event.isin(MEMBER_INDICES[member])]
        doys = rows.doy_nl.astype(int).tolist()
        sigma = np.maximum(sigma_all.sel(doy_nl=doys, lev=LEVELS).mean("doy_nl").values.astype(np.float64), 1.0e-12)
        sigma_used[mi] = sigma.astype(np.float32)
        for fi, factor in enumerate(["Qe", "Fe"]):
            print(f"[SOR] factor={factor} member={member} lag0-30 mean", flush=True)
            field = forcing["forcing_interval_mean"].sel(factor=factor, member=member, lev=LEVELS).values.astype(np.float32)
            phi[fi, mi] = qefegt.invert_one(field, sigma, forcing.lat.values.astype(np.float64), factor)

    out = xr.Dataset(
        {
            "phi_tendency": (("factor", "member", "lev", "lat", "lon"), phi),
            "sigma1_interval_mean": (("member", "lev"), sigma_used),
        },
        coords={
            "factor": ["Qe", "Fe"],
            "member": MEMBERS,
            "lev": np.asarray(LEVELS, dtype=np.int16),
            "lat": forcing.lat.values,
            "lon": forcing.lon.values,
        },
        attrs={"lag_window": "0-30 inclusive", "n_days": 31, "method": "forcing-first mean then author SOR"},
    )
    out["phi_tendency"].attrs["units"] = "m2 s-3"
    encoding = {"phi_tendency": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    out.to_netcdf(GT_FILE, encoding=encoding)
    out.close()
    print(f"[NC] {GT_FILE}", flush=True)
    return xr.open_dataset(GT_FILE)


def format_lat(value: float) -> str:
    return f"{int(round(value))}°N"


def plot(gt: xr.Dataset) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.0, 3.25))
    grid = fig.add_gridspec(1, 2, left=0.070, right=0.982, bottom=0.175, top=0.900, wspace=0.145)
    rows: list[dict[str, object]] = []

    for col, member in enumerate(MEMBERS):
        panel_grid = grid[0, col].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.035)
        ax = fig.add_subplot(panel_grid[0, 0])
        cax = fig.add_subplot(panel_grid[0, 1])
        field = gt["phi_tendency"].sel(member=member).sum("factor")
        if float(field.lon.max()) <= 180.0:
            field = field.assign_coords(lon=field.lon % 360).sortby("lon")
        section = (
            field.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
            .mean("lon", skipna=True)
            .sortby("lat")
            .sortby("lev")
            * 1.0e4
        )
        values = section.transpose("lev", "lat").values
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError(f"GTeddy {member} contains NaN or Inf")
        vmax = plotbase.nice_vmax(float(np.nanmax(np.abs(values))))
        levels = np.linspace(-vmax, vmax, 17)
        contour = ax.contourf(section.lat, section.lev, values, levels=levels, cmap=plotbase.paper_cmap(), extend="neither")

        ax.axhline(500, color="0.35", linewidth=0.35, alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylim(1000, 100)
        ax.set_xlim(LAT_MIN, LAT_MAX)
        ax.set_title(f"({'ab'[col]})", loc="left", fontsize=9.4, pad=3)
        ax.set_title(r"GT$_{\mathrm{eddy}}$", loc="center", fontsize=9.6, pad=3)
        ax.set_title(member, loc="right", fontsize=9.3, pad=3)
        pressure_ticks = [1000, 850, 700, 500, 300, 200, 100]
        ax.yaxis.set_major_locator(FixedLocator(pressure_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([str(x) for x in pressure_ticks] if col == 0 else [""] * len(pressure_ticks)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        lat_ticks = [20, 40, 60, 80]
        ax.xaxis.set_major_locator(FixedLocator(lat_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([format_lat(x) for x in lat_ticks]))
        ax.tick_params(length=3, width=0.7, pad=1.5, labelsize=7.8)
        labels = ax.get_xticklabels()
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        if col == 0:
            ax.set_ylabel("hPa", fontsize=8.6, labelpad=2)

        ticks = [-vmax, -vmax / 2.0, 0.0, vmax / 2.0, vmax]
        colorbar = fig.colorbar(contour, cax=cax, orientation="vertical", ticks=ticks)
        colorbar.ax.tick_params(labelsize=7.4, length=2.5, pad=1.5)
        colorbar.outline.set_linewidth(0.7)
        rows.append(
            {
                "member": member,
                "lag_start": 0,
                "lag_end": 30,
                "n_days": 31,
                "aggregation": "forcing-first mean then SOR",
                "lon_min_E": LON_MIN,
                "lon_max_E": LON_MAX,
                "min_1e4_m2_s3": float(np.nanmin(values)),
                "max_1e4_m2_s3": float(np.nanmax(values)),
                "mean_1e4_m2_s3": float(np.nanmean(values)),
                "vmax_1e4_m2_s3": vmax,
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
            }
        )

    fig.text(0.525, 0.055, r"Geopotential tendency ($10^{-4}$ m$^2$ s$^{-3}$)", ha="center", va="center", fontsize=8.4)
    fig.savefig(FIG_FILE, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(CHECK_FILE, index=False, encoding="utf-8-sig")
    print(f"[FIG] {FIG_FILE}", flush=True)
    print(f"[CSV] {CHECK_FILE}", flush=True)


def main() -> None:
    for path in [OLD_EVENT_FILE, OLD_VAR_CLIM_FILE, OLD_QEFE_CLIM_FILE, OLD_SIGMA_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)
    configure_modules()
    missing_doys = required_missing_doys()
    missing_clim = build_missing_qefe_climatology(missing_doys)
    missing_sigma = build_missing_sigma(missing_doys)
    try:
        forcing = build_forcing_mean(missing_clim)
        try:
            gt = build_gt(forcing, missing_sigma)
            try:
                plot(gt)
            finally:
                gt.close()
        finally:
            forcing.close()
    finally:
        missing_clim.close()
        missing_sigma.close()


if __name__ == "__main__":
    main()
