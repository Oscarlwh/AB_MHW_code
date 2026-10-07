#!/usr/bin/env python3
"""Compute forcing-first lag0-30 GT by Qd and plot Qd/Qe/Fe sections."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, StrMethodFormatter
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_era5_qd_peak_lag0_7_author_sor_17lev as qdgt  # noqa: E402
import compute_era5_qe_fe_peak_lag0_7 as qefe  # noqa: E402
import figure_s05_eddy_geopotential_tendency as corrected  # noqa: E402
import plot_hu2025_lag3_qd_qe_fe_comparison as plotbase  # noqa: E402
import plot_era5_eady_lag2_center5day_two_panel as eadyplot  # noqa: E402


DATA_ROOT = corrected.DATA_ROOT
OLD_QD_FILE = corrected.OLD_SIGMA_FILE
OUT_ROOT = DATA_ROOT / "era5_qd_geopotential_tendency_lag0_30_corrected"
MISSING_QD_CLIM_FILE = OUT_ROOT / "ERA5_Qd_climatology_missing_doy21_23_12lev_1981_2020.nc"
QD_FORCING_FILE = OUT_ROOT / "ERA5_Qd_lag0_30_forcing_mean_corrected.nc"
QD_GT_FILE = OUT_ROOT / "ERA5_GT_by_Qd_lag0_30_forcing_mean_corrected_author_SOR.nc"
QEFE_GT_FILE = corrected.GT_FILE
MISSING_SIGMA_FILE = corrected.MISSING_SIGMA_FILE

FIG_ROOT = DATA_ROOT / "hu2025_GT_Qd_Qe_Fe_N13_vs_2004_lag0_30_mean_corrected_lon180_225E"
FIG_FILE = Path(
    "/path/to/user/Documents/DesktopOrganizer/Folders/Paper/论文/图片汇总/GT/"
    "GT_by_Qd_Qe_Fe_font_unified_2004label.png"
)
FIG_PDF = FIG_FILE.with_suffix(".pdf")
CHECK_FILE = FIG_FILE.with_name("GT_by_Qd_Qe_Fe_font_unified_check.csv")

LEVELS = corrected.LEVELS
LAGS = corrected.LAGS
NEW_LAGS = corrected.NEW_LAGS
MEMBERS = corrected.MEMBERS
MEMBER_INDICES = corrected.MEMBER_INDICES
DISPLAY_MEMBER_LABELS = {
    "N=13": "N=13",
    "2004-12-24": "2004",
}
FACTORS = ["Qd", "Qe", "Fe"]
GT_ROWS = {0: "Qd", 2: "Qe", 3: "Fe"}
EGR_DATA_FILE = eadyplot.DATA_FILE
EGR_LEVEL = 700
EGR_CENTER_LAG = 5
EGR_LAG_WINDOW = list(range(EGR_CENTER_LAG - 2, EGR_CENTER_LAG + 3))
LON_MIN, LON_MAX = corrected.LON_MIN, corrected.LON_MAX
LAT_MIN, LAT_MAX = corrected.LAT_MIN, corrected.LAT_MAX
PANEL_LETTERS = {
    (0, 0): "a",
    (1, 0): "b",
    (2, 0): "c",
    (3, 0): "d",
    (0, 1): "e",
    (1, 1): "f",
    (2, 1): "g",
    (3, 1): "h",
}


def build_missing_qd_climatology(missing_doys: list[int]) -> xr.Dataset:
    if MISSING_QD_CLIM_FILE.exists():
        print(f"[CACHE] {MISSING_QD_CLIM_FILE}", flush=True)
        return xr.open_dataset(MISSING_QD_CLIM_FILE)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    qd_sum: np.ndarray | None = None
    counts = np.zeros(len(missing_doys), dtype=np.int16)
    coords: dict[str, np.ndarray] | None = None
    for year in qdgt.BASELINE_YEARS:
        centers = [qdgt.date_from_doy_nl(year, doy) for doy in missing_doys]
        print(f"[MISSING-QD-CLIM] year={year}", flush=True)
        qd, sigma0, sigma1 = qdgt.compute_qd_for_centers(centers)
        try:
            if qd_sum is None:
                coords = {
                    "doy_nl": np.asarray(missing_doys, dtype=np.int16),
                    "lev": qd.lev.values,
                    "lat": qd.lat.values,
                    "lon": qd.lon.values,
                }
                qd_sum = np.zeros((len(missing_doys), qd.sizes["lev"], qd.sizes["lat"], qd.sizes["lon"]), dtype=np.float64)
            for di, center in enumerate(centers):
                qd_sum[di] += qd.sel(date=np.datetime64(center)).values.astype(np.float64)
                counts[di] += 1
        finally:
            qd.close()
            sigma0.close()
            sigma1.close()
            gc.collect()
    if qd_sum is None or coords is None or int(counts.min()) != len(qdgt.BASELINE_YEARS):
        raise RuntimeError(f"Incomplete missing Qd climatology counts: {counts.tolist()}")
    out = xr.Dataset(
        {
            "Qd_climatology_doy": (("doy_nl", "lev", "lat", "lon"), (qd_sum / counts[:, None, None, None]).astype(np.float32)),
            "count": ("doy_nl", counts),
        },
        coords=coords,
        attrs={"baseline_period": "1981-2020", "purpose": "Exact missing DOYs for corrected lag0-30 Qd forcing"},
    )
    out.to_netcdf(MISSING_QD_CLIM_FILE, encoding={"Qd_climatology_doy": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    out.close()
    print(f"[NC] {MISSING_QD_CLIM_FILE}", flush=True)
    return xr.open_dataset(MISSING_QD_CLIM_FILE)


def combined_qd_climatology(old: xr.Dataset, missing: xr.Dataset) -> xr.DataArray:
    clim = xr.concat([old["Qd_climatology_doy"], missing["Qd_climatology_doy"]], dim="doy_nl").sortby("doy_nl")
    _, unique_index = np.unique(clim.doy_nl.values, return_index=True)
    return clim.isel(doy_nl=np.sort(unique_index)).load()


def build_qd_forcing_mean(missing_clim: xr.Dataset) -> xr.Dataset:
    if QD_FORCING_FILE.exists():
        print(f"[CACHE] {QD_FORCING_FILE}", flush=True)
        return xr.open_dataset(QD_FORCING_FILE)

    target = qefe.target_table()
    with xr.open_dataset(OLD_QD_FILE) as old:
        event_sums = old["Qd_anomaly_event_lag"].sel(lag=list(range(28)), lev=LEVELS).sum("lag").values.astype(np.float64)
        qd_clim = combined_qd_climatology(old, missing_clim)
        lev = old.lev.values
        lat = old.lat.values
        lon = old.lon.values

    new_target = target[target.lag.isin(NEW_LAGS)].copy()
    centers = [pd.Timestamp(x) for x in sorted(pd.to_datetime(new_target.target_date).unique())]
    print(f"[EVENT-QD-LAG28-30] centers={len(centers)}", flush=True)
    qd_days, sigma0, sigma1 = qdgt.compute_qd_for_centers(centers)
    try:
        for row in new_target.itertuples(index=False):
            field = qd_days.sel(date=np.datetime64(pd.Timestamp(row.target_date))) - qd_clim.sel(doy_nl=int(row.doy_nl))
            event_sums[int(row.event)] += field.values.astype(np.float64)
    finally:
        qd_days.close()
        sigma0.close()
        sigma1.close()
        gc.collect()

    forcing = np.empty((len(MEMBERS), len(lev), len(lat), len(lon)), dtype=np.float32)
    for mi, member in enumerate(MEMBERS):
        indices = MEMBER_INDICES[member]
        forcing[mi] = (event_sums[indices].sum(axis=0) / (len(indices) * len(LAGS))).astype(np.float32)
    out = xr.Dataset(
        {"Qd_forcing_interval_mean": (("member", "lev", "lat", "lon"), forcing)},
        coords={"member": MEMBERS, "lev": lev, "lat": lat, "lon": lon},
        attrs={"lag_window": "0-30 inclusive", "n_days": 31, "aggregation": "forcing-first interval mean before SOR inversion"},
    )
    out["Qd_forcing_interval_mean"].attrs["units"] = "K s-1"
    out.to_netcdf(QD_FORCING_FILE, encoding={"Qd_forcing_interval_mean": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    out.close()
    print(f"[NC] {QD_FORCING_FILE}", flush=True)
    return xr.open_dataset(QD_FORCING_FILE)


def build_qd_gt(forcing: xr.Dataset, missing_sigma: xr.Dataset) -> xr.Dataset:
    if QD_GT_FILE.exists():
        print(f"[CACHE] {QD_GT_FILE}", flush=True)
        return xr.open_dataset(QD_GT_FILE)

    target = qefe.target_table()
    with xr.open_dataset(OLD_QD_FILE) as old:
        sigma_all = corrected.combined_sigma(old, missing_sigma)
    phi = np.empty((len(MEMBERS), len(LEVELS), forcing.sizes["lat"], forcing.sizes["lon"]), dtype=np.float32)
    sigma_used = np.empty((len(MEMBERS), len(LEVELS)), dtype=np.float32)
    for mi, member in enumerate(MEMBERS):
        rows = target[target.event.isin(MEMBER_INDICES[member])]
        sigma = np.maximum(
            sigma_all.sel(doy_nl=rows.doy_nl.astype(int).tolist(), lev=LEVELS).mean("doy_nl").values.astype(np.float64),
            1.0e-12,
        )
        sigma_used[mi] = sigma.astype(np.float32)
        q = forcing["Qd_forcing_interval_mean"].sel(member=member, lev=LEVELS).values.astype(np.float64)
        print(f"[SOR] factor=Qd member={member} lag0-30 mean", flush=True)
        gt = qdgt.Laplac_SOR(
            np.transpose(q, (2, 1, 0)),
            sigma,
            1.0,
            1.0e-12,
            np.pi / 180.0,
            np.pi / 180.0,
            np.asarray(LEVELS, dtype=np.float64) * 100.0,
            forcing.lat.values.astype(np.float64),
            1,
        )
        phi[mi] = np.transpose(gt, (2, 1, 0)).astype(np.float32)
    out = xr.Dataset(
        {
            "phi_tendency_qd": (("member", "lev", "lat", "lon"), phi),
            "sigma1_interval_mean": (("member", "lev"), sigma_used),
        },
        coords={"member": MEMBERS, "lev": LEVELS, "lat": forcing.lat.values, "lon": forcing.lon.values},
        attrs={"lag_window": "0-30 inclusive", "n_days": 31, "method": "forcing-first mean then author SOR"},
    )
    out["phi_tendency_qd"].attrs["units"] = "m2 s-3"
    out.to_netcdf(QD_GT_FILE, encoding={"phi_tendency_qd": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    out.close()
    print(f"[NC] {QD_GT_FILE}", flush=True)
    return xr.open_dataset(QD_GT_FILE)


def get_field(factor: str, member: str, qd_ds: xr.Dataset, qefe_ds: xr.Dataset) -> xr.DataArray:
    if factor == "Qd":
        return qd_ds["phi_tendency_qd"].sel(member=member)
    return qefe_ds["phi_tendency"].sel(factor=factor, member=member)


def plot_egr_row(
    fig: plt.Figure,
    grid,
    egr_ds: xr.Dataset,
    rows: list[dict[str, object]],
) -> None:
    if not eadyplot.HAS_CARTOPY:
        raise RuntimeError("Cartopy is required for the inserted EGR map row")

    n13, case_2004, p_value = eadyplot.member_fields(
        egr_ds,
        EGR_LEVEL,
        EGR_LAG_WINDOW,
    )
    sources = {"N=13": n13, "2004-12-24": case_2004}
    projection = eadyplot.ccrs.PlateCarree(central_longitude=180)
    data_crs = eadyplot.ccrs.PlateCarree()

    for col, member in enumerate(MEMBERS):
        panel_grid = grid[1, col].subgridspec(
            1,
            2,
            width_ratios=[1.0, 0.035],
            wspace=0.035,
        )
        ax = fig.add_subplot(panel_grid[0, 0], projection=projection)
        cax = fig.add_subplot(panel_grid[0, 1])
        field = sources[member].sel(
            lat=slice(
                eadyplot.base.DISPLAY_LAT_MIN,
                eadyplot.base.DISPLAY_LAT_MAX,
            ),
            lon=slice(
                eadyplot.base.DISPLAY_LON_MIN,
                eadyplot.base.DISPLAY_LON_MAX,
            ),
        )
        vmax = eadyplot.base.nice_vmax(
            float(np.nanpercentile(np.abs(field.values), 98.0))
        )
        levels = np.linspace(-vmax, vmax, 17)
        plot_values = np.clip(field.values, -vmax, vmax)
        contour = ax.contourf(
            field.lon.values,
            field.lat.values,
            plot_values,
            levels=levels,
            cmap=eadyplot.base.paper_cmap(),
            extend="neither",
            transform=data_crs,
        )
        ax.set_extent(
            [
                eadyplot.base.DISPLAY_LON_MIN,
                eadyplot.base.DISPLAY_LON_MAX,
                eadyplot.base.DISPLAY_LAT_MIN,
                eadyplot.base.DISPLAY_LAT_MAX,
            ],
            crs=data_crs,
        )
        # Fill the same GridSpec cell as the latitude-pressure panels so the
        # map row shares identical left/right/top/bottom edges.
        ax.set_aspect("auto")
        ax.coastlines(
            resolution="110m",
            linewidth=0.65,
            color="black",
            zorder=8,
        )
        ax.add_feature(
            eadyplot.cfeature.BORDERS.with_scale("110m"),
            linewidth=0.25,
            edgecolor="black",
            zorder=8,
        )
        eadyplot.base.add_box(ax, data_crs)

        lon_ticks = [150, 180, 210, 240]
        lat_ticks = [20, 40, 60, 80]
        ax.set_xticks(lon_ticks, crs=data_crs)
        ax.set_yticks(lat_ticks, crs=data_crs)
        ax.set_xticklabels(
            [eadyplot.base.lon_label(x) for x in lon_ticks],
            fontsize=9.5,
        )
        ax.set_yticklabels(
            [f"{y}\N{DEGREE SIGN}N" for y in lat_ticks]
            if col == 0
            else [""] * len(lat_ticks),
            fontsize=9.5,
        )
        xlabels = ax.get_xticklabels()
        xlabels[0].set_ha("left")
        xlabels[-1].set_ha("right")
        ax.tick_params(length=3.4, width=0.85, pad=1.8)

        letter = PANEL_LETTERS[(1, col)]
        ax.set_title(
            f"({letter})",
            loc="left",
            fontsize=11.3,
            fontweight="semibold",
            pad=3,
        )
        ax.set_title("EGR", loc="center", fontsize=10.8, fontweight="semibold", pad=3)
        ax.set_title(
            DISPLAY_MEMBER_LABELS[member],
            loc="right",
            fontsize=10.5,
            fontweight="semibold",
            pad=3,
        )

        significant_fraction = np.nan
        if col == 0:
            p_plot = p_value.sel(
                lat=slice(
                    eadyplot.base.DISPLAY_LAT_MIN,
                    eadyplot.base.DISPLAY_LAT_MAX,
                ),
                lon=slice(
                    eadyplot.base.DISPLAY_LON_MIN,
                    eadyplot.base.DISPLAY_LON_MAX,
                ),
            )
            significant = p_plot.values[::2, ::2] < 0.05
            yy, xx = np.meshgrid(
                field.lat.values[::2],
                field.lon.values[::2],
                indexing="ij",
            )
            ax.scatter(
                xx[significant],
                yy[significant],
                s=5.2,
                color="black",
                marker="o",
                linewidths=0,
                transform=data_crs,
                zorder=10,
            )
            significant_fraction = float((p_plot.values < 0.05).mean())

        ticks = [-vmax, -vmax / 2.0, 0.0, vmax / 2.0, vmax]
        colorbar = fig.colorbar(
            contour,
            cax=cax,
            orientation="vertical",
            ticks=ticks,
        )
        colorbar.ax.tick_params(labelsize=8.8, length=2.8, width=0.8, pad=1.7)
        colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        colorbar.outline.set_linewidth(0.7)
        rows.append(
            {
                "factor": "EGR_700hPa",
                "member": member,
                "center_lag": EGR_CENTER_LAG,
                "window_lags": ",".join(str(x) for x in EGR_LAG_WINDOW),
                "aggregation": "centered five-day mean",
                "min_1e-2_day-1": float(np.nanmin(field.values)),
                "max_1e-2_day-1": float(np.nanmax(field.values)),
                "mean_1e-2_day-1": float(np.nanmean(field.values)),
                "vmax_1e-2_day-1": vmax,
                "significant_fraction_p005": significant_fraction,
                "nan_count": int(np.isnan(field.values).sum()),
                "inf_count": int(np.isinf(field.values).sum()),
            }
        )


def plot(
    qd_ds: xr.Dataset,
    qefe_ds: xr.Dataset,
    egr_ds: xr.Dataset,
) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 10.4,
            "axes.labelsize": 11.4,
            "axes.linewidth": 0.90,
            "xtick.labelsize": 9.7,
            "ytick.labelsize": 9.7,
            "text.color": "#20242A",
            "axes.labelcolor": "#20242A",
            "xtick.color": "#20242A",
            "ytick.color": "#20242A",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(9.0, 10.95))
    grid = fig.add_gridspec(
        4,
        2,
        left=0.070,
        right=0.982,
        bottom=0.055,
        top=0.970,
        hspace=0.245,
        wspace=0.145,
    )
    rows: list[dict[str, object]] = []
    for row, factor in GT_ROWS.items():
        for col, member in enumerate(MEMBERS):
            panel_grid = grid[row, col].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.035)
            ax = fig.add_subplot(panel_grid[0, 0])
            cax = fig.add_subplot(panel_grid[0, 1])
            field = get_field(factor, member, qd_ds, qefe_ds)
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
                raise ValueError(f"{factor} {member} contains NaN or Inf")
            vmax = plotbase.nice_vmax(float(np.nanmax(np.abs(values))))
            levels = np.linspace(-vmax, vmax, 17)
            contour = ax.contourf(section.lat, section.lev, values, levels=levels, cmap=plotbase.paper_cmap(), extend="neither")
            ax.axhline(500, color="0.35", linewidth=0.35, alpha=0.35)
            ax.set_yscale("log")
            ax.set_ylim(1000, 100)
            ax.set_xlim(LAT_MIN, LAT_MAX)
            letter = PANEL_LETTERS[(row, col)]
            ax.set_title(f"({letter})", loc="left", fontsize=11.3, fontweight="semibold", pad=3)
            ax.set_title(f"GT by {factor}", loc="center", fontsize=10.8, fontweight="semibold", pad=3)
            ax.set_title(
                DISPLAY_MEMBER_LABELS[member],
                loc="right",
                fontsize=10.5,
                fontweight="semibold",
                pad=3,
            )
            pressure_ticks = [1000, 850, 700, 500, 300, 200, 100]
            ax.yaxis.set_major_locator(FixedLocator(pressure_ticks))
            ax.yaxis.set_major_formatter(FixedFormatter([str(x) for x in pressure_ticks] if col == 0 else [""] * len(pressure_ticks)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            lat_ticks = [20, 40, 60, 80]
            ax.xaxis.set_major_locator(FixedLocator(lat_ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([corrected.format_lat(x) for x in lat_ticks]))
            ax.tick_params(length=3.4, width=0.85, pad=1.8, labelsize=9.5)
            labels = ax.get_xticklabels()
            labels[0].set_ha("left")
            labels[-1].set_ha("right")
            if col == 0:
                ax.set_ylabel("hPa", fontsize=10.4, labelpad=3)
            ticks = [-vmax, -vmax / 2.0, 0.0, vmax / 2.0, vmax]
            colorbar = fig.colorbar(contour, cax=cax, orientation="vertical", ticks=ticks)
            colorbar.ax.tick_params(labelsize=8.8, length=2.8, width=0.8, pad=1.7)
            colorbar.outline.set_linewidth(0.7)
            rows.append(
                {
                    "factor": factor,
                    "member": member,
                    "lag_start": 0,
                    "lag_end": 30,
                    "n_days": 31,
                    "aggregation": "forcing-first mean then SOR",
                    "min_1e4_m2_s3": float(np.nanmin(values)),
                    "max_1e4_m2_s3": float(np.nanmax(values)),
                    "mean_1e4_m2_s3": float(np.nanmean(values)),
                    "vmax_1e4_m2_s3": vmax,
                    "nan_count": int(np.isnan(values).sum()),
                    "inf_count": int(np.isinf(values).sum()),
                }
            )
    plot_egr_row(fig, grid, egr_ds, rows)
    fig.text(
        0.525,
        0.012,
        r"Geopotential tendency ($10^{-4}$ m$^2$ s$^{-3}$)",
        ha="center",
        va="center",
        fontsize=11.4,
    )
    fig.savefig(FIG_FILE, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIG_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(CHECK_FILE, index=False, encoding="utf-8-sig")
    print(f"[FIG] {FIG_FILE}", flush=True)
    print(f"[CSV] {CHECK_FILE}", flush=True)


def main() -> None:
    for path in [
        OLD_QD_FILE,
        QEFE_GT_FILE,
        MISSING_SIGMA_FILE,
        EGR_DATA_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    corrected.configure_modules()
    missing_doys = corrected.required_missing_doys()
    missing_qd_clim = build_missing_qd_climatology(missing_doys)
    try:
        qd_forcing = build_qd_forcing_mean(missing_qd_clim)
        try:
            with xr.open_dataset(MISSING_SIGMA_FILE) as missing_sigma:
                qd_gt = build_qd_gt(qd_forcing, missing_sigma)
            try:
                with xr.open_dataset(QEFE_GT_FILE) as qefe_gt:
                    with xr.open_dataset(EGR_DATA_FILE) as egr_source:
                        egr_ds = egr_source.load()
                    try:
                        plot(qd_gt, qefe_gt, egr_ds)
                    finally:
                        egr_ds.close()
            finally:
                qd_gt.close()
        finally:
            qd_forcing.close()
    finally:
        missing_qd_clim.close()


if __name__ == "__main__":
    main()
