#!/usr/bin/env python3
"""CAM5 fixed-SST idealized sensitivity diagnostics.

The archive contains 10 MHW-forced and 10 climatological-control atmospheric
realisations, each with 29 daily samples on 17 pressure levels.  The anonymous
first and second dimensions are interpreted as ensemble member and integration
day, respectively.  All reported responses are MHW minus CTL.

This script deliberately does not diagnose the imposed SST anomaly because the
archive contains only the MHW forcing file, not its control counterpart.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cam5_mpl")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from netCDF4 import Dataset
from scipy.stats import ttest_ind, ttest_rel


ROOT_DEFAULT = Path("/path/to/data/CAM5-低分辨率-SSTrun")
WINDOWS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 29)]
WINDOW_LABELS = ["Days 1–5", "Days 6–10", "Days 11–15",
                 "Days 16–20", "Days 21–25", "Days 26–29"]
SOURCE_BOX = (200.0, 225.0, 40.0, 50.0)
ALASKA_BOX = (180.0, 200.0, 50.0, 65.0)


FILES = {
    "T_mhw": "z/FAMIPC5_Nov_NEP_MHW_mjl_VerticalTransfer_T_daily.nc",
    "T_ctl": "z/FAMIPC5_clim_2002_VerticalTransfer_T_daily.nc",
    "W_mhw": "z/FAMIPC5_Nov_NEP_MHW_mjl_VerticalTransfer_W.nc",
    "W_ctl": "z/FAMIPC5_clim_2002_VerticalTransfer_W_daily.nc",
    "Z_mhw": "z/FAMIPC5_Nov_NEP_MHW_mjl_VerticalTransfer_Z3_daily.nc",
    "Z_ctl": "z/FAMIPC5_clim_2002_VerticalTransfer_Z3_daily.nc",
    "UV_mhw": "z/FAMIPC5_Nov_NEP_MHW_mjl_VerticalTransfer_UV_daily.nc",
    "UV_ctl": "z/FAMIPC5_clim_2002_VerticalTransfer_UV_3D_daily.nc",
}


COLORS = {
    "blue": "#277DA1",
    "orange": "#D97706",
    "dark": "#263238",
    "gray": "#6B7280",
    "light": "#E5E7EB",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": COLORS["dark"],
        "axes.labelcolor": COLORS["dark"],
        "xtick.color": COLORS["dark"],
        "ytick.color": COLORS["dark"],
        "text.color": COLORS["dark"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    })


def read_coords(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as ds:
        return (np.asarray(ds["lat"][:], dtype=float),
                np.asarray(ds["lon"][:], dtype=float),
                np.asarray(ds["lev_p"][:], dtype=float))


def read_field(path: Path, variable: str) -> np.ndarray:
    """Read one full classic-NetCDF variable once; float32 limits peak memory."""
    print(f"Reading {path.name}:{variable}", flush=True)
    with Dataset(path) as ds:
        var = ds[variable]
        var.set_auto_mask(False)
        arr = np.asarray(var[:], dtype=np.float32)
        fill = getattr(var, "_FillValue", None)
    if fill is not None:
        arr[arr > 1.0e30] = np.nan
    if arr.shape[:3] != (10, 29, 17):
        raise ValueError(f"Unexpected shape for {path}:{variable}: {arr.shape}")
    return arr


def area_mean(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
              box: tuple[float, float, float, float]) -> np.ndarray:
    lon0, lon1, lat0, lat1 = box
    yi = np.where((lat >= lat0) & (lat <= lat1))[0]
    xi = np.where((lon >= lon0) & (lon <= lon1))[0]
    sub = field[..., yi[:, None], xi]
    weights = np.cos(np.deg2rad(lat[yi]))[:, None]
    valid = np.isfinite(sub)
    num = np.nansum(sub * weights, axis=(-2, -1))
    den = np.sum(valid * weights, axis=(-2, -1))
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float32),
                     where=den > 0)


def window_means(field: np.ndarray, level_indices: list[int]) -> np.ndarray:
    """Return member × window × selected-level × lat × lon."""
    return np.stack(
        [np.nanmean(field[:, start:stop, level_indices], axis=1)
         for start, stop in WINDOWS], axis=1
    ).astype(np.float32)


def centered_five_day(a: np.ndarray) -> np.ndarray:
    """Centered 5-day mean along member × day × ... day axis."""
    out = np.empty_like(a, dtype=np.float32)
    nday = a.shape[1]
    for day in range(nday):
        start, stop = max(0, day - 2), min(nday, day + 3)
        out[:, day] = np.nanmean(a[:, start:stop], axis=1)
    return out


def welch(mhw: np.ndarray, ctl: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        stat, p = ttest_ind(mhw, ctl, axis=axis, equal_var=False, nan_policy="omit")
    return np.asarray(stat), np.asarray(p)


def paired_t(mhw: np.ndarray, ctl: np.ndarray,
             axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided paired t-test for index-matched MHW and CTL members."""
    if mhw.shape != ctl.shape:
        raise ValueError(
            f"Paired samples must have identical shapes: {mhw.shape} != {ctl.shape}"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        stat, p = ttest_rel(mhw, ctl, axis=axis, nan_policy="omit")
    return np.asarray(stat), np.asarray(p)


def pooled_se(mhw: np.ndarray, ctl: np.ndarray, axis: int = 0) -> np.ndarray:
    n1 = np.sum(np.isfinite(mhw), axis=axis)
    n0 = np.sum(np.isfinite(ctl), axis=axis)
    v1 = np.nanvar(mhw, axis=axis, ddof=1)
    v0 = np.nanvar(ctl, axis=axis, ddof=1)
    return np.sqrt(v1 / n1 + v0 / n0)


def extract(root: Path, cache_path: Path) -> dict[str, np.ndarray]:
    coord_path = root / FILES["T_mhw"]
    lat, lon, lev = read_coords(coord_path)
    selected_lev_values = [850, 700, 600, 500, 400, 300, 250]
    selected_idx = [int(np.where(lev == value)[0][0]) for value in selected_lev_values]
    out: dict[str, np.ndarray] = {"lat": lat, "lon": lon, "lev": lev,
                                 "selected_lev": np.asarray(selected_lev_values)}

    for exp in ("mhw", "ctl"):
        field = read_field(root / FILES[f"T_{exp}"], "T")
        out[f"T_source_{exp}"] = area_mean(field, lat, lon, SOURCE_BOX)
        out[f"T_alaska_{exp}"] = area_mean(field, lat, lon, ALASKA_BOX)
        out[f"T_windows_{exp}"] = window_means(field, selected_idx)
        del field

    for exp in ("mhw", "ctl"):
        field = read_field(root / FILES[f"W_{exp}"], "OMEGA")
        out[f"W_source_{exp}"] = area_mean(field, lat, lon, SOURCE_BOX)
        out[f"W_alaska_{exp}"] = area_mean(field, lat, lon, ALASKA_BOX)
        del field

    for exp in ("mhw", "ctl"):
        field = read_field(root / FILES[f"Z_{exp}"], "Z3")
        out[f"Z_source_{exp}"] = area_mean(field, lat, lon, SOURCE_BOX)
        out[f"Z_alaska_{exp}"] = area_mean(field, lat, lon, ALASKA_BOX)
        out[f"Z_windows_{exp}"] = window_means(field, selected_idx)
        del field

    idx250 = int(np.where(lev == 250)[0][0])
    for exp in ("mhw", "ctl"):
        for variable in ("U", "V"):
            field = read_field(root / FILES[f"UV_{exp}"], variable)
            out[f"{variable}250_windows_{exp}"] = np.stack(
                [np.nanmean(field[:, start:stop, idx250], axis=1)
                 for start, stop in WINDOWS], axis=1
            ).astype(np.float32)
            del field

    np.savez_compressed(cache_path, **out)
    return out


def load_cache(path: Path) -> dict[str, np.ndarray]:
    print(f"Loading derived cache {path}", flush=True)
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def draw_stipple(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                 p: np.ndarray, stride_x: int = 1, stride_y: int = 1,
                 transform=None, size: float = 3.2,
                 alpha: float = 0.45) -> None:
    mask = p[::stride_y, ::stride_x] < 0.05
    xx, yy = np.meshgrid(x[::stride_x], y[::stride_y])
    kwargs = {"transform": transform} if transform is not None else {}
    ax.scatter(xx[mask], yy[mask], s=size, c=COLORS["dark"], alpha=alpha,
               linewidths=0, **kwargs)


def pressure_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(1000, 100)
    ticks = [1000, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FixedFormatter([str(value) for value in ticks])
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylabel("Pressure (hPa)")
    ax.grid(True, axis="x", color=COLORS["light"], lw=0.6)


def plot_time_pressure(d: dict[str, np.ndarray], out: Path) -> None:
    days = np.arange(1, 30)
    lev = d["lev"]
    keep = lev >= 100
    fig, axes = plt.subplots(2, 1, figsize=(10.7, 8.0), sharex=True,
                             constrained_layout=True)
    cmap = mpl.colormaps["RdBu_r"]
    temp_levels = np.arange(-6, 6.01, 0.5)

    t1 = centered_five_day(d["T_source_mhw"])
    t0 = centered_five_day(d["T_source_ctl"])
    w1 = centered_five_day(d["W_source_mhw"])
    w0 = centered_five_day(d["W_source_ctl"])
    dt = np.nanmean(t1, axis=0) - np.nanmean(t0, axis=0)
    dw = np.nanmean(w1, axis=0) - np.nanmean(w0, axis=0)
    _, p = welch(t1, t0)
    cf = axes[0].contourf(days, lev[keep], dt[:, keep].T,
                          levels=temp_levels, cmap=cmap, extend="both")
    omega_levels = [-0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08]
    cs = axes[0].contour(days, lev[keep], dw[:, keep].T, levels=omega_levels,
                         colors=COLORS["dark"], linewidths=0.8)
    axes[0].clabel(cs, fmt="%.02f", fontsize=7)
    draw_stipple(axes[0], days, lev[keep], p[:, keep].T, stride_x=1,
                 size=10.0, alpha=0.78)
    axes[0].set_title("(a) MHW source region: ΔT shading and Δω contours",
                      loc="left", fontweight="bold")
    axes[0].text(1.0, 1.02, "40°–50°N, 160°–135°W",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold", color=COLORS["dark"])
    pressure_axis(axes[0])

    t1 = centered_five_day(d["T_alaska_mhw"])
    t0 = centered_five_day(d["T_alaska_ctl"])
    z1 = centered_five_day(d["Z_alaska_mhw"])
    z0 = centered_five_day(d["Z_alaska_ctl"])
    dt = np.nanmean(t1, axis=0) - np.nanmean(t0, axis=0)
    dz = np.nanmean(z1, axis=0) - np.nanmean(z0, axis=0)
    _, p = welch(t1, t0)
    axes[1].contourf(days, lev[keep], dt[:, keep].T,
                     levels=temp_levels, cmap=cmap, extend="both")
    z_levels = [x for x in np.arange(-280, 281, 40) if x != 0]
    cs = axes[1].contour(days, lev[keep], dz[:, keep].T, levels=z_levels,
                         colors=COLORS["dark"], linewidths=0.85)
    axes[1].clabel(cs, fmt="%d", fontsize=7)
    draw_stipple(axes[1], days, lev[keep], p[:, keep].T, stride_x=1,
                 size=10.0, alpha=0.78)
    axes[1].set_title("(b) Alaska region: ΔT shading and ΔZ3 contours",
                      loc="left", fontweight="bold")
    axes[1].text(1.0, 1.02, "50°–65°N, 180°–160°W",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold", color=COLORS["dark"])
    pressure_axis(axes[1])
    axes[1].set_xlabel("Integration day")
    axes[1].set_xticks([1, 5, 10, 15, 20, 25, 29])
    cbar = fig.colorbar(cf, ax=axes, location="bottom", shrink=0.72, pad=0.04,
                        aspect=35)
    cbar.set_label("Temperature response, MHW − CTL (K)")
    cbar.ax.set_title("Contours: (a) Δω (Pa s⁻¹); (b) ΔZ3 (m)",
                      fontsize=9.5, color="black", pad=9)
    fig.suptitle("CAM5 fixed-SST sensitivity: vertical atmospheric adjustment",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(out)
    plt.close(fig)


def add_region_boxes(ax, transform) -> None:
    for box, color, label in ((SOURCE_BOX, COLORS["orange"], "MHW source"),
                              (ALASKA_BOX, COLORS["blue"], "Alaska")):
        lon0, lon1, lat0, lat1 = box
        ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                               fill=False, ec=color, lw=1.5, transform=transform,
                               zorder=8, label=label))


def read_z500_member_period_means(
        path: Path, first_day: int = 15, last_day: int = 29
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read exact member-mean Z500 maps for an inclusive model-day period."""
    if first_day < 1 or last_day < first_day:
        raise ValueError(f"Invalid model-day range: {first_day}–{last_day}")
    print(f"Reading {path.name}:Z3 at 500 hPa, days {first_day}–{last_day}",
          flush=True)
    with Dataset(path) as ds:
        lat = np.asarray(ds["lat"][:], dtype=float)
        lon = np.asarray(ds["lon"][:], dtype=float)
        lev = np.asarray(ds["lev_p"][:], dtype=float)
        level_matches = np.where(lev == 500)[0]
        if level_matches.size != 1:
            raise ValueError(f"Expected one 500-hPa level in {path}; got {lev}")
        var = ds["Z3"]
        if var.shape[:3] != (10, 29, 17):
            raise ValueError(f"Unexpected Z3 shape for {path}: {var.shape}")
        if last_day > var.shape[1]:
            raise ValueError(
                f"Requested day {last_day}, but {path} has only {var.shape[1]} days"
            )
        var.set_auto_mask(False)
        daily = np.asarray(
            var[:, first_day - 1:last_day, int(level_matches[0]), :, :],
            dtype=np.float32,
        )
        fill = getattr(var, "_FillValue", None)
    if fill is not None:
        daily[daily > 1.0e30] = np.nan
    expected_days = last_day - first_day + 1
    if daily.shape != (10, expected_days, lat.size, lon.size):
        raise ValueError(f"Unexpected period subset shape for {path}: {daily.shape}")
    return np.nanmean(daily, axis=1).astype(np.float32), lat, lon


def plot_z500_days15_29(root: Path, out: Path, diagnostics_out: Path,
                        cache_path: Path | None = None,
                        show_region_boxes: bool = True) -> dict[str, object]:
    """Plot the model-day 15–29 mean Z500 response (MHW minus CTL)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from matplotlib.lines import Line2D

    first_day, last_day = 15, 29
    mhw, lat_mhw, lon_mhw = read_z500_member_period_means(
        root / FILES["Z_mhw"], first_day, last_day
    )
    ctl, lat_ctl, lon_ctl = read_z500_member_period_means(
        root / FILES["Z_ctl"], first_day, last_day
    )
    if not (np.array_equal(lat_mhw, lat_ctl) and np.array_equal(lon_mhw, lon_ctl)):
        raise ValueError("MHW and CTL Z500 coordinates do not match")
    lat, lon = lat_mhw, lon_mhw

    response = np.nanmean(mhw, axis=0) - np.nanmean(ctl, axis=0)
    _, p = paired_t(mhw, ctl)
    domain = ((lat[:, None] >= 20.0) & (lat[:, None] <= 80.0) &
              (lon[None, :] >= 150.0) & (lon[None, :] <= 260.0))
    p98 = float(np.nanpercentile(np.abs(response[domain]), 98))
    vmax = float(max(80.0, np.ceil(p98 / 20.0) * 20.0))
    fill_levels = np.arange(-vmax, vmax + 0.1, 20.0)
    contour_interval = float(max(20.0, np.ceil((vmax / 5.0) / 20.0) * 20.0))
    contour_limit = np.floor(vmax / contour_interval) * contour_interval
    negative_levels = np.arange(-contour_limit, -0.1, contour_interval)
    positive_levels = np.arange(contour_interval, contour_limit + 0.1,
                                contour_interval)

    proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        1, 1, figsize=(12.4, 6.5), subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    cf = ax.contourf(
        lon, lat, np.clip(response, -vmax, vmax), levels=fill_levels,
        cmap="RdBu_r", extend="neither",
        transform=data_crs,
    )
    contour_sets = []
    if negative_levels.size:
        contour_sets.append(ax.contour(
            lon, lat, response, levels=negative_levels, colors="black",
            linewidths=0.85, linestyles="--", transform=data_crs,
        ))
    if positive_levels.size:
        contour_sets.append(ax.contour(
            lon, lat, response, levels=positive_levels, colors="black",
            linewidths=0.85, linestyles="-", transform=data_crs,
        ))
    ax.contour(
        lon, lat, response, levels=[0.0], colors="black", linewidths=1.35,
        linestyles="-", transform=data_crs,
    )
    for cs in contour_sets:
        ax.clabel(cs, fmt="%d", fontsize=8, inline=True, inline_spacing=3)

    draw_stipple(
        ax, lon, lat, p, stride_x=2, stride_y=2, transform=data_crs,
        size=9.0, alpha=0.80,
    )
    ax.set_extent([150, 260, 20, 80], crs=data_crs)
    ax.coastlines(resolution="110m", lw=0.75, color=COLORS["dark"])
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.4,
                   edgecolor=COLORS["dark"])
    gl = ax.gridlines(
        draw_labels=False, linewidth=0.45, color=COLORS["gray"], alpha=0.55,
        linestyle=":", xlocs=[150, 180, -150, -120, -100],
        ylocs=[20, 30, 40, 50, 60, 70, 80],
    )
    ax.set_xticks([150, 180, 210, 240, 260], crs=data_crs)
    ax.set_yticks([20, 30, 40, 50, 60, 70, 80], crs=data_crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter(
        zero_direction_label=False, dateline_direction_label=False
    ))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", which="major", labelsize=10, pad=5,
                   colors=COLORS["dark"])
    if show_region_boxes:
        add_region_boxes(ax, data_crs)
    ax.set_title("CAM5 500hPa response", loc="left",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_title("EXP_NEP_SST − CTL", loc="right",
                 fontsize=12, fontweight="bold", pad=12)
    if show_region_boxes:
        legend_handles = [
            Rectangle((0, 0), 1, 1, fill=False, ec=COLORS["orange"], lw=1.8,
                      label="MHW source: 40°–50°N, 160°–135°W"),
            Rectangle((0, 0), 1, 1, fill=False, ec=COLORS["blue"], lw=1.8,
                      label="Alaska: 50°–65°N, 180°–160°W"),
            Line2D([], [], marker="o", linestyle="None", color=COLORS["dark"],
                   markersize=4.2, label="Paired t-test p < 0.05"),
        ]
        ax.legend(handles=legend_handles, loc="lower left", frameon=True,
                  framealpha=0.92, fontsize=9)
    cbar = fig.colorbar(cf, ax=ax, location="bottom", shrink=0.72, pad=0.025,
                        aspect=55)
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))
    cbar.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
    fig.savefig(out, dpi=300)
    plt.close(fig)

    alaska_mhw = area_mean(mhw, lat, lon, ALASKA_BOX)
    alaska_ctl = area_mean(ctl, lat, lon, ALASKA_BOX)
    _, alaska_p = paired_t(alaska_mhw, alaska_ctl)
    alaska_response = float(np.nanmean(alaska_mhw) - np.nanmean(alaska_ctl))
    cache_reference = None
    cache_difference = None
    if cache_path is not None and cache_path.exists():
        cached = load_cache(cache_path)
        lev_index = int(np.where(cached["lev"] == 500)[0][0])
        ref_mhw = np.nanmean(cached["Z_alaska_mhw"][:, 14:29, lev_index], axis=1)
        ref_ctl = np.nanmean(cached["Z_alaska_ctl"][:, 14:29, lev_index], axis=1)
        cache_reference = float(np.nanmean(ref_mhw) - np.nanmean(ref_ctl))
        cache_difference = float(alaska_response - cache_reference)

    diagnostics: dict[str, object] = {
        "field": "Z3 at 500 hPa",
        "response_definition": "ensemble mean EXP_NEP_SST minus ensemble mean CTL",
        "units": "gpm (numerically equivalent to the archived Z3 metres)",
        "model_days_inclusive": [first_day, last_day],
        "day_count": last_day - first_day + 1,
        "member_count_per_experiment": int(mhw.shape[0]),
        "significance_test": (
            "two-sided paired t-test across index-matched MHW and CTL members"
        ),
        "region_boxes_and_legend_shown": show_region_boxes,
        "map_extent": {"longitude_degE": [150.0, 260.0],
                       "latitude_degN": [20.0, 80.0]},
        "source_box": {"longitude_degE": [200.0, 225.0],
                       "latitude_degN": [40.0, 50.0]},
        "alaska_box": {"longitude_degE": [180.0, 200.0],
                       "latitude_degN": [50.0, 65.0]},
        "domain_abs_response_p98_gpm": p98,
        "symmetric_color_limit_gpm": vmax,
        "filled_contour_interval_gpm": 20.0,
        "line_contour_interval_gpm": contour_interval,
        "domain_response_min_gpm": float(np.nanmin(response[domain])),
        "domain_response_max_gpm": float(np.nanmax(response[domain])),
        "domain_significant_fraction_p_lt_0.05": float(np.mean(p[domain] < 0.05)),
        "alaska_area_mean_response_gpm": alaska_response,
        "alaska_area_mean_paired_t_p": float(alaska_p),
        "cached_vertical_diagnostic_alaska_response_gpm": cache_reference,
        "map_minus_cached_alaska_response_gpm": cache_difference,
        "interpretation_boundary": (
            "Positive values mean higher Z500 in EXP_NEP_SST than CTL; they do not by "
            "themselves identify an absolute high-pressure center. Model days "
            "are integration days under fixed SST forcing, not days relative "
            "to an observed MHW peak."
        ),
    }
    diagnostics_out.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return diagnostics


def plot_maps(d: dict[str, np.ndarray], out: Path) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lat, lon = d["lat"], d["lon"]
    ilev = int(np.where(d["selected_lev"] == 500)[0][0])
    z1, z0 = d["Z_windows_mhw"][:, :, ilev], d["Z_windows_ctl"][:, :, ilev]
    u1, u0 = d["U250_windows_mhw"], d["U250_windows_ctl"]
    v1, v0 = d["V250_windows_mhw"], d["V250_windows_ctl"]
    dz = np.nanmean(z1, axis=0) - np.nanmean(z0, axis=0)
    du = np.nanmean(u1, axis=0) - np.nanmean(u0, axis=0)
    dv = np.nanmean(v1, axis=0) - np.nanmean(v0, axis=0)
    _, pz = welch(z1, z0)

    proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.1), subplot_kw={"projection": proj},
                             constrained_layout=True)
    levels = np.arange(-260, 261, 20)
    cf = None
    for i, ax in enumerate(axes.flat):
        ax.set_extent([150, 260, 20, 80], crs=data_crs)
        cf = ax.contourf(lon, lat, dz[i], levels=levels, cmap="RdBu_r",
                         extend="both", transform=data_crs)
        draw_stipple(ax, lon, lat, pz[i], stride_x=3, stride_y=2,
                     transform=data_crs)
        step_y, step_x = 4, 4
        q = ax.quiver(lon[::step_x], lat[::step_y],
                      du[i, ::step_y, ::step_x], dv[i, ::step_y, ::step_x],
                      transform=data_crs, color=COLORS["dark"], width=0.0022,
                      headwidth=3.3, scale=220, regrid_shape=None)
        ax.coastlines(resolution="110m", lw=0.65, color=COLORS["dark"])
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.3,
                       edgecolor=COLORS["gray"])
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, color=COLORS["gray"],
                          alpha=0.45, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        if i % 3 != 0:
            gl.left_labels = False
        if i < 3:
            gl.bottom_labels = False
        add_region_boxes(ax, data_crs)
        ax.set_title(f"({chr(97+i)}) {WINDOW_LABELS[i]}", loc="left",
                     fontweight="bold")
        if i == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=7.5)
        if i == 2:
            ax.quiverkey(q, 0.88, 1.08, 10, "10 m s⁻¹", labelpos="E",
                         coordinates="axes", fontproperties={"size": 8})
    cbar = fig.colorbar(cf, ax=axes, location="bottom", shrink=0.72, pad=0.02,
                        aspect=42)
    cbar.set_label("500-hPa geopotential-height response, MHW − CTL (m)")
    fig.suptitle("Evolution of the CAM5 circulation response",
                 fontsize=15, fontweight="bold", y=1.035)
    fig.text(0.5, 1.005,
             "Shading: ΔZ500 | vectors: Δ(U,V)250 | dots: Welch p < 0.05 for Z500 | n = 10 per experiment",
             ha="center", va="top", color=COLORS["gray"], fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def plot_series(d: dict[str, np.ndarray], out: Path) -> None:
    days = np.arange(1, 30)
    lev = d["lev"]
    idx = {int(v): int(np.where(lev == v)[0][0]) for v in (850, 700, 500)}
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8.3), sharex=True,
                             constrained_layout=True)

    def response_with_se(mhw, ctl):
        return np.nanmean(mhw, axis=0) - np.nanmean(ctl, axis=0), pooled_se(mhw, ctl)

    for pressure, color in ((850, COLORS["orange"]), (700, COLORS["blue"])):
        mhw = centered_five_day(d["T_source_mhw"][:, :, idx[pressure]])
        ctl = centered_five_day(d["T_source_ctl"][:, :, idx[pressure]])
        mean, se = response_with_se(mhw, ctl)
        axes[0].plot(days, mean, lw=2, color=color, label=f"T{pressure}")
        axes[0].fill_between(days, mean-se, mean+se, color=color, alpha=0.18)
    axes[0].set_ylabel("ΔT (K)")
    axes[0].set_title("(a) MHW source-region lower-tropospheric temperature",
                      loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2)

    mhw = centered_five_day(d["W_source_mhw"][:, :, idx[500]])
    ctl = centered_five_day(d["W_source_ctl"][:, :, idx[500]])
    mean, se = response_with_se(mhw, ctl)
    axes[1].plot(days, mean, lw=2, color=COLORS["dark"])
    axes[1].fill_between(days, mean-se, mean+se, color=COLORS["gray"], alpha=0.22)
    axes[1].axhspan(-0.12, 0, color=COLORS["blue"], alpha=0.05)
    axes[1].set_ylabel("Δω500 (Pa s⁻¹)")
    axes[1].set_title("(b) MHW source-region vertical motion (negative = ascent)",
                      loc="left", fontweight="bold")

    mhw = centered_five_day(d["Z_alaska_mhw"][:, :, idx[500]])
    ctl = centered_five_day(d["Z_alaska_ctl"][:, :, idx[500]])
    mean, se = response_with_se(mhw, ctl)
    axes[2].plot(days, mean, lw=2, color=COLORS["blue"])
    axes[2].fill_between(days, mean-se, mean+se, color=COLORS["blue"], alpha=0.18)
    axes[2].set_ylabel("ΔZ500 (m)")
    axes[2].set_title("(c) Alaska-region mid-tropospheric circulation response",
                      loc="left", fontweight="bold")
    axes[2].set_xlabel("Integration day")
    axes[2].set_xticks([1, 5, 10, 15, 20, 25, 29])

    for ax in axes:
        ax.axhline(0, color=COLORS["dark"], lw=0.8)
        ax.grid(True, color=COLORS["light"], lw=0.65)
        ax.axvline(10.5, color=COLORS["gray"], ls="--", lw=0.8)
        ax.axvline(15.5, color=COLORS["gray"], ls="--", lw=0.8)
    fig.suptitle("Regional adjustment reveals a two-stage response",
                 fontsize=15, fontweight="bold", y=1.035)
    fig.text(0.5, 1.005,
             "MHW − CTL, 10-member ensemble mean; shaded range is pooled standard error; centered 5-day mean",
             ha="center", va="top", color=COLORS["gray"], fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def eady_rate(t: np.ndarray, selected_lev: np.ndarray,
              target: int, lat: np.ndarray) -> tuple[np.ndarray, float]:
    """Eady growth rate (day^-1) from window-mean state.

    sigma = 0.31 g |dtheta/dy| / (N theta). Static stability follows the same
    pressure-coordinate formula and 1e-6 s-2 floor used by the project's ERA5
    Eady diagnostic.
    """
    levels = [int(v) for v in selected_lev]
    it = levels.index(target)
    kappa, gravity, rd = 287.0 / 1004.0, 9.80665, 287.0
    pressure_pa = selected_lev.astype(np.float64) * 100.0
    theta_all = t.astype(np.float64) * (
        1000.0 / selected_lev[None, None, :, None, None]
    ) ** kappa
    dtheta_dp = np.gradient(theta_all, pressure_pa, axis=2, edge_order=2)
    n2_raw = -(gravity * gravity) * pressure_pa[None, None, :, None, None] * dtheta_dp
    n2_raw /= rd * t.astype(np.float64) * theta_all
    invalid = (~np.isfinite(n2_raw)) | (n2_raw < 1.0e-6)
    n2 = n2_raw.copy()
    for ilev in range(n2.shape[2] - 1):
        replacement = n2[:, :, ilev + 1]
        replacement = np.where(np.isfinite(replacement) & (replacement >= 1.0e-6),
                               replacement, 1.0e-6)
        n2[:, :, ilev] = np.where(invalid[:, :, ilev], replacement,
                                  n2[:, :, ilev])
    n2[:, :, -1] = np.where(invalid[:, :, -1], 1.0e-6, n2[:, :, -1])
    theta_t = theta_all[:, :, it]
    nfreq = np.sqrt(n2[:, :, it])
    y = 6_371_000.0 * np.deg2rad(lat)
    dtheta_dy = np.gradient(theta_t, y, axis=-2, edge_order=2)
    rate = (0.31 * gravity * np.abs(dtheta_dy) /
            (nfreq * theta_t) * 86400.0).astype(np.float32)
    return rate, float(np.mean(invalid[:, :, it]))


def plot_eady(d: dict[str, np.ndarray], out: Path) -> dict[str, float]:
    import cartopy.crs as ccrs

    lat, lon = d["lat"], d["lon"]
    rates = {}
    for exp in ("mhw", "ctl"):
        rates[exp] = {}
        for pressure in (700, 400):
            rate, floor_fraction = eady_rate(
                d[f"T_windows_{exp}"], d["selected_lev"], pressure, lat
            )
            rates[exp][pressure] = rate
            rates[exp][f"floor_{pressure}"] = floor_fraction

    phase_windows = {"Days 1–10": [0, 1], "Days 16–25": [3, 4]}
    proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), subplot_kw={"projection": proj},
                             constrained_layout=True)
    levels = np.arange(-0.60, 0.601, 0.05)
    cf = None
    diagnostics: dict[str, float] = {}
    for exp in ("mhw", "ctl"):
        for pressure in (700, 400):
            diagnostics[f"n2_floor_fraction_{exp}_{pressure}"] = float(
                rates[exp][f"floor_{pressure}"]
            )
    for row, (phase, wins) in enumerate(phase_windows.items()):
        for col, pressure in enumerate((700, 400)):
            ax = axes[row, col]
            mhw = np.nanmean(rates["mhw"][pressure][:, wins], axis=1)
            ctl = np.nanmean(rates["ctl"][pressure][:, wins], axis=1)
            response = np.nanmean(mhw, axis=0) - np.nanmean(ctl, axis=0)
            _, p = welch(mhw, ctl)
            cf = ax.contourf(lon, lat, response, levels=levels, cmap="PuOr_r",
                             extend="both", transform=data_crs)
            draw_stipple(ax, lon, lat, p, stride_x=3, stride_y=2,
                         transform=data_crs)
            ax.set_extent([150, 260, 20, 80], crs=data_crs)
            ax.coastlines(resolution="110m", lw=0.65, color=COLORS["dark"])
            gl = ax.gridlines(draw_labels=True, linewidth=0.35,
                              color=COLORS["gray"], alpha=0.45, linestyle=":")
            gl.top_labels = gl.right_labels = False
            if col:
                gl.left_labels = False
            if row == 0:
                gl.bottom_labels = False
            add_region_boxes(ax, data_crs)
            panel = chr(97 + row * 2 + col)
            ax.set_title(f"({panel}) {phase}, {pressure} hPa", loc="left",
                         fontweight="bold")
            region = (lat >= 30) & (lat <= 70)
            diagnostics[f"eady_{pressure}_{phase}_p95_abs"] = float(
                np.nanpercentile(np.abs(response[region]), 95))
    cbar = fig.colorbar(cf, ax=axes, location="bottom", shrink=0.74, pad=0.02,
                        aspect=40)
    cbar.set_label("Eady growth-rate response, MHW − CTL (day⁻¹)")
    fig.suptitle("CAM5 baroclinicity response to the fixed SST perturbation",
                 fontsize=15, fontweight="bold", y=1.035)
    fig.text(0.5, 1.005,
             "σ = 0.31g|∂θ/∂y|/(Nθ); dots: Welch p < 0.05; boxes mark source and Alaska regions",
             ha="center", va="top", color=COLORS["gray"], fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return diagnostics


def write_summary(d: dict[str, np.ndarray], out_csv: Path, diagnostics: dict[str, float]) -> None:
    lev = d["lev"]
    idx = {int(v): int(np.where(lev == v)[0][0]) for v in (850, 700, 500, 250)}
    metrics = {
        "source_T850_K": (d["T_source_mhw"][:, :, idx[850]], d["T_source_ctl"][:, :, idx[850]]),
        "source_T700_K": (d["T_source_mhw"][:, :, idx[700]], d["T_source_ctl"][:, :, idx[700]]),
        "source_omega500_Pa_s-1": (d["W_source_mhw"][:, :, idx[500]], d["W_source_ctl"][:, :, idx[500]]),
        "alaska_T500_K": (d["T_alaska_mhw"][:, :, idx[500]], d["T_alaska_ctl"][:, :, idx[500]]),
        "alaska_Z500_m": (d["Z_alaska_mhw"][:, :, idx[500]], d["Z_alaska_ctl"][:, :, idx[500]]),
        "alaska_Z250_m": (d["Z_alaska_mhw"][:, :, idx[250]], d["Z_alaska_ctl"][:, :, idx[250]]),
    }
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "window", "response", "pooled_se", "welch_p"])
        writer.writeheader()
        for name, (mhw, ctl) in metrics.items():
            for label, (start, stop) in zip(WINDOW_LABELS, WINDOWS):
                a = np.nanmean(mhw[:, start:stop], axis=1)
                b = np.nanmean(ctl[:, start:stop], axis=1)
                _, p = welch(a, b)
                writer.writerow({
                    "metric": name,
                    "window": label,
                    "response": f"{np.nanmean(a)-np.nanmean(b):.6g}",
                    "pooled_se": f"{pooled_se(a, b):.6g}",
                    "welch_p": f"{float(p):.6g}",
                })
    diag_path = out_csv.with_name("eady_diagnostics.json")
    diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")


def forcing_qc(root: Path, out: Path) -> dict[str, object]:
    path = root / "FAMIPC5_inputdata_forcing_clim_NEP_MHW_daily_2002_Nov.nc"
    with Dataset(path) as ds:
        sst = np.asarray(ds["SST_cpl"][:], dtype=np.float32)
        prediddle = np.asarray(ds["SST_cpl_prediddle"][:], dtype=np.float32)
        lat = np.asarray(ds["lat"][:], dtype=float)
        lon = np.asarray(ds["lon"][:], dtype=float)
    yi = np.where((lat >= SOURCE_BOX[2]) & (lat <= SOURCE_BOX[3]))[0]
    xi = np.where((lon >= SOURCE_BOX[0]) & (lon <= SOURCE_BOX[1]))[0]
    w = np.cos(np.deg2rad(lat[yi]))[:, None]
    source = sst[:, yi[:, None], xi]
    source_mean = np.sum(source * w, axis=(-2, -1)) / np.sum(w) / len(xi)
    qc = {
        "forcing_records": int(sst.shape[0]),
        "global_max_abs_temporal_change_degC": float(np.nanmax(np.abs(np.diff(sst, axis=0)))),
        "max_abs_SST_vs_prediddle_degC": float(np.nanmax(np.abs(sst - prediddle))),
        "source_SST_mean_degC": float(np.nanmean(source_mean)),
        "source_SST_temporal_std_degC": float(np.nanstd(source_mean)),
        "control_SST_file_available": False,
        "interpretation": "stationary prescribed MHW SST field; exact anomaly unavailable without control SST",
    }
    out.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    return qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--only-z500-days15-29", action="store_true",
        help="Generate only the exact model-day 15–29 Z500 MHW-minus-CTL map",
    )
    parser.add_argument(
        "--z500-no-boxes", action="store_true",
        help="Omit the two region boxes and the lower-left legend from the Z500 map",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    data_dir = args.output / "data"
    figures.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    setup_style()

    cache = data_dir / "cam5_derived_cache.npz"
    if args.only_z500_days15_29:
        suffix = "_no_boxes" if args.z500_no_boxes else ""
        figure_path = figures / f"Fig_CAM5_Z500_MHW_minus_CTL_days15_29{suffix}.png"
        diagnostics_path = data_dir / (
            f"Fig_CAM5_Z500_MHW_minus_CTL_days15_29{suffix}_diagnostics.json"
        )
        diagnostics = plot_z500_days15_29(
            args.input,
            figure_path,
            diagnostics_path,
            cache,
            show_region_boxes=not args.z500_no_boxes,
        )
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False), flush=True)
        print(f"Finished. Output: {figure_path}")
        return

    qc = forcing_qc(args.input, data_dir / "forcing_qc.json")
    print(json.dumps(qc, ensure_ascii=False), flush=True)
    d = extract(args.input, cache) if args.rebuild_cache or not cache.exists() else load_cache(cache)
    plot_time_pressure(d, figures / "Fig1_CAM5_vertical_adjustment.png")
    plot_maps(d, figures / "Fig2_CAM5_Z500_UV250_evolution.png")
    plot_z500_days15_29(
        args.input,
        figures / "Fig_CAM5_Z500_MHW_minus_CTL_days15_29.png",
        data_dir / "Fig_CAM5_Z500_MHW_minus_CTL_days15_29_diagnostics.json",
        cache,
    )
    plot_series(d, figures / "Fig3_CAM5_regional_two_stage_response.png")
    diagnostics = plot_eady(d, figures / "Fig4_CAM5_Eady_response.png")
    write_summary(d, data_dir / "regional_window_statistics.csv", diagnostics)
    print(f"Finished. Outputs: {args.output}")


if __name__ == "__main__":
    main()
