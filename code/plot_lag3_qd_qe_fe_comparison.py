#!/usr/bin/env python3
"""Plot centered five-day GT sections for Qd, Qe, and Fe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import xarray as xr


DATA_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推")
QD_FILE = (
    DATA_ROOT
    / "era5_qd_peak_lagm2_27_author_sor_12lev"
    / "ERA5_Qd_peak_lag-2_27_phi_tendency_author_SOR_1000-100hPa_sigma1_SC1p0.nc"
)
QE_FILE = (
    DATA_ROOT
    / "era5_qe_fe_geopotential_tendency_lagm2_27"
    / "ERA5_GT_by_Qe_lagm2_27_direct_N13_plus_2004_author_SOR_1000-100hPa.nc"
)
FE_FILE = (
    DATA_ROOT
    / "era5_qe_fe_geopotential_tendency_lagm2_27"
    / "ERA5_GT_by_Fe_lagm2_27_direct_N13_plus_2004_author_SOR_1000-100hPa.nc"
)
FACTORS = ("Qd", "Qe", "Fe")
SOURCE_FILES = {"Qd": QD_FILE, "Qe": QE_FILE, "Fe": FE_FILE}
VARIABLES = {"Qd": "phi_tendency_qd", "Qe": "phi_tendency_member", "Fe": "phi_tendency_member"}
LON_MIN, LON_MAX = 180.0, 225.0
LAT_MIN, LAT_MAX = 20.0, 80.0
LEV_MIN, LEV_MAX = 100.0, 1000.0
MEMBERS = ("N=13", "2004-12-24")
PANEL_LETTERS = {
    (0, 0): "a",
    (1, 0): "b",
    (2, 0): "c",
    (0, 1): "d",
    (1, 1): "e",
    (2, 1): "f",
}


def paper_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "hu2025_like_blue_red",
        [
            "#08306b",
            "#2171b5",
            "#6baed6",
            "#c6dbef",
            "#f7fbff",
            "#fff7bc",
            "#fee391",
            "#fdae61",
            "#f46d43",
            "#b30000",
        ],
        N=256,
    )


def format_lat(value: float) -> str:
    return f"{int(round(value))}°N"


def nice_vmax(value: float) -> float:
    for candidate in [2, 4, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100]:
        if value <= candidate:
            return float(candidate)
    return float(np.ceil(value / 25.0) * 25.0)


def select_member(field: xr.DataArray, factor: str, member: str) -> xr.DataArray:
    if member == "2004-12-24":
        return field.sel(member="event06_2004-12-24")
    if factor == "Qd":
        return (
            12.0 * field.sel(member="group1_N12")
            + field.sel(member="event11_2020-11-13")
        ) / 13.0
    return field.sel(member="N13_plus_2020")


def load_section(factor: str, member: str, lags: list[int]) -> xr.DataArray:
    path = SOURCE_FILES[factor]
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as dataset:
        field = select_member(dataset[VARIABLES[factor]], factor, member).load()

    if float(field.lon.max()) <= 180.0:
        field = field.assign_coords(lon=field.lon % 360).sortby("lon")
    missing_lags = [lag for lag in lags if lag not in field.lag.values]
    if missing_lags:
        raise KeyError(f"{factor} {member} is missing lags {missing_lags}")

    field = field.sel(
        lag=lags,
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )
    field = field.where((field.lev >= LEV_MIN) & (field.lev <= LEV_MAX), drop=True)
    section = field.mean("lag", skipna=True).mean("lon", skipna=True) * 1.0e4
    return section.sortby("lat").sortby("lev")


def configure_axis(
    ax: plt.Axes,
    row: int,
    col: int,
    factor: str,
    member: str,
    panel_lag: int,
) -> None:
    ax.axhline(500, color="0.35", linewidth=0.35, alpha=0.35)
    ax.set_yscale("log")
    ax.set_ylim(1000, 100)
    ax.set_xlim(LAT_MIN, LAT_MAX)

    letter = PANEL_LETTERS[(row, col)]
    day_label = "day" if abs(panel_lag) <= 1 else "days"
    ax.set_title(f"({letter}) lag {panel_lag} {day_label}", loc="left", fontsize=9.6, pad=3)
    ax.set_title(f"GT by {factor}", loc="center", fontsize=9.6, pad=3)
    ax.set_title(member, loc="right", fontsize=9.3, pad=3)

    pressure_ticks = [1000, 850, 700, 500, 300, 200, 100]
    ax.yaxis.set_major_locator(FixedLocator(pressure_ticks))
    pressure_labels = [str(value) for value in pressure_ticks] if col == 0 else [""] * len(pressure_ticks)
    ax.yaxis.set_major_formatter(FixedFormatter(pressure_labels))
    ax.yaxis.set_minor_formatter(NullFormatter())

    lat_ticks = [20, 40, 60, 80]
    ax.xaxis.set_major_locator(FixedLocator(lat_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([format_lat(value) for value in lat_ticks]))
    ax.tick_params(length=3, width=0.7, pad=1.5, labelsize=7.8)
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
    if col == 0:
        ax.set_ylabel("hPa", fontsize=8.6, labelpad=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-lag", type=int, default=3)
    parser.add_argument("--qd-lag", type=int)
    return parser.parse_args()


def output_paths(qd_lag: int, qefe_lag: int) -> tuple[Path, Path]:
    if qd_lag == qefe_lag:
        lag_tag = f"lag{qefe_lag}"
    else:
        lag_tag = f"Qd_lag{qd_lag}_QeFe_lag{qefe_lag}"
    out_dir = DATA_ROOT / (
        "hu2025_fig4_style_own_region_N13_vs_2004_"
        f"{lag_tag}_center5day_lon180_225E"
    )
    out_file = out_dir / (
        "ERA5_Hu2025_GT_by_Qd_Qe_Fe_N13_vs_2004_"
        f"{lag_tag}_center5day_lat_pressure.png"
    )
    return out_dir, out_file


def main() -> None:
    args = parse_args()
    qefe_lag = args.panel_lag
    qd_lag = qefe_lag if args.qd_lag is None else args.qd_lag
    factor_lags = {"Qd": qd_lag, "Qe": qefe_lag, "Fe": qefe_lag}
    factor_windows = {
        factor: list(range(center_lag - 2, center_lag + 3))
        for factor, center_lag in factor_lags.items()
    }
    out_dir, out_file = output_paths(qd_lag, qefe_lag)
    sections = {
        (factor, member): load_section(factor, member, factor_windows[factor])
        for factor in FACTORS
        for member in MEMBERS
    }

    fig = plt.figure(figsize=(9.0, 8.25))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.070,
        right=0.982,
        bottom=0.070,
        top=0.965,
        hspace=0.255,
        wspace=0.145,
    )
    cmap = paper_cmap()

    for row, factor in enumerate(FACTORS):
        for col, member in enumerate(MEMBERS):
            panel_grid = grid[row, col].subgridspec(
                1,
                2,
                width_ratios=[1.0, 0.035],
                wspace=0.035,
            )
            ax = fig.add_subplot(panel_grid[0, 0])
            color_axis = fig.add_subplot(panel_grid[0, 1])
            section = sections[(factor, member)]
            values = section.transpose("lev", "lat").values
            if np.isnan(values).any() or np.isinf(values).any():
                raise ValueError(f"{factor} {member} contains NaN or Inf")
            vmax = nice_vmax(float(np.nanpercentile(np.abs(values), 98.0)))
            levels = np.linspace(-vmax, vmax, 17)
            contour = ax.contourf(
                section.lat.values.astype(float),
                section.lev.values.astype(float),
                values,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            configure_axis(ax, row, col, factor, member, factor_lags[factor])
            ticks = [-vmax, -vmax / 2.0, 0.0, vmax / 2.0, vmax]
            colorbar = fig.colorbar(contour, cax=color_axis, orientation="vertical", ticks=ticks)
            colorbar.ax.tick_params(labelsize=7.4, length=2.5, pad=1.5)
            colorbar.outline.set_linewidth(0.7)
            print(
                f"[CHECK] {factor} {member}: lags={factor_windows[factor]}, "
                f"min={np.nanmin(values):.4f}, max={np.nanmax(values):.4f}, "
                f"vmax={vmax:g}, nan=0, inf=0",
                flush=True,
            )

    fig.text(
        0.525,
        0.018,
        r"Geopotential tendency ($10^{-4}$ m$^2$ s$^{-3}$)",
        ha="center",
        va="center",
        fontsize=8.4,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[FIG] {out_file}", flush=True)


if __name__ == "__main__":
    main()
