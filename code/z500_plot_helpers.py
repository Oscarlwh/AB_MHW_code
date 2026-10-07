from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


INPUT_NC = Path(
    "/path/to/data/ABH_170E240E_1981_2024_5N90N/composite/"
    "composite_13_plus_2004_5d_peak_twocolumn/"
    "Z500_anomaly_5dmean_group13_vs_2004_peak_lead15_lag15_"
    "170E240E_20N75N_p005.nc"
)
OUTPUT_DIR = Path("/path/to/output")
OUTPUT_STEM = (
    "Z500_anomaly_5dmean_group13_vs_2004_peak_"
    "lead15_lag15_GRL_optimized"
)

LON_MIN, LON_MAX = 170.0, 240.0
LAT_MIN, LAT_MAX = 20.0, 75.0

# A shared range is retained so amplitudes remain directly comparable.
# Rectangular colorbar end segments disclose values outside the plotted range.
FILL_LIMIT = 240
FILL_LEVELS = np.arange(-FILL_LIMIT, FILL_LIMIT + 30, 30)
CONTOUR_LEVELS = np.arange(-480, 481, 40)
CONTOUR_LEVELS = CONTOUR_LEVELS[CONTOUR_LEVELS != 0]
CONTOUR_LABEL_LEVELS = np.arange(-400, 401, 80)
CONTOUR_LABEL_LEVELS = CONTOUR_LABEL_LEVELS[
    CONTOUR_LABEL_LEVELS != 0
]

LON_TICKS = [180, 200, 220, 240]
LON_LABELS = ["180°", "160°W", "140°W", "120°W"]
LAT_TICKS = [20, 30, 40, 50, 60, 70]
LAT_LABELS = ["20°N", "30°N", "40°N", "50°N", "60°N", "70°N"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "text.color": "#20242a",
            "axes.labelcolor": "#20242a",
            "xtick.color": "#20242a",
            "ytick.color": "#20242a",
            "axes.edgecolor": "#20242a",
            "axes.linewidth": 0.65,
            "contour.negative_linestyle": "dashed",
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_map_axis(
    ax: plt.Axes,
    *,
    show_x: bool,
    y_side: str,
) -> None:
    data_crs = ccrs.PlateCarree()
    ax.set_extent(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        crs=data_crs,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.62,
        edgecolor="#101318",
        zorder=7,
    )
    ax.set_xticks(LON_TICKS, crs=data_crs)
    ax.set_yticks(LAT_TICKS, crs=data_crs)
    ax.set_xticklabels(LON_LABELS if show_x else [])
    ax.set_yticklabels(LAT_LABELS)
    ax.tick_params(
        axis="y",
        labelleft=y_side == "left",
        labelright=y_side == "right",
        left=y_side == "left",
        right=y_side == "right",
    )
    if y_side == "right":
        ax.yaxis.set_ticks_position("right")
    else:
        ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        direction="out",
        length=2.6,
        width=0.55,
        pad=1.8,
    )
    ax.gridlines(
        crs=data_crs,
        xlocs=LON_TICKS,
        ylocs=LAT_TICKS,
        linewidth=0.28,
        color="#6f7782",
        alpha=0.26,
        linestyle=":",
        draw_labels=False,
        zorder=1,
    )


def add_field(
    ax: plt.Axes,
    *,
    data: xr.DataArray,
    significant: xr.DataArray | None,
    panel_label: str,
    time_label: str,
    show_x: bool,
    y_side: str,
    lons: np.ndarray,
    lats: np.ndarray,
) -> mpl.contour.QuadContourSet:
    data_crs = ccrs.PlateCarree()
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_under("#08306b")
    cmap.set_over("#67001f")

    cf = ax.contourf(
        lons,
        lats,
        data.values,
        levels=FILL_LEVELS,
        cmap=cmap,
        extend="both",
        transform=data_crs,
        zorder=0,
    )

    cs = ax.contour(
        lons,
        lats,
        data.values,
        levels=CONTOUR_LEVELS,
        colors="#30343b",
        linewidths=0.38,
        alpha=0.74,
        transform=data_crs,
        zorder=3,
    )
    available_label_levels = [
        level
        for level in CONTOUR_LABEL_LEVELS
        if np.any(np.isclose(cs.levels, level))
    ]
    ax.clabel(
        cs,
        levels=available_label_levels,
        inline=True,
        inline_spacing=2,
        fontsize=5.5,
        fmt="%d",
        colors="#30343b",
    )
    ax.contour(
        lons,
        lats,
        data.values,
        levels=[0],
        colors="#20242a",
        linewidths=0.78,
        transform=data_crs,
        zorder=4,
    )

    if significant is not None:
        stride = 3
        mask = significant.values[::stride, ::stride].astype(bool)
        lon_sub = lons[::stride]
        lat_sub = lats[::stride]
        xx, yy = np.meshgrid(lon_sub, lat_sub)
        ax.scatter(
            xx[mask],
            yy[mask],
            s=4.0,
            c="#111418",
            marker="o",
            alpha=0.82,
            edgecolors="white",
            linewidths=0.18,
            transform=data_crs,
            zorder=6,
        )

    format_map_axis(ax, show_x=show_x, y_side=y_side)
    ax.text(
        0.025,
        0.955,
        f"({panel_label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="semibold",
        color="black",
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        clip_on=True,
        zorder=10,
    )
    ax.text(
        0.975,
        0.045,
        time_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        fontweight="semibold",
        color="black",
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        clip_on=True,
        zorder=10,
    )
    return cf


def main() -> None:
    configure_style()
    ds = xr.open_dataset(INPUT_NC)
    group = ds["z500_anomaly_5dmean_group13"]
    case = ds["z500_anomaly_5dmean_special2004"]
    significance = ds["significant_p005"]
    lons = ds["lon"].values
    lats = ds["lat"].values

    projection = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(6.7, 14.9), facecolor="white")
    grid = fig.add_gridspec(
        7,
        2,
        left=0.078,
        right=0.845,
        bottom=0.030,
        top=0.988,
        wspace=0.018,
        hspace=0.025,
    )

    days = [-15, -10, -5, 0, 5, 10, 15]
    time_labels = {
        -15: "lead 15 days",
        -10: "lead 10 days",
        -5: "lead 5 days",
        0: "peak",
        5: "lag 5 days",
        10: "lag 10 days",
        15: "lag 15 days",
    }

    cf = None
    for row, day in enumerate(days):
        ax_comp = fig.add_subplot(grid[row, 0], projection=projection)
        ax_case = fig.add_subplot(grid[row, 1], projection=projection)
        cf = add_field(
            ax_comp,
            data=group.sel(relative_day=day),
            significant=significance.sel(relative_day=day),
            panel_label=chr(ord("a") + row),
            time_label=time_labels[day],
            show_x=row == len(days) - 1,
            y_side="left",
            lons=lons,
            lats=lats,
        )
        add_field(
            ax_case,
            data=case.sel(relative_day=day),
            significant=None,
            panel_label=chr(ord("h") + row),
            time_label=time_labels[day],
            show_x=row == len(days) - 1,
            y_side="right",
            lons=lons,
            lats=lats,
        )

    cbar_ax = fig.add_axes([0.920, 0.075, 0.026, 0.85])
    cbar = fig.colorbar(
        cf,
        cax=cbar_ax,
        orientation="vertical",
        ticks=[-240, -160, -80, 0, 80, 160, 240],
        extend="both",
        extendrect=True,
        extendfrac=0.035,
        spacing="uniform",
    )
    cbar.ax.tick_params(labelsize=7.6, length=2.4, width=0.55)
    cbar.outline.set_linewidth(0.6)

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=400, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
