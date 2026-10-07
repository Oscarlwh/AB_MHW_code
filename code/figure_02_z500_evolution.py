from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import xarray as xr

import z500_plot_helpers as base


OUTPUT_DIR = Path("/path/to/output")
OUTPUT_STEM = (
    "Z500_anomaly_5dmean_group13_vs_2004_peak_"
    "lead15_lag15_GRL_2col7row_shortwide_highcontrast_labels"
)

DAYS = [-15, -10, -5, 0, 5, 10, 15]
TIME_LABELS = {
    -15: "lead 15 days",
    -10: "lead 10 days",
    -5: "lead 5 days",
    0: "peak",
    5: "lag 5 days",
    10: "lag 10 days",
    15: "lag 15 days",
}

LAT_TICKS = [30, 50, 70]
LAT_LABELS = ["30°N", "50°N", "70°N"]


def polish_axis(ax: plt.Axes, *, y_side: str) -> None:
    """Increase map-label readability without changing plotted data."""
    data_crs = ccrs.PlateCarree()
    ax.set_yticks(LAT_TICKS, crs=data_crs)
    ax.set_yticklabels(LAT_LABELS)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9.4,
        length=3.2,
        width=0.65,
        pad=2.2,
        labelleft=y_side == "left",
        labelright=y_side == "right",
        left=y_side == "left",
        right=y_side == "right",
    )
    ax.yaxis.set_ticks_position("right" if y_side == "right" else "left")

    # Contour labels are numeric text artists; panel and time labels are
    # retained separately and enlarged slightly for a consistent hierarchy.
    for text in ax.texts:
        content = text.get_text().strip()
        numeric = content.replace("−", "-").lstrip("-").isdigit()
        if numeric:
            text.set_fontsize(8.0)
            text.set_fontweight("semibold")
            text.set_color("#15181C")
            text.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=1.45,
                        foreground="white",
                    ),
                    path_effects.Normal(),
                ]
            )
        elif content.startswith("("):
            text.set_fontsize(9.8)
        elif content in TIME_LABELS.values():
            text.set_fontsize(9.4)


def main() -> None:
    base.configure_style()
    ds = xr.open_dataset(base.INPUT_NC)
    group = ds["z500_anomaly_5dmean_group13"]
    case = ds["z500_anomaly_5dmean_special2004"]
    significance = ds["significant_p005"]
    lons = ds["lon"].values
    lats = ds["lat"].values

    projection = ccrs.PlateCarree(central_longitude=180)
    # Sized for a near-full-width portrait manuscript page.  The individual
    # map axes are deliberately shallow, matching the compact GRL example.
    fig = plt.figure(figsize=(7.25, 9.10), facecolor="white")
    grid = fig.add_gridspec(
        7,
        2,
        left=0.072,
        right=0.928,
        bottom=0.105,
        top=0.986,
        wspace=0.040,
        hspace=0.055,
    )

    cf = None
    for row, day in enumerate(DAYS):
        ax_left = fig.add_subplot(grid[row, 0], projection=projection)
        ax_right = fig.add_subplot(grid[row, 1], projection=projection)

        cf = base.add_field(
            ax_left,
            data=group.sel(relative_day=day),
            significant=significance.sel(relative_day=day),
            panel_label=chr(ord("a") + row),
            time_label=TIME_LABELS[day],
            show_x=row == len(DAYS) - 1,
            y_side="left",
            lons=lons,
            lats=lats,
        )
        base.add_field(
            ax_right,
            data=case.sel(relative_day=day),
            significant=None,
            panel_label=chr(ord("h") + row),
            time_label=TIME_LABELS[day],
            show_x=row == len(DAYS) - 1,
            y_side="right",
            lons=lons,
            lats=lats,
        )

        polish_axis(ax_left, y_side="left")
        polish_axis(ax_right, y_side="right")

        # Fill the shallow manuscript panels without cropping the requested
        # longitude-latitude domain or changing any plotted values.
        ax_left.set_aspect("auto")
        ax_right.set_aspect("auto")

    cbar_ax = fig.add_axes([0.185, 0.045, 0.630, 0.022])
    cbar = fig.colorbar(
        cf,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=[-240, -160, -80, 0, 80, 160, 240],
        extend="both",
        extendrect=True,
        extendfrac=0.025,
        spacing="uniform",
    )
    cbar.ax.tick_params(labelsize=9.2, length=3.0, width=0.65, pad=2.5)
    cbar.set_label("gpm", fontsize=10.2, labelpad=4.0)
    cbar.outline.set_linewidth(0.6)

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=400, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    ds.close()

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
