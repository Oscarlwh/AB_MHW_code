from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

import ab_bootstrap_helpers as figure


OUT_STEM = (
    "Figure_AB_features_14MHW_peak_lag30_ONDJF"
    "_anomaly_5d_bootstrap_p005_equalwidth_largefont_terminology"
)
EVENT_RED = "#FF0000"
SERIES_COLORS = {
    "blocking_count": "#5B8DB8",
    "intensity": "#D96B5F",
    "duration": "#6BAA87",
}
THIN_WIDTH = 1.25
THICK_WIDTH = 3.35


def add_variable_width_line(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    significant: np.ndarray,
    color: str,
) -> None:
    segments = []
    widths = []
    for index in range(len(x) - 1):
        midpoint = np.array(
            [
                0.5 * (x[index] + x[index + 1]),
                0.5 * (y[index] + y[index + 1]),
            ]
        )
        segments.append(
            np.array([[x[index], y[index]], midpoint])
        )
        widths.append(
            THICK_WIDTH if significant[index] else THIN_WIDTH
        )
        segments.append(
            np.array([midpoint, [x[index + 1], y[index + 1]]])
        )
        widths.append(
            THICK_WIDTH if significant[index + 1] else THIN_WIDTH
        )
    collection = LineCollection(
        segments,
        colors=color,
        linewidths=widths,
        capstyle="round",
        joinstyle="round",
        zorder=4,
    )
    axis.add_collection(collection)


def style_series_axis(
    axis: plt.Axes,
    summary: pd.DataFrame,
    y_col: str,
    sig_col: str,
    color: str,
    ylabel: str,
    title: str,
) -> None:
    x = summary["lag"].to_numpy(dtype=float)
    y = summary[y_col].to_numpy(dtype=float)
    significant = summary[sig_col].to_numpy(dtype=bool)
    add_variable_width_line(axis, x, y, significant, color)

    axis.axvline(0, color="black", lw=1)
    axis.axhline(0, color="0.55", lw=1, ls="--")
    axis.set_xlim(-30, 30)
    figure.set_5d_anomaly_ylim(axis, summary[y_col])
    axis.set_xlabel("Lag relative to MHW peak day (days)")
    axis.set_ylabel(ylabel)
    axis.set_title(
        title,
        loc="left",
        fontsize=13.0,
        fontweight="semibold",
        pad=7,
    )
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=11.0,
        width=0.9,
        length=4.2,
    )
    axis.grid(color="0.9", lw=0.8)


def draw_figure(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.0,
            "axes.labelsize": 12.2,
            "axes.titlesize": 13.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )

    heat = samples.pivot(
        index="case_id",
        columns="lag",
        values="daily_intensity_gpm",
    )
    block = samples.pivot(
        index="case_id",
        columns="lag",
        values="is_blocking",
    ).astype(bool)
    heat_masked = np.ma.masked_where(~block.values, heat.values)
    peak_labels = [
        f"{index:02d}  {date.strftime('%Y-%m-%d')}"
        for index, date in enumerate(figure.core.PEAK_DATES, start=1)
    ]

    canvas = plt.figure(
        figsize=(13.4, 9.25),
        constrained_layout=False,
    )
    grid = canvas.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.145,
        right=0.975,
        bottom=0.075,
        top=0.96,
        wspace=0.31,
        hspace=0.26,
    )

    axis_a = canvas.add_subplot(grid[0, 0])
    colormap = plt.get_cmap("YlOrRd").copy()
    colormap.set_bad("white")
    maximum = np.nanpercentile(heat.values[block.values], 95)
    image = axis_a.imshow(
        heat_masked,
        aspect="auto",
        cmap=colormap,
        norm=Normalize(vmin=0, vmax=maximum),
        extent=[
            figure.core.LAGS.min() - 0.5,
            figure.core.LAGS.max() + 0.5,
            len(figure.core.PEAK_DATES) + 0.5,
            0.5,
        ],
        interpolation="nearest",
    )
    axis_a.axvline(0, color="black", lw=1.2)
    axis_a.set_xlim(-30.5, 30.5)
    axis_a.set_xticks(np.arange(-30, 31, 10))
    axis_a.set_yticks(
        np.arange(1, len(figure.core.PEAK_DATES) + 1)
    )
    axis_a.set_yticklabels(peak_labels)
    for tick_label, peak_date in zip(
        axis_a.get_yticklabels(),
        figure.core.PEAK_DATES,
    ):
        tick_label.set_color(
            "black"
            if peak_date == pd.Timestamp("2004-12-24")
            else EVENT_RED
        )
    axis_a.set_xlabel("Lag relative to MHW peak day (days)")
    axis_a.set_ylabel("MHW peak date")
    axis_a.set_title(
        "(a) Alaska blocking occurrence and daily intensity",
        loc="left",
        fontsize=13.0,
        fontweight="semibold",
        pad=7,
    )
    axis_a.tick_params(
        axis="both",
        which="major",
        labelsize=10.7,
        width=0.9,
        length=4.2,
    )
    axis_a.grid(axis="x", color="0.88", lw=0.7)
    colorbar = canvas.colorbar(
        image,
        ax=axis_a,
        pad=0.014,
        fraction=0.045,
    )
    colorbar.set_label("Daily AB intensity (gpm)", fontsize=12.0)
    colorbar.ax.tick_params(labelsize=10.5, width=0.8, length=3.5)

    axis_b = canvas.add_subplot(grid[0, 1])
    style_series_axis(
        axis_b,
        summary,
        y_col="blocking_count_anom_5d",
        sig_col="bootstrap_blocking_count_anom_5d_sig_p005",
        color=SERIES_COLORS["blocking_count"],
        ylabel="AB occurrence anomaly (count)",
        title="(b) Occurrence",
    )

    axis_c = canvas.add_subplot(grid[1, 0])
    style_series_axis(
        axis_c,
        summary,
        y_col="intensity_anom_5d_gpm",
        sig_col="bootstrap_intensity_anom_5d_gpm_sig_p005",
        color=SERIES_COLORS["intensity"],
        ylabel="Daily AB intensity anomaly (gpm)",
        title="(c) Intensity",
    )

    axis_d = canvas.add_subplot(grid[1, 1])
    style_series_axis(
        axis_d,
        summary,
        y_col="duration_anom_5d_days",
        sig_col="bootstrap_duration_anom_5d_days_sig_p005",
        color=SERIES_COLORS["duration"],
        ylabel="AB event–duration anomaly (days)",
        title="(d) Event duration",
    )

    output_png = figure.core.OUT_DIR / f"{OUT_STEM}.png"
    output_pdf = figure.core.OUT_DIR / f"{OUT_STEM}.pdf"
    canvas.savefig(output_png, facecolor="white")
    canvas.savefig(output_pdf, facecolor="white")
    plt.close(canvas)
    return output_png


def main() -> None:
    events = pd.read_csv(figure.core.EVENT_FILE)
    daily = pd.read_csv(figure.core.DAILY_INTENSITY_FILE)

    lookup = figure.core.build_daily_lookup(events, daily)
    samples = figure.core.build_lag_samples(lookup)
    summary = figure.core.summarize(samples)
    summary_with_climatology, _ = (
        figure.core.add_climatology_baseline_and_ttest(
            samples,
            summary,
            lookup,
        )
    )
    summary_with_anomaly = figure.core.add_anomaly_metrics(
        summary_with_climatology
    )
    summary_with_bootstrap = figure.add_5d_bootstrap(
        samples,
        summary_with_anomaly,
    )

    output_png = draw_figure(
        samples,
        summary_with_bootstrap,
    )
    print(output_png)


if __name__ == "__main__":
    main()
