import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

import ab_plot_helpers as core


N_BOOTSTRAP = 10000
RANDOM_SEED = 20260521

OUT_STEM = (
    "Figure_AB_features_14MHW_peak_lag30_ONDJF"
    "_anomaly_5d_bootstrap_p005"
)
OUT_CSV = (
    core.OUT_DIR
    / "AB_MHW_peak_lag_summary_14cases_ONDJF_anomaly_5d_bootstrap.csv"
)
DEFAULT_EVENT_RED = "#C00000"
DEFAULT_SERIES_COLORS = {
    "blocking_count": "#12355B",
    "intensity": "#8F1D14",
    "duration": "#2F7D5F",
}

BOOTSTRAP_SPECS = {
    "blocking_count": {
        "sample_col": "is_blocking",
        "clim_col": "clim_n_blocking_expected",
        "scale": len(core.PEAK_DATES),
        "observed_col": "blocking_count_anom_5d",
        "prefix": "bootstrap_blocking_count_anom_5d",
    },
    "intensity": {
        "sample_col": "daily_intensity_gpm",
        "clim_col": "clim_mean_intensity_all_gpm",
        "scale": 1.0,
        "observed_col": "intensity_anom_5d_gpm",
        "prefix": "bootstrap_intensity_anom_5d_gpm",
    },
    "duration": {
        "sample_col": "active_event_duration_days",
        "clim_col": "clim_mean_active_duration_all_days",
        "scale": 1.0,
        "observed_col": "duration_anom_5d_days",
        "prefix": "bootstrap_duration_anom_5d_days",
    },
}


def centered_rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("Expected a 2-D array with bootstrap iterations by lag.")
    if window % 2 == 0:
        raise ValueError("The centered rolling window must be odd.")

    half_window = window // 2
    smoothed = np.empty_like(values, dtype=float)
    for index in range(values.shape[1]):
        start = max(0, index - half_window)
        stop = min(values.shape[1], index + half_window + 1)
        smoothed[:, index] = values[:, start:stop].mean(axis=1)
    return smoothed


def add_5d_bootstrap(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    result = summary.copy()
    n_cases = samples["case_id"].nunique()
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n_cases, size=(n_bootstrap, n_cases))

    for spec in BOOTSTRAP_SPECS.values():
        values_by_case = (
            samples.pivot(
                index="case_id",
                columns="lag",
                values=spec["sample_col"],
            )
            .reindex(columns=core.LAGS)
            .to_numpy(dtype=float)
        )
        bootstrap_composites = (
            values_by_case[sample_indices].mean(axis=1) * spec["scale"]
        )
        climatology = result[spec["clim_col"]].to_numpy(dtype=float)
        bootstrap_anomalies = bootstrap_composites - climatology[None, :]
        bootstrap_5d = centered_rolling_mean(bootstrap_anomalies, window=5)

        prefix = spec["prefix"]
        result[f"{prefix}_mean"] = bootstrap_5d.mean(axis=0)
        result[f"{prefix}_p025"] = np.percentile(
            bootstrap_5d, 2.5, axis=0
        )
        result[f"{prefix}_p975"] = np.percentile(
            bootstrap_5d, 97.5, axis=0
        )
        result[f"{prefix}_sig_p005"] = (
            (result[f"{prefix}_p025"] > 0)
            | (result[f"{prefix}_p975"] < 0)
        )

    return result


def add_significance_dots(
    ax: plt.Axes,
    summary: pd.DataFrame,
    y_col: str,
    sig_col: str,
) -> None:
    significant = summary[sig_col].astype(bool)
    if not significant.any():
        return
    ax.scatter(
        summary.loc[significant, "lag"],
        summary.loc[significant, y_col],
        s=25,
        color="black",
        edgecolor="white",
        linewidth=0.35,
        zorder=8,
    )


def set_5d_anomaly_ylim(ax: plt.Axes, series: pd.Series) -> None:
    values = series.dropna().to_numpy(dtype=float)
    ymin = min(float(values.min()), 0.0)
    ymax = max(float(values.max()), 0.0)
    span = ymax - ymin
    pad = span * 0.14 if span > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)


def style_series_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    y_col: str,
    sig_col: str,
    color: str,
    ylabel: str,
    title: str,
) -> None:
    ax.plot(
        summary["lag"],
        summary[y_col],
        color=color,
        lw=2.4,
        label="5-day running mean anomaly",
        zorder=4,
    )
    add_significance_dots(ax, summary, y_col, sig_col)
    ax.axvline(0, color="black", lw=1)
    ax.axhline(0, color="0.55", lw=1, ls="--")
    ax.set_xlim(-30, 30)
    set_5d_anomaly_ylim(ax, summary[y_col])
    ax.set_xlabel("Lag relative to MHW peak day (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(color="0.9", lw=0.8)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="upper right",
        fontsize=9,
    )


def draw_figure(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    output_stem: str = OUT_STEM,
    event_red: str = DEFAULT_EVENT_RED,
    series_colors: dict[str, str] | None = None,
    hspace: float = 0.36,
) -> tuple[Path, Path]:
    if series_colors is None:
        series_colors = DEFAULT_SERIES_COLORS

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
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
        for index, date in enumerate(core.PEAK_DATES, start=1)
    ]

    fig = plt.figure(figsize=(13.4, 9.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.04, 1.0],
        width_ratios=[1.36, 1.0],
        left=0.145,
        right=0.965,
        bottom=0.085,
        top=0.90,
        wspace=0.29,
        hspace=hspace,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("white")
    vmax = np.nanpercentile(heat.values[block.values], 95)
    image = ax0.imshow(
        heat_masked,
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=vmax),
        extent=[
            core.LAGS.min() - 0.5,
            core.LAGS.max() + 0.5,
            len(core.PEAK_DATES) + 0.5,
            0.5,
        ],
        interpolation="nearest",
    )
    ax0.axvline(0, color="black", lw=1.2)
    ax0.set_xlim(-30.5, 30.5)
    ax0.set_xticks(np.arange(-30, 31, 10))
    ax0.set_yticks(np.arange(1, len(core.PEAK_DATES) + 1))
    ax0.set_yticklabels(peak_labels)
    for tick_label, peak_date in zip(
        ax0.get_yticklabels(),
        core.PEAK_DATES,
    ):
        tick_label.set_color(
            "black" if peak_date == pd.Timestamp("2004-12-24") else event_red
        )
    ax0.set_xlabel("Lag relative to MHW peak day (days)")
    ax0.set_ylabel("MHW peak date")
    ax0.set_title(
        "(a) Alaska blocking occurrence and daily intensity",
        loc="left",
        fontweight="bold",
    )
    ax0.grid(axis="x", color="0.88", lw=0.7)
    colorbar = fig.colorbar(image, ax=ax0, pad=0.014, fraction=0.045)
    colorbar.set_label("Daily AB intensity (gpm)")

    ax1 = fig.add_subplot(gs[0, 1])
    style_series_axis(
        ax1,
        summary,
        y_col="blocking_count_anom_5d",
        sig_col="bootstrap_blocking_count_anom_5d_sig_p005",
        color=series_colors["blocking_count"],
        ylabel="Blocking occurrence anomaly (count)",
        title="(b) Frequency",
    )

    ax2 = fig.add_subplot(gs[1, 0])
    style_series_axis(
        ax2,
        summary,
        y_col="intensity_anom_5d_gpm",
        sig_col="bootstrap_intensity_anom_5d_gpm_sig_p005",
        color=series_colors["intensity"],
        ylabel="Daily intensity anomaly (gpm)",
        title="(c) Intensity",
    )

    ax3 = fig.add_subplot(gs[1, 1])
    style_series_axis(
        ax3,
        summary,
        y_col="duration_anom_5d_days",
        sig_col="bootstrap_duration_anom_5d_days_sig_p005",
        color=series_colors["duration"],
        ylabel="Active AB event duration anomaly (days)",
        title="(d) Persistence",
    )

    fig.text(
        0.965,
        0.025,
        "Dots: 95% case-bootstrap CI for the 5-day mean anomaly excludes zero",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.25",
    )
    fig.suptitle(
        "Alaska blocking evolution from 30 days before to 30 days after "
        "14 early-winter MHW peaks",
        fontsize=13,
        fontweight="bold",
    )

    output_png = core.OUT_DIR / f"{output_stem}.png"
    output_pdf = core.OUT_DIR / f"{output_stem}.pdf"
    fig.savefig(output_png)
    fig.savefig(output_pdf)
    plt.close(fig)
    return output_png, output_pdf


def export_summary(summary: pd.DataFrame) -> None:
    source_columns = [
        "lag",
        "n_cases",
        "n_blocking",
        "clim_n_blocking_expected",
        "blocking_count_anom",
        "blocking_count_anom_5d",
        "mean_intensity_all_gpm",
        "clim_mean_intensity_all_gpm",
        "intensity_anom_gpm",
        "intensity_anom_5d_gpm",
        "mean_active_duration_all_days",
        "clim_mean_active_duration_all_days",
        "duration_anom_days",
        "duration_anom_5d_days",
    ]
    bootstrap_columns = [
        column
        for column in summary.columns
        if column.startswith("bootstrap_") and "_5d_" in column
    ]
    summary[source_columns + bootstrap_columns].to_csv(OUT_CSV, index=False)


def main() -> None:
    events = pd.read_csv(core.EVENT_FILE)
    daily = pd.read_csv(core.DAILY_INTENSITY_FILE)

    lookup = core.build_daily_lookup(events, daily)
    samples = core.build_lag_samples(lookup)
    summary = core.summarize(samples)
    summary_with_climatology, climatology_period = (
        core.add_climatology_baseline_and_ttest(
            samples,
            summary,
            lookup,
        )
    )
    summary_with_anomaly = core.add_anomaly_metrics(
        summary_with_climatology
    )
    summary_with_bootstrap = add_5d_bootstrap(
        samples,
        summary_with_anomaly,
    )

    export_summary(summary_with_bootstrap)
    output_png, output_pdf = draw_figure(
        samples,
        summary_with_bootstrap,
    )

    print("Saved figure PNG:", output_png)
    print("Saved figure PDF:", output_pdf)
    print("Saved 5-day bootstrap summary:", OUT_CSV)
    print("Cases:", len(core.PEAK_DATES))
    print("Lag range:", int(core.LAGS.min()), "to", int(core.LAGS.max()))
    print(
        "Climatology period:",
        f"{climatology_period[0]}-{climatology_period[1]}",
    )
    print("Bootstrap iterations:", N_BOOTSTRAP)
    print("Random seed:", RANDOM_SEED)
    for metric_name, spec in BOOTSTRAP_SPECS.items():
        sig_col = f"{spec['prefix']}_sig_p005"
        significant_lags = summary_with_bootstrap.loc[
            summary_with_bootstrap[sig_col],
            "lag",
        ].tolist()
        print(
            f"{metric_name} significant lags "
            f"({len(significant_lags)}):",
            significant_lags,
        )


if __name__ == "__main__":
    main()
