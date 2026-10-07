from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ttest_1samp


HEAT_BUDGET_SCRIPT = Path(__file__).resolve().parent / "thermal_plot_helpers.py"
HEAT_FLUX_N13 = Path(
    "/path/to/user/Documents/Codex/heat_flux/"
    "early_winter_peak_heatflux_group1_2020in_p005.csv"
)
HEAT_FLUX_2004 = Path(
    "/path/to/user/Documents/Codex/heat_flux/"
    "early_winter_peak_heatflux_group2_2004only_p005.csv"
)
Q1_PROFILES = Path(
    "/path/to/data/非绝热加热异常_ERA5反推/era5_jra55_box_vertical_profiles/"
    "JRA55_Q1_component_vertical_profiles_jra55_components_raw_clim_"
    "two_groups_13plus2004_values.csv"
)
Q1_EVENT_PROFILES = Path(
    "/path/to/data/非绝热加热异常_ERA5反推/era5_jra55_box_vertical_profiles/"
    "JRA55_Q1_component_vertical_profiles_jra55_components_raw_clim_"
    "lag_windows_1_7_2_7_two_groups_13plus2004_values.nc"
)
OUTPUT_TEMPERATURE = Path(
    "/path/to/user/Documents/Codex/heat_flux/"
    "early_winter_temperature_budget_two_panel_emphasis_v2.png"
)
OUTPUT_FLUX_Q1 = Path(
    "/path/to/user/Documents/Codex/heat_flux/"
    "early_winter_turbulent_heat_flux_q1_fourpanel_emphasis_v4_q1_significance.png"
)

LEAD_COLOR = "#EEF1F3"
LAG_COLOR = "#FAEEEE"
TEXT_COLOR = "#20242A"
GRID_COLOR = "#B9BEC2"

TOP_STYLES = {
    "tendency": {
        "label": "OISST SST tendency",
        "color": "#B33A3A",
        "linestyle": "-",
        "linewidth": 2.3,
    },
    "surface": {
        "label": r"ERA5 $Q_{\mathrm{net}}$ contribution",
        "color": "#315A7D",
        "linestyle": (0, (6.0, 2.2)),
        "linewidth": 2.0,
    },
    "residual": {
        "label": r"Residual $R^\ast$",
        "color": "#5A5148",
        "linestyle": (0, (1.2, 1.8)),
        "linewidth": 2.1,
    },
}

FLUX_STYLES = {
    "lh": {
        "column": "lh_mean",
        "significance": "lh_sig_p_lt_0.05",
        "label": "Latent heat",
        "color": "#0072B2",
        "linewidth": 2.7,
    },
    "sh": {
        "column": "sh_mean",
        "significance": "sh_sig_p_lt_0.05",
        "label": "Sensible heat",
        "color": "#E69F00",
        "linewidth": 2.7,
    },
    "total": {
        "column": "heat_flux_mean",
        "significance": "heat_flux_sig_p_lt_0.05",
        "label": "Total turbulent",
        "color": "#D2472F",
        "linewidth": 3.4,
    },
}

Q1_STYLES = {
    "cnvhr": ("Convective", "#D62728", "o"),
    "lrghr": ("Large-scale condensation", "#FF8C00", "s"),
    "lwhr": ("Longwave radiation", "#0072B2", "^"),
    "swhr": ("Shortwave radiation", "#009E73", "v"),
    "vdfhr": ("Vertical diffusion", "#7A5195", "D"),
    "Q1_sum_components": ("Sum Q1", "#111111", "o"),
}

N13_EVENT_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 10.4,
            "axes.labelsize": 11.4,
            "axes.linewidth": 0.90,
            "xtick.labelsize": 9.7,
            "ytick.labelsize": 9.7,
            "legend.fontsize": 9.0,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )


def load_temperature_budget() -> tuple[pd.DataFrame, pd.DataFrame]:
    module = load_module(HEAT_BUDGET_SCRIPT, "heat_budget_three_day")
    daily = module.DAILY
    args = argparse.Namespace(
        output_dir=daily.DEFAULT_OUTPUT_DIR,
        cache_root=daily.DEFAULT_CACHE_ROOT,
        oisst=daily.DEFAULT_OISST,
        era5_annual=daily.DEFAULT_ERA5_ANNUAL,
        era5_radiation=daily.DEFAULT_ERA5_RADIATION,
        era5_turbulent=daily.DEFAULT_ERA5_TURBULENT,
    )
    qnet = daily.load_era5_qnet(args)
    mld = daily.load_cfs_mld(args.cache_root)
    surface = daily.build_surface_anomalies(qnet, mld)
    tendency = daily.load_oisst_forward_tendency(args.oisst)
    daily_terms = daily.assemble_terms(tendency, surface)
    anomalies, _ = module.smooth_terms(daily_terms)
    daily.validate_event_windows(anomalies)

    n13_dates = pd.DatetimeIndex(
        pd.to_datetime(
            daily.BASE.GROUPS["cfs_n13_four_term"]["dates"]
        )
    )
    event_date = pd.DatetimeIndex(
        pd.to_datetime(
            daily.BASE.GROUPS["cfs_20041224_four_term"]["dates"]
        )
    )
    n13 = daily.make_group_frame(anomalies, n13_dates)
    event = daily.make_group_frame(anomalies, event_date)
    daily.verify_group_frame(n13, "n13", 13)
    daily.verify_group_frame(event, "2004", 1)
    return n13, event


def add_q1_n13_significance(q1: pd.DataFrame) -> pd.DataFrame:
    """Add two-sided one-sample t-test results across 13 event means."""
    records: list[dict[str, object]] = []
    with xr.open_dataset(Q1_EVENT_PROFILES) as dataset:
        for series in Q1_STYLES:
            if series == "Q1_sum_components":
                values = dataset["Q1_sum_anomaly"]
            else:
                values = dataset["component_anomaly"].sel(component=series)
            event_means = (
                values.isel(event=N13_EVENT_INDICES)
                .sel(lag=np.arange(0, 8))
                .mean("lag")
                .load()
            )
            result = ttest_1samp(
                event_means.values,
                popmean=0.0,
                axis=0,
                nan_policy="omit",
                alternative="two-sided",
            )
            composite_mean = event_means.mean("event").values
            for level, mean_value, p_value in zip(
                event_means["level_hPa"].values,
                composite_mean,
                result.pvalue,
            ):
                records.append(
                    {
                        "series": series,
                        "level_hPa": int(level),
                        "event_mean_K_day": float(mean_value),
                        "p_value": float(p_value),
                        "significant_p005": bool(p_value < 0.05),
                    }
                )

    significance = pd.DataFrame(records)
    n13 = q1.loc[q1["panel"] == "A"].merge_z500_daily(
        significance,
        on=["series", "level_hPa"],
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        n13["value_K_day"],
        n13["event_mean_K_day"],
        rtol=0,
        atol=2e-6,
    ):
        raise ValueError("Q1 event means do not reproduce the plotted N=13 profile.")
    return n13


def add_panel_header(
    axis: plt.Axes,
    label: str,
    column_label: str | None = None,
) -> None:
    axis.text(
        0.0,
        1.018,
        f"({label})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.3,
        fontweight="semibold",
        color=TEXT_COLOR,
        clip_on=False,
    )
    axis.text(
        1.0,
        1.018,
        "MHW region",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.8,
        fontweight="semibold",
        color=TEXT_COLOR,
        clip_on=False,
    )
    if column_label is not None:
        axis.text(
            0.5,
            1.018,
            column_label,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.8,
            fontweight="semibold",
            color=TEXT_COLOR,
            clip_on=False,
        )


def style_time_axis(axis: plt.Axes, right_side: bool) -> None:
    axis.axvspan(-10, 0, color=LEAD_COLOR, zorder=0)
    axis.axvspan(0, 10, color=LAG_COLOR, zorder=0)
    axis.axhline(0, color="#6F7478", linewidth=0.8, zorder=1)
    axis.axvline(
        0,
        color="#4D5154",
        linewidth=0.95,
        linestyle=(0, (4.0, 3.0)),
        zorder=2,
    )
    axis.set_xlim(-10, 10)
    axis.margins(x=0)
    axis.set_xticks([-10, -5, 0, 5, 10])
    axis.set_xticklabels(
        ["Lead 10", "Lead 5", "Peak", "Lag 5", "Lag 10"]
    )
    axis.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.5,
        alpha=0.55,
        zorder=0,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(TEXT_COLOR)
    if right_side:
        axis.yaxis.set_ticks_position("right")
        axis.tick_params(labelleft=False, labelright=True)
    else:
        axis.yaxis.set_ticks_position("left")
        axis.tick_params(labelleft=True, labelright=False)
    axis.tick_params(direction="out", length=3.2, width=0.75)


def plot_temperature_budget(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    label: str,
    show_significance: bool,
    right_side: bool,
    show_legend: bool,
) -> None:
    style_time_axis(axis, right_side)
    lags = frame["relative_day"].to_numpy(dtype=float)
    for term, style in TOP_STYLES.items():
        values = frame[f"{term}_mean"].to_numpy(dtype=float)
        axis.plot(
            lags,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=3,
            label=style["label"],
        )
        if show_significance:
            significant = frame[
                f"{term}_significant_p005"
            ].to_numpy(dtype=bool)
            axis.scatter(
                lags[significant],
                values[significant],
                s=18,
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.55,
                zorder=5,
            )
    axis.set_ylim(-0.068, 0.062)
    axis.set_yticks(np.arange(-0.06, 0.061, 0.02))
    add_panel_header(axis, label)
    if show_legend:
        axis.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.90,
            facecolor="white",
            edgecolor="#C9CDD0",
            ncol=1,
            handlelength=2.7,
            borderpad=0.45,
            labelspacing=0.35,
        )


def plot_heat_flux(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    label: str,
    show_significance: bool,
    right_side: bool,
    show_legend: bool,
    column_label: str | None = None,
) -> None:
    style_time_axis(axis, right_side)
    lags = frame["lag_day"].to_numpy(dtype=float)
    for style in FLUX_STYLES.values():
        values = frame[style["column"]].to_numpy(dtype=float)
        axis.plot(
            lags,
            values,
            color=style["color"],
            linewidth=style["linewidth"],
            solid_capstyle="round",
            zorder=3,
            label=style["label"],
        )
        if show_significance:
            significant = frame[
                style["significance"]
            ].to_numpy(dtype=bool)
            axis.scatter(
                lags[significant],
                values[significant],
                s=31,
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.55,
                zorder=5,
            )
    axis.set_ylim(-130, 120)
    axis.set_yticks([-120, -80, -40, 0, 40, 80, 120])
    add_panel_header(axis, label, column_label)
    if show_legend:
        axis.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.90,
            facecolor="white",
            edgecolor="#C9CDD0",
            ncol=1,
            handlelength=2.7,
            borderpad=0.45,
            labelspacing=0.35,
        )


def plot_q1_profile(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    label: str,
    right_side: bool,
    column_label: str | None = None,
    show_significance: bool = False,
) -> list[Line2D]:
    handles = []
    for series, (legend_label, color, marker) in Q1_STYLES.items():
        subset = frame.loc[frame["series"] == series].sort_values(
            "level_hPa",
            ascending=False,
        )
        line, = axis.plot(
            subset["value_K_day"],
            subset["level_hPa"],
            color=color,
            linewidth=2.55 if series != "Q1_sum_components" else 3.55,
            label=legend_label,
            zorder=4 if series == "Q1_sum_components" else 3,
        )
        handles.append(line)
        if show_significance:
            significant = subset["significant_p005"].fillna(False).to_numpy(
                dtype=bool
            )
            axis.scatter(
                subset.loc[significant, "value_K_day"],
                subset.loc[significant, "level_hPa"],
                s=29 if series != "Q1_sum_components" else 34,
                facecolor=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=6,
            )
    axis.axvline(0, color="#5B6064", linewidth=0.8, zorder=1)
    axis.set_xlim(-3.25, 3.25)
    axis.set_xticks([-2.5, 0, 2.5])
    axis.set_ylim(1025, 235)
    axis.set_yticks([1000, 925, 850, 700, 500, 400, 300, 250])
    axis.grid(
        color=GRID_COLOR,
        linewidth=0.55,
        linestyle=":",
        alpha=0.80,
        zorder=0,
    )
    if right_side:
        axis.yaxis.set_ticks_position("right")
        axis.tick_params(labelleft=False, labelright=True)
    else:
        axis.yaxis.set_ticks_position("left")
        axis.tick_params(labelleft=True, labelright=False)
    axis.tick_params(direction="out", length=3.2, width=0.75)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color(TEXT_COLOR)
    add_panel_header(axis, label, column_label)
    return handles


def main() -> None:
    configure_style()
    temperature_n13, temperature_2004 = load_temperature_budget()
    flux_n13 = pd.read_csv(HEAT_FLUX_N13, encoding="utf-8-sig")
    flux_2004 = pd.read_csv(HEAT_FLUX_2004, encoding="utf-8-sig")
    q1 = pd.read_csv(Q1_PROFILES, encoding="utf-8-sig")

    temperature_figure = plt.figure(figsize=(10.1, 3.65), facecolor="white")
    temperature_grid = temperature_figure.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.96,
        bottom=0.16,
        top=0.92,
        wspace=0.12,
    )
    temperature_axes = [
        temperature_figure.add_subplot(temperature_grid[0, column])
        for column in range(2)
    ]

    plot_temperature_budget(
        temperature_axes[0],
        temperature_n13,
        label="a",
        show_significance=True,
        right_side=False,
        show_legend=True,
    )
    plot_temperature_budget(
        temperature_axes[1],
        temperature_2004,
        label="b",
        show_significance=False,
        right_side=True,
        show_legend=False,
    )
    temperature_axes[0].set_ylabel(
        r"Temperature-budget anomaly ($^\circ$C day$^{-1}$)"
    )
    OUTPUT_TEMPERATURE.parent.mkdir(parents=True, exist_ok=True)
    temperature_figure.savefig(
        OUTPUT_TEMPERATURE,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    temperature_figure.savefig(
        OUTPUT_TEMPERATURE.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(temperature_figure)

    process_figure = plt.figure(figsize=(10.1, 7.6), facecolor="white")
    process_grid = process_figure.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.96,
        bottom=0.190,
        top=0.965,
        wspace=0.12,
        hspace=0.23,
        height_ratios=[1.0, 1.22],
    )
    process_axes = np.empty((2, 2), dtype=object)
    for row in range(2):
        for column in range(2):
            process_axes[row, column] = process_figure.add_subplot(
                process_grid[row, column]
            )

    plot_heat_flux(
        process_axes[0, 0],
        flux_n13,
        label="a",
        show_significance=True,
        right_side=False,
        show_legend=True,
        column_label="N=13",
    )
    plot_heat_flux(
        process_axes[0, 1],
        flux_2004,
        label="b",
        show_significance=False,
        right_side=True,
        show_legend=False,
        column_label="2004",
    )
    process_axes[0, 0].set_ylabel(
        r"Turbulent heat-flux anomaly (W m$^{-2}$)"
    )

    q1_n13 = add_q1_n13_significance(q1)
    q1_2004 = q1.loc[q1["panel"] == "B"]
    q1_handles = plot_q1_profile(
        process_axes[1, 0],
        q1_n13,
        label="c",
        right_side=False,
        column_label="N=13",
        show_significance=True,
    )
    plot_q1_profile(
        process_axes[1, 1],
        q1_2004,
        label="d",
        right_side=True,
        column_label="2004",
    )
    process_axes[1, 0].set_ylabel("Pressure (hPa)")
    process_axes[1, 0].set_xlabel(
        r"Heating anomaly (K day$^{-1}$)"
    )
    process_axes[1, 1].set_xlabel(
        r"Heating anomaly (K day$^{-1}$)"
    )

    process_figure.legend(
        handles=q1_handles,
        labels=[item[0] for item in Q1_STYLES.values()],
        loc="lower left",
        bbox_to_anchor=(0.105, 0.048, 0.805, 0.105),
        ncol=3,
        mode="expand",
        frameon=False,
        handlelength=2.9,
        columnspacing=2.3,
        handletextpad=0.72,
        labelspacing=0.72,
        borderaxespad=0.0,
        markerscale=1.16,
        fontsize=10.2,
    )

    OUTPUT_FLUX_Q1.parent.mkdir(parents=True, exist_ok=True)
    process_figure.savefig(
        OUTPUT_FLUX_Q1,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    process_figure.savefig(
        OUTPUT_FLUX_Q1.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(process_figure)
    print(OUTPUT_TEMPERATURE)
    print(OUTPUT_FLUX_Q1)


if __name__ == "__main__":
    main()
