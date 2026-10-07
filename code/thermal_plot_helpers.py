#!/usr/bin/env python3
"""Plot three-day means of the OISST-ERA5 three-term diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DAILY_SCRIPT = SCRIPT_DIR / "thermal_daily_helpers.py"
TERMS = ("tendency", "surface", "residual")
LAGS = np.arange(-10, 11, dtype=int)


def load_daily_module():
    spec = importlib.util.spec_from_file_location("three_term_daily", DAILY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load daily script: {DAILY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DAILY = load_daily_module()
BASE = DAILY.BASE


def output_path(output_dir: Path) -> Path:
    path = output_dir / (
        "early_winter_oisst_era5_qnet_three_term_anomaly_"
        "3day_running_mean_n13_20041224_ab_publication_v1.png"
    )
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite PNG: {path}")
    return path


def smooth_terms(
    anomalies: dict[str, dict[str, pd.Series]],
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, pd.Series]]:
    smoothed = {}
    counts = {}
    for source, terms in anomalies.items():
        frame = pd.concat(
            [terms["tendency"], terms["surface"]],
            axis=1,
        )
        frame.columns = ["tendency", "surface"]
        rolling = frame.rolling("3D", center=True, min_periods=2)
        means = rolling.mean()
        available = rolling.count().min(axis=1)
        means["residual"] = means["tendency"] - means["surface"]
        smoothed[source] = {
            term: means[term].rename(term)
            for term in TERMS
        }
        counts[source] = available.rename("days_in_running_mean")
    return smoothed, counts


def event_window_counts(
    counts: dict[str, pd.Series],
) -> np.ndarray:
    values = []
    for config in BASE.GROUPS.values():
        for text in config["dates"]:
            peak = pd.Timestamp(text)
            source = BASE.event_source(peak)
            for lag in LAGS:
                date = peak + pd.Timedelta(days=int(lag))
                values.append(int(counts[source].loc[date]))
    return np.asarray(values, dtype=int)


def plot_figure(
    n13: pd.DataFrame,
    single: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    DAILY.style_axis(axes[0], "(a)")
    DAILY.style_axis(axes[1], "(b)")
    DAILY.plot_panel(axes[0], n13, show_significance=True)
    DAILY.plot_panel(axes[1], single, show_significance=False)

    columns = [f"{term}_mean" for term in TERMS]
    combined = np.concatenate(
        [
            n13[columns].to_numpy(dtype=float).ravel(),
            single[columns].to_numpy(dtype=float).ravel(),
        ]
    )
    lower = float(np.nanmin(combined))
    upper = float(np.nanmax(combined))
    padding = 0.09 * (upper - lower)
    axes[0].set_ylim(lower - padding, upper + padding)
    axes[0].set_ylabel(
        r"3-day-mean temperature-budget anomaly ($^\circ$C day$^{-1}$)"
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        for style in DAILY.PLOT_STYLE.values()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        handlelength=3.2,
        columnspacing=2.2,
        handletextpad=0.7,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.92,
        bottom=0.225,
        wspace=0.10,
    )
    figure.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def main() -> None:
    args = DAILY.parse_args()
    DAILY.configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = output_path(args.output_dir)

    qnet = DAILY.load_era5_qnet(args)
    mld = DAILY.load_cfs_mld(args.cache_root)
    surface = DAILY.build_surface_anomalies(qnet, mld)
    tendency = DAILY.load_oisst_forward_tendency(args.oisst)
    daily_terms = DAILY.assemble_terms(tendency, surface)
    anomalies, rolling_counts = smooth_terms(daily_terms)
    DAILY.validate_event_windows(anomalies)

    n13_dates = pd.DatetimeIndex(
        pd.to_datetime(BASE.GROUPS["cfs_n13_four_term"]["dates"])
    )
    single_dates = pd.DatetimeIndex(
        pd.to_datetime(BASE.GROUPS["cfs_20041224_four_term"]["dates"])
    )
    n13 = DAILY.make_group_frame(anomalies, n13_dates)
    single = DAILY.make_group_frame(anomalies, single_dates)
    DAILY.verify_group_frame(n13, "n13", 13)
    DAILY.verify_group_frame(single, "2004", 1)
    plot_figure(n13, single, output)

    used_counts = event_window_counts(rolling_counts)
    if used_counts.min() < 2 or used_counts.max() > 3:
        raise AssertionError(f"Unexpected running-mean counts: {used_counts}")
    max_closure = max(
        float(np.nanmax(np.abs(n13["closure_error"]))),
        float(np.nanmax(np.abs(single["closure_error"]))),
    )
    significant = {
        term: int(n13[f"{term}_significant_p005"].sum())
        for term in TERMS
    }
    print(
        "Event-window running means: "
        f"{int((used_counts == 3).sum())} full 3-day, "
        f"{int((used_counts == 2).sum())} two-day boundary"
    )
    print(f"Maximum composite closure error: {max_closure:.3e} degree_C day-1")
    print(f"n=13 significant-point counts: {significant}")
    print(f"Saved PNG: {output}")


if __name__ == "__main__":
    main()
