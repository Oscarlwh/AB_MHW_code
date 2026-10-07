#!/usr/bin/env python3
"""Plot JRA-55 Q1 component profiles for lag1-7 and lag2-7 windows.

The calculation follows the existing raw-climatology component diagnostic:
1981-2020 no-leap daily climatology, no detrending, and cosine-latitude
weighted averaging over 200-225E, 40-50N.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plot_jra55_era5_box_vertical_profiles import (
    BOX,
    LEVELS,
    LAGS,
    OUT_ROOT,
    PEAK_DATES,
    setup_pressure_axis,
    symmetric_xlim,
)
from plot_jra55_q1_component_vertical_profiles_raw_clim_two_groups_13plus2004 import (
    month_series,
    target_table,
)
from plot_jra55_q1_component_vertical_profiles_two_groups_13plus2004 import (
    CLIM_YEARS,
    COMPONENTS,
    DIA_ROOT,
    PANEL_SPECS,
    calc_doy_nl,
)


WINDOWS = {
    "lag1_7": np.arange(1, 8, dtype=np.int16),
    "lag2_7": np.arange(2, 8, dtype=np.int16),
}
TAG = "jra55_components_raw_clim_lag_windows_1_7_2_7_two_groups_13plus2004"
LOW_LEVELS = [1000, 925, 850]


def component_event_lag(
    level: int,
    component: str,
    targets: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event value, matched climatology, and anomaly as event x lag."""
    cache: dict[tuple[int, str, int, int], pd.Series] = {}
    needed_months = sorted(int(value) for value in targets["month"].unique())

    clim_parts = []
    for year in CLIM_YEARS:
        for month in needed_months:
            clim_parts.append(month_series(level, component, year, month, cache))
    clim_series = pd.concat(clim_parts).sort_index()
    clim_doy = pd.Series(
        clim_series.to_numpy(dtype="float64"),
        index=[calc_doy_nl(ts) for ts in clim_series.index],
    ).groupby(level=0).mean()

    shape = (len(PEAK_DATES), len(LAGS))
    event_value = np.full(shape, np.nan, dtype=np.float32)
    climatology = np.full(shape, np.nan, dtype=np.float32)
    anomaly = np.full(shape, np.nan, dtype=np.float32)

    for row in targets.itertuples(index=False):
        target = pd.Timestamp(row.target_date)
        daily = month_series(level, component, target.year, target.month, cache)
        value = float(daily.loc[target])
        clim_value = float(clim_doy.loc[int(row.doy_nl)])
        event_value[int(row.event), int(row.lag)] = value
        climatology[int(row.event), int(row.lag)] = clim_value
        anomaly[int(row.event), int(row.lag)] = value - clim_value

    return event_value, climatology, anomaly


def aggregate_windows(values: np.ndarray) -> np.ndarray:
    """Aggregate event x lag x level x component data to window x panel."""
    result = np.full(
        (len(WINDOWS), len(PANEL_SPECS), len(LEVELS), len(COMPONENTS)),
        np.nan,
        dtype=np.float32,
    )
    for wi, lag_values in enumerate(WINDOWS.values()):
        for pi, spec in enumerate(PANEL_SPECS):
            event_values = np.asarray(spec["event_indices"], dtype=np.int16)
            selected = values[
                event_values[:, None],
                lag_values[None, :],
                :,
                :,
            ]
            result[wi, pi] = np.mean(selected, axis=(0, 1), dtype=np.float64)
    return result


def build_dataset() -> xr.Dataset:
    targets = target_table()
    component_keys = list(COMPONENTS)
    shape = (len(PEAK_DATES), len(LAGS), len(LEVELS), len(component_keys))
    event_value = np.full(shape, np.nan, dtype=np.float32)
    climatology = np.full(shape, np.nan, dtype=np.float32)
    anomaly = np.full(shape, np.nan, dtype=np.float32)

    for li, level in enumerate(LEVELS):
        print(f"[LEVEL] {level} hPa", flush=True)
        for ci, component in enumerate(component_keys):
            print(f"  [COMP] {component}", flush=True)
            values = component_event_lag(level, component, targets)
            event_value[:, :, li, ci] = values[0]
            climatology[:, :, li, ci] = values[1]
            anomaly[:, :, li, ci] = values[2]

    profile_value = aggregate_windows(event_value)
    profile_clim = aggregate_windows(climatology)
    profile_anom = aggregate_windows(anomaly)

    target_dates = np.full((len(PEAK_DATES), len(LAGS)), np.datetime64("NaT"), dtype="datetime64[ns]")
    doy_nl = np.full((len(PEAK_DATES), len(LAGS)), -1, dtype=np.int16)
    for row in targets.itertuples(index=False):
        target_dates[int(row.event), int(row.lag)] = np.datetime64(pd.Timestamp(row.target_date))
        doy_nl[int(row.event), int(row.lag)] = int(row.doy_nl)

    ds = xr.Dataset(
        {
            "component_event_value": (("event", "lag", "level_hPa", "component"), event_value),
            "component_climatology": (("event", "lag", "level_hPa", "component"), climatology),
            "component_anomaly": (("event", "lag", "level_hPa", "component"), anomaly),
            "Q1_sum_event_value": (("event", "lag", "level_hPa"), event_value.sum(axis=3)),
            "Q1_sum_climatology": (("event", "lag", "level_hPa"), climatology.sum(axis=3)),
            "Q1_sum_anomaly": (("event", "lag", "level_hPa"), anomaly.sum(axis=3)),
            "component_profile_event_value": (
                ("window", "panel", "level_hPa", "component"),
                profile_value,
            ),
            "component_profile_climatology": (
                ("window", "panel", "level_hPa", "component"),
                profile_clim,
            ),
            "component_profile_anomaly": (
                ("window", "panel", "level_hPa", "component"),
                profile_anom,
            ),
            "Q1_sum_profile_event_value": (
                ("window", "panel", "level_hPa"),
                profile_value.sum(axis=3),
            ),
            "Q1_sum_profile_climatology": (
                ("window", "panel", "level_hPa"),
                profile_clim.sum(axis=3),
            ),
            "Q1_sum_profile_anomaly": (
                ("window", "panel", "level_hPa"),
                profile_anom.sum(axis=3),
            ),
            "target_date": (("event", "lag"), target_dates),
            "doy_nl": (("event", "lag"), doy_nl),
            "window_lag_mask": (
                ("window", "lag"),
                np.array([[lag in values for lag in LAGS] for values in WINDOWS.values()], dtype=np.int8),
            ),
            "panel_event_mask": (
                ("panel", "event"),
                np.array(
                    [[event in spec["event_indices"] for event in range(len(PEAK_DATES))] for spec in PANEL_SPECS],
                    dtype=np.int8,
                ),
            ),
        },
        coords={
            "event": np.arange(len(PEAK_DATES), dtype=np.int16),
            "lag": np.asarray(LAGS, dtype=np.int16),
            "level_hPa": np.asarray(LEVELS, dtype=np.int32),
            "component": component_keys,
            "window": list(WINDOWS),
            "panel": [spec["key"] for spec in PANEL_SPECS],
            "peak_date": ("event", np.asarray(PEAK_DATES, dtype="datetime64[ns]")),
        },
        attrs={
            "title": "JRA-55 Q1 component box-mean lag-window sensitivity profiles",
            "units": "K day-1",
            "box": f"{BOX['lon_min']}-{BOX['lon_max']}E, {BOX['lat_min']}-{BOX['lat_max']}N",
            "windows": "lag1-7 mean and lag2-7 mean",
            "grouping": "13-event mean excludes only 2004-12-24; 2004-12-24 is separate",
            "method": "raw component daily values relative to 1981-2020 no-leap daily climatology; no detrending",
            "source": str(DIA_ROOT),
        },
    )
    for name in ds.data_vars:
        if "component_" in name or "Q1_sum" in name:
            ds[name].attrs["units"] = "K day-1"
    return ds


def profile_records(ds: xr.Dataset) -> pd.DataFrame:
    records = []
    for window in ds.window.values:
        for pi, spec in enumerate(PANEL_SPECS):
            common_panel = {
                "window": str(window),
                "lags": ",".join(str(value) for value in WINDOWS[str(window)]),
                "panel": spec["panel"],
                "key": spec["key"],
                "title": spec["title"],
                "event_indices": ",".join(str(value) for value in spec["event_indices"]),
                "event_dates": ",".join(PEAK_DATES[value] for value in spec["event_indices"]),
                "n_events": len(spec["event_indices"]),
            }
            for level in LEVELS:
                for component, info in COMPONENTS.items():
                    selection = dict(window=window, panel=spec["key"], level_hPa=level, component=component)
                    records.append(
                        {
                            **common_panel,
                            "level_hPa": level,
                            "series": component,
                            "label": info["label"],
                            "event_value_K_day": float(ds.component_profile_event_value.sel(**selection)),
                            "climatology_K_day": float(ds.component_profile_climatology.sel(**selection)),
                            "anomaly_K_day": float(ds.component_profile_anomaly.sel(**selection)),
                        }
                    )
                selection = dict(window=window, panel=spec["key"], level_hPa=level)
                records.append(
                    {
                        **common_panel,
                        "level_hPa": level,
                        "series": "Q1_sum_components",
                        "label": "Sum of five components",
                        "event_value_K_day": float(ds.Q1_sum_profile_event_value.sel(**selection)),
                        "climatology_K_day": float(ds.Q1_sum_profile_climatology.sel(**selection)),
                        "anomaly_K_day": float(ds.Q1_sum_profile_anomaly.sel(**selection)),
                    }
                )
    return pd.DataFrame(records)


def low_level_records(profile_df: pd.DataFrame) -> pd.DataFrame:
    subset = profile_df[profile_df["level_hPa"].isin(LOW_LEVELS)]
    index_columns = ["window", "lags", "level_hPa", "series", "label"]
    wide = subset.pivot(index=index_columns, columns="key", values=[
        "event_value_K_day",
        "climatology_K_day",
        "anomaly_K_day",
    ])
    wide.columns = [f"{metric}_{panel}" for metric, panel in wide.columns]
    wide = wide.reset_index()
    wide["anomaly_difference_13minus2004_K_day"] = (
        wide["anomaly_K_day_mean_13_events"] - wide["anomaly_K_day_event_2004_12_24"]
    )
    return wide


def cnvhr_event_records(ds: xr.Dataset) -> pd.DataFrame:
    records = []
    for window, lags in WINDOWS.items():
        for event, peak_date in enumerate(PEAK_DATES):
            selection = dict(event=event, lag=lags, level_hPa=850, component="cnvhr")
            event_value = float(ds.component_event_value.sel(**selection).mean("lag"))
            climatology = float(ds.component_climatology.sel(**selection).mean("lag"))
            anomaly = float(ds.component_anomaly.sel(**selection).mean("lag"))
            records.append(
                {
                    "window": window,
                    "lags": ",".join(str(value) for value in lags),
                    "event": event,
                    "peak_date": peak_date,
                    "group": "2004 case" if peak_date == "2004-12-24" else "13-event group",
                    "event_value_K_day": event_value,
                    "climatology_K_day": climatology,
                    "anomaly_K_day": anomaly,
                }
            )
    return pd.DataFrame(records)


def add_component_lines(ax: plt.Axes, ds: xr.Dataset, window: str, panel: str, show_labels: bool) -> None:
    levels = ds.level_hPa.values
    for component, info in COMPONENTS.items():
        ax.plot(
            ds.component_profile_anomaly.sel(window=window, panel=panel, component=component),
            levels,
            color=info["color"],
            linewidth=1.5,
            marker=info["marker"],
            markersize=3.5,
            label=info["label"] if show_labels else None,
        )
    ax.plot(
        ds.Q1_sum_profile_anomaly.sel(window=window, panel=panel),
        levels,
        color="#111111",
        linewidth=2.3,
        marker="o",
        markersize=4.2,
        label="Sum Q1" if show_labels else None,
    )
    setup_pressure_axis(ax)


def plot_single_window(ds: xr.Dataset, window: str, xlim: tuple[float, float], out_root: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.5), sharey=True)
    for i, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        add_component_lines(ax, ds, window, spec["key"], show_labels=i == 0)
        ax.set_xlim(*xlim)
        ax.set_title(f"({spec['panel']}) {spec['title']}", loc="left", fontsize=11)
        ax.set_xlabel("Heating anomaly (K day$^{-1}$)")
        if i > 0:
            ax.set_ylabel("")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.5)
    lag_text = window.replace("lag", "lag").replace("_", "-")
    fig.suptitle(f"JRA-55 Q1 component contributions, raw-clim {lag_text} mean", y=0.97, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 0.88, 0.94])
    stem = f"JRA55_Q1_component_vertical_profiles_raw_clim_{window}_two_groups_13plus2004"
    png = out_root / f"{stem}.png"
    pdf = out_root / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_combined(ds: xr.Dataset, xlim: tuple[float, float], out_root: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.0), sharex=True, sharey=True)
    panel_letters = iter("ABCD")
    for row, window in enumerate(WINDOWS):
        for col, spec in enumerate(PANEL_SPECS):
            ax = axes[row, col]
            add_component_lines(ax, ds, window, spec["key"], show_labels=row == 0 and col == 0)
            ax.set_xlim(*xlim)
            ax.set_title(f"({next(panel_letters)}) {spec['title']}", loc="left", fontsize=10.5)
            ax.text(0.98, 0.98, window.replace("_", "-"), transform=ax.transAxes, ha="right", va="top", fontsize=9)
            if row == 1:
                ax.set_xlabel("Heating anomaly (K day$^{-1}$)")
            if col == 1:
                ax.set_ylabel("")
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.03, 1.18), ncol=3, frameon=False, fontsize=7.3)
    fig.suptitle("JRA-55 Q1 component contributions, raw-clim lag-window sensitivity", y=0.99, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = "JRA55_Q1_component_vertical_profiles_raw_clim_lag_windows_1_7_2_7_two_groups_13plus2004"
    png = out_root / f"{stem}.png"
    pdf = out_root / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_cnvhr_box(event_df: pd.DataFrame, out_root: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.8), sharey=True)
    rng = np.random.default_rng(20260715)
    for ax, window in zip(axes, WINDOWS):
        data = event_df[event_df["window"] == window]
        group_values = data[data["group"] == "13-event group"]["anomaly_K_day"].to_numpy()
        case_value = float(data[data["group"] == "2004 case"]["anomaly_K_day"].iloc[0])
        ax.boxplot(
            [group_values],
            positions=[0],
            widths=0.38,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#d9e6f2", "edgecolor": "#1f4e79", "linewidth": 1.1},
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": "#1f4e79"},
            capprops={"color": "#1f4e79"},
        )
        jitter = rng.uniform(-0.08, 0.08, len(group_values))
        ax.scatter(jitter, group_values, s=22, color="#1f77b4", edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter([1], [case_value], s=48, marker="D", color="#d62728", edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks([0, 1], ["13 events", "2004 case"])
        ax.set_title(window.replace("_", "-"), fontsize=10.5)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, linestyle=":")
    axes[0].set_ylabel("850 hPa CNVHR anomaly (K day$^{-1}$)")
    fig.suptitle("Event distribution of JRA-55 convective heating anomaly", y=0.98, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    stem = "JRA55_850hPa_CNVHR_event_distribution_raw_clim_lag_windows_1_7_2_7"
    png = out_root / f"{stem}.png"
    pdf = out_root / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def validate(ds: xr.Dataset, profile_df: pd.DataFrame, event_df: pd.DataFrame) -> None:
    expected_sizes = {
        "event": 14,
        "lag": 8,
        "level_hPa": 8,
        "component": 5,
        "window": 2,
        "panel": 2,
    }
    if dict(ds.sizes) != expected_sizes:
        raise ValueError(f"Unexpected dataset sizes: {dict(ds.sizes)} != {expected_sizes}")
    for name in ds.data_vars:
        if ds[name].dtype.kind in "fc" and not np.isfinite(ds[name].values).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if len(profile_df) != 2 * 2 * 8 * 6:
        raise ValueError(f"Unexpected profile CSV rows: {len(profile_df)}")
    if len(event_df) != 2 * 14:
        raise ValueError(f"Unexpected event CSV rows: {len(event_df)}")

    checks = {
        ("lag1_7", "mean_13_events"): 2.416,
        ("lag2_7", "mean_13_events"): 2.390,
        ("lag1_7", "event_2004_12_24"): -0.283,
        ("lag2_7", "event_2004_12_24"): -0.382,
    }
    for (window, panel), expected in checks.items():
        actual = float(
            ds.component_profile_anomaly.sel(
                window=window,
                panel=panel,
                level_hPa=850,
                component="cnvhr",
            )
        )
        if not np.isclose(actual, expected, atol=0.01):
            raise ValueError(f"850 hPa CNVHR check failed for {window}/{panel}: {actual} vs {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    ds = build_dataset()
    profile_df = profile_records(ds)
    low_level_df = low_level_records(profile_df)
    event_df = cnvhr_event_records(ds)
    validate(ds, profile_df, event_df)

    xlim = symmetric_xlim(
        ds.component_profile_anomaly.values,
        ds.Q1_sum_profile_anomaly.values,
        pad=0.15,
    )
    figure_files = []
    for window in WINDOWS:
        figure_files.extend(plot_single_window(ds, window, xlim, args.out_root))
    figure_files.extend(plot_combined(ds, xlim, args.out_root))
    figure_files.extend(plot_cnvhr_box(event_df, args.out_root))

    nc_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_values.nc"
    csv_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_profiles.csv"
    low_file = args.out_root / f"JRA55_Q1_component_vertical_profiles_{TAG}_low_level_diagnostics.csv"
    event_file = args.out_root / f"JRA55_850hPa_CNVHR_events_{TAG}.csv"
    ds.to_netcdf(nc_file)
    profile_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    low_level_df.to_csv(low_file, index=False, encoding="utf-8-sig")
    event_df.to_csv(event_file, index=False, encoding="utf-8-sig")

    print(f"[NC] {nc_file}")
    print(f"[CSV] {csv_file}")
    print(f"[LOW] {low_file}")
    print(f"[EVENT] {event_file}")
    for figure in figure_files:
        print(f"[FIG] {figure}")
    print(f"[XLIM] {xlim}")


if __name__ == "__main__":
    main()
