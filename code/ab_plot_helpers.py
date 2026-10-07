import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats
import xarray as xr


BASE_DIR = Path("/path/to/data/ABH_170E240E_1981_2024")
EVENT_FILE = BASE_DIR / "events_ONDJF" / "ERA5_ABH_events_ONDJF_170E-240E_50N-75N_1981_2024.csv"
DAILY_INTENSITY_FILE = BASE_DIR / "compute_blocking_intensity" / "ABH_daily_intensity_by_event_ONDJF_170E-240E_50N-75N_1981_2024.csv"
Z500_CLIM_FILE = (
    BASE_DIR
    / "anomaly"
    / "ERA5_Z500_clim_NOLEAP_1981_2020_gpm_170E240E_35N90N_detrended.nc"
)

OUT_DIR = Path("/path/to/user/Documents/Codex/ABH/figures/AB_MHW_peak_core")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PEAK_DATES = pd.to_datetime(
    [
        "1985-11-29",
        "1986-11-19",
        "1989-11-18",
        "1989-12-24",
        "1991-11-10",
        "1993-11-26",
        "2004-12-24",
        "2015-11-06",
        "2015-11-30",
        "2018-11-19",
        "2019-11-10",
        "2020-11-13",
        "2023-11-03",
        "2023-12-14",
    ]
)

LAGS = np.arange(-30, 31)
N_MONTE_CARLO = 10000
N_BOOTSTRAP = 10000
RANDOM_SEED = 20260521

METRIC_SPECS = {
    "blocking_frequency": {
        "sample_col": "is_blocking",
        "mc_p_col": "mc_p_blocking_frequency",
        "t_p_col": "ttest_p_blocking_frequency",
        "null_mean_col": "mc_null_mean_blocking_frequency",
        "clim_mean_col": "clim_blocking_frequency",
        "bootstrap_sig_p005_col": "bootstrap_sig_p005_blocking_frequency",
        "bootstrap_sig_p010_col": "bootstrap_sig_p010_blocking_frequency",
    },
    "mean_intensity_all_gpm": {
        "sample_col": "daily_intensity_gpm",
        "mc_p_col": "mc_p_mean_intensity_all_gpm",
        "t_p_col": "ttest_p_mean_intensity_all_gpm",
        "null_mean_col": "mc_null_mean_intensity_all_gpm",
        "clim_mean_col": "clim_mean_intensity_all_gpm",
        "bootstrap_sig_p005_col": "bootstrap_sig_p005_mean_intensity_all_gpm",
        "bootstrap_sig_p010_col": "bootstrap_sig_p010_mean_intensity_all_gpm",
    },
    "mean_active_duration_all_days": {
        "sample_col": "active_event_duration_days",
        "mc_p_col": "mc_p_mean_active_duration_all_days",
        "t_p_col": "ttest_p_mean_active_duration_all_days",
        "null_mean_col": "mc_null_mean_active_duration_all_days",
        "clim_mean_col": "clim_mean_active_duration_all_days",
        "bootstrap_sig_p005_col": "bootstrap_sig_p005_mean_active_duration_all_days",
        "bootstrap_sig_p010_col": "bootstrap_sig_p010_mean_active_duration_all_days",
    },
}


def build_daily_lookup(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    daily = daily.copy()

    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])
    daily["date"] = pd.to_datetime(daily["date"])

    rows = []
    daily_by_event = {
        int(eid): sub.set_index("date")
        for eid, sub in daily.groupby("event_id", sort=False)
    }

    for _, ev in events.iterrows():
        event_id = int(ev["event_id"])
        event_days = pd.date_range(ev["start_date"], ev["end_date"], freq="D")
        daily_sub = daily_by_event.get(event_id)

        for date in event_days:
            if daily_sub is None or date not in daily_sub.index:
                intensity = np.nan
            else:
                intensity = float(daily_sub.loc[date, "daily_intensity_gpm"])

            rows.append(
                {
                    "date": date,
                    "event_id": event_id,
                    "daily_intensity_gpm": intensity,
                    "duration_days": int(ev["duration_days"]),
                    "start_date": ev["start_date"],
                    "end_date": ev["end_date"],
                }
            )

    lookup = pd.DataFrame(rows)
    if lookup["date"].duplicated().any():
        dup = lookup.loc[lookup["date"].duplicated(keep=False), ["date", "event_id"]]
        raise ValueError(f"Overlapping AB events found:\n{dup}")

    return lookup.set_index("date").sort_index()


def build_lag_samples(lookup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, peak_date in enumerate(PEAK_DATES, start=1):
        for lag in LAGS:
            date = peak_date + pd.Timedelta(days=int(lag))
            is_blocking = date in lookup.index

            if is_blocking:
                rec = lookup.loc[date]
                event_id = int(rec["event_id"])
                intensity = float(rec["daily_intensity_gpm"])
                duration = int(rec["duration_days"])
                ab_start = pd.to_datetime(rec["start_date"])
                ab_end = pd.to_datetime(rec["end_date"])
            else:
                event_id = 0
                intensity = 0.0
                duration = 0
                ab_start = pd.NaT
                ab_end = pd.NaT

            rows.append(
                {
                    "case_id": case_id,
                    "mhw_peak_date": peak_date,
                    "lag": int(lag),
                    "date": date,
                    "is_blocking": int(is_blocking),
                    "ab_event_id": event_id,
                    "daily_intensity_gpm": intensity,
                    "active_event_duration_days": duration,
                    "ab_start_date": ab_start,
                    "ab_end_date": ab_end,
                }
            )

    return pd.DataFrame(rows)


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    def conditional_mean(series: pd.Series, mask: pd.Series) -> float:
        sub = series[mask.astype(bool)]
        if len(sub) == 0:
            return np.nan
        return float(sub.mean())

    rows = []
    for lag, sub in samples.groupby("lag", sort=True):
        mask = sub["is_blocking"].astype(bool)
        rows.append(
            {
                "lag": int(lag),
                "n_cases": int(len(sub)),
                "n_blocking": int(mask.sum()),
                "blocking_frequency": float(mask.mean()),
                "mean_intensity_all_gpm": float(sub["daily_intensity_gpm"].mean()),
                "mean_intensity_blocking_only_gpm": conditional_mean(
                    sub["daily_intensity_gpm"], mask
                ),
                "mean_active_duration_all_days": float(
                    sub["active_event_duration_days"].mean()
                ),
                "mean_active_duration_blocking_only_days": conditional_mean(
                    sub["active_event_duration_days"], mask
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["blocking_frequency_5d"] = (
        summary["blocking_frequency"].rolling(5, center=True, min_periods=1).mean()
    )
    summary["n_blocking_5d"] = (
        summary["n_blocking"].rolling(5, center=True, min_periods=1).mean()
    )
    summary["mean_intensity_all_5d_gpm"] = (
        summary["mean_intensity_all_gpm"].rolling(5, center=True, min_periods=1).mean()
    )
    summary["mean_active_duration_all_5d_days"] = (
        summary["mean_active_duration_all_days"].rolling(5, center=True, min_periods=1).mean()
    )
    summary["mean_intensity_blocking_only_5d_gpm"] = (
        summary["mean_intensity_blocking_only_gpm"]
        .rolling(5, center=True, min_periods=1)
        .mean()
    )
    summary["mean_active_duration_blocking_only_5d_days"] = (
        summary["mean_active_duration_blocking_only_days"]
        .rolling(5, center=True, min_periods=1)
        .mean()
    )
    return summary


def build_metric_frame(lookup: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("1981-01-01")
    end = pd.Timestamp("2024-12-31")
    metrics = pd.DataFrame(
        0.0,
        index=pd.date_range(start, end, freq="D"),
        columns=[
            "is_blocking",
            "daily_intensity_gpm",
            "active_event_duration_days",
        ],
    )

    common_dates = metrics.index.intersection(lookup.index)
    metrics.loc[common_dates, "is_blocking"] = 1.0
    metrics.loc[common_dates, "daily_intensity_gpm"] = lookup.loc[
        common_dates, "daily_intensity_gpm"
    ].astype(float)
    metrics.loc[common_dates, "active_event_duration_days"] = lookup.loc[
        common_dates, "duration_days"
    ].astype(float)
    return metrics


def no_leap_dayofyear(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    doy = dates.dayofyear.astype(int)
    is_after_feb_in_leap_year = dates.is_leap_year & (doy > 59)
    doy = doy - is_after_feb_in_leap_year.astype(int)
    return pd.Series(doy, index=dates)


def read_climatology_period(clim_file: Path = Z500_CLIM_FILE) -> tuple[int, int]:
    with xr.open_dataset(clim_file) as ds:
        period = str(ds.attrs.get("climatology_period", ""))

    try:
        start_year, end_year = [int(part) for part in period.split("-")]
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse climatology_period={period!r} from {clim_file}"
        ) from exc

    if (start_year, end_year) != (1981, 2020):
        raise ValueError(
            f"Expected AB climatology period 1981-2020, got {period!r}."
        )
    return start_year, end_year


def build_ab_indicator_climatology(
    lookup: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    metric_frame = build_metric_frame(lookup)
    base = metric_frame.loc[f"{start_year}-01-01": f"{end_year}-12-31"].copy()

    base["month"] = base.index.month
    base = base.loc[base["month"].isin([10, 11, 12, 1, 2])].copy()
    base["doy"] = no_leap_dayofyear(base.index).to_numpy()

    is_feb29 = (base.index.month == 2) & (base.index.day == 29)
    base = base.loc[~is_feb29]

    climatology = (
        base.groupby("doy")[
            [
                "is_blocking",
                "daily_intensity_gpm",
                "active_event_duration_days",
            ]
        ]
        .mean()
        .rename(
            columns={
                "is_blocking": "clim_blocking_frequency",
                "daily_intensity_gpm": "clim_mean_intensity_all_gpm",
                "active_event_duration_days": "clim_mean_active_duration_all_days",
            }
        )
    )
    return climatology


def add_climatology_baseline_and_ttest(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[int, int]]:
    start_year, end_year = read_climatology_period()
    climatology = build_ab_indicator_climatology(lookup, start_year, end_year)
    summary = summary.copy()

    climatology_rows = []
    for lag in LAGS:
        target_dates = PEAK_DATES + pd.to_timedelta(int(lag), unit="D")
        doys = no_leap_dayofyear(pd.DatetimeIndex(target_dates)).to_numpy()
        clim_at_lag = climatology.reindex(doys).mean(axis=0)
        climatology_rows.append(
            {
                "lag": int(lag),
                "clim_blocking_frequency": float(clim_at_lag["clim_blocking_frequency"]),
                "clim_n_blocking_expected": float(
                    clim_at_lag["clim_blocking_frequency"] * len(PEAK_DATES)
                ),
                "clim_mean_intensity_all_gpm": float(
                    clim_at_lag["clim_mean_intensity_all_gpm"]
                ),
                "clim_mean_active_duration_all_days": float(
                    clim_at_lag["clim_mean_active_duration_all_days"]
                ),
            }
        )

    summary = summary.merge_z500_daily(pd.DataFrame(climatology_rows), on="lag", how="left")

    for metric_name, spec in METRIC_SPECS.items():
        observed_by_case = (
            samples.pivot(index="case_id", columns="lag", values=spec["sample_col"])
            .reindex(columns=LAGS)
            .to_numpy(dtype=float)
        )
        clim_mean = summary[spec["clim_mean_col"]].to_numpy(dtype=float)
        t_p_values = []
        for ilag in range(len(LAGS)):
            result = stats.ttest_1samp(
                observed_by_case[:, ilag],
                popmean=float(clim_mean[ilag]),
                nan_policy="omit",
                alternative="two-sided",
            )
            if np.isfinite(result.pvalue):
                t_p_values.append(float(result.pvalue))
            elif np.isclose(np.nanmean(observed_by_case[:, ilag]), clim_mean[ilag]):
                t_p_values.append(1.0)
            else:
                t_p_values.append(0.0)

        summary[spec["t_p_col"]] = t_p_values

    return summary, (start_year, end_year)


def candidate_dates_same_year_month(peak_date: pd.Timestamp) -> pd.DatetimeIndex:
    month_start = peak_date.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    return pd.date_range(month_start, month_end, freq="D")


def lag_profiles_for_candidates(
    candidates: pd.DatetimeIndex,
    metric_frame: pd.DataFrame,
) -> np.ndarray:
    profiles = []
    for date in candidates:
        target_dates = date + pd.to_timedelta(LAGS, unit="D")
        profiles.append(
            metric_frame.reindex(target_dates, fill_value=0.0)[
                [
                    "is_blocking",
                    "daily_intensity_gpm",
                    "active_event_duration_days",
                ]
            ].to_numpy(dtype=float)
        )
    return np.stack(profiles, axis=0)


def monte_carlo_null(
    lookup: pd.DataFrame,
    n_iter: int = N_MONTE_CARLO,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    metric_frame = build_metric_frame(lookup)

    # Shape after accumulation: iteration, lag, metric.
    null_metrics = np.zeros((n_iter, len(LAGS), 3), dtype=float)

    for peak_date in PEAK_DATES:
        candidates = candidate_dates_same_year_month(pd.Timestamp(peak_date))
        profiles = lag_profiles_for_candidates(candidates, metric_frame)
        sample_idx = rng.integers(0, len(candidates), size=n_iter)
        null_metrics += profiles[sample_idx]

    null_metrics /= len(PEAK_DATES)
    return null_metrics


def two_sided_mc_pvalue(observed: np.ndarray, null_values: np.ndarray) -> np.ndarray:
    null_mean = null_values.mean(axis=0)
    obs_distance = np.abs(observed - null_mean)
    null_distance = np.abs(null_values - null_mean)
    return ((null_distance >= obs_distance).sum(axis=0) + 1) / (len(null_values) + 1)


def add_significance(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    null_metrics = monte_carlo_null(lookup)

    null_summary = pd.DataFrame({"lag": LAGS})
    metric_order = [
        "blocking_frequency",
        "mean_intensity_all_gpm",
        "mean_active_duration_all_days",
    ]

    summary = summary.copy()

    for metric_idx, metric_name in enumerate(metric_order):
        spec = METRIC_SPECS[metric_name]
        null_values = null_metrics[:, :, metric_idx]
        observed = summary[metric_name].to_numpy(dtype=float)
        null_mean = null_values.mean(axis=0)

        null_summary[f"{metric_name}_null_mean"] = null_mean
        null_summary[f"{metric_name}_null_p025"] = np.percentile(null_values, 2.5, axis=0)
        null_summary[f"{metric_name}_null_p975"] = np.percentile(null_values, 97.5, axis=0)
        null_summary[f"{metric_name}_null_p05"] = np.percentile(null_values, 5, axis=0)
        null_summary[f"{metric_name}_null_p95"] = np.percentile(null_values, 95, axis=0)

        summary[spec["null_mean_col"]] = null_mean
        summary[spec["mc_p_col"]] = two_sided_mc_pvalue(observed, null_values)

        observed_by_case = (
            samples.pivot(index="case_id", columns="lag", values=spec["sample_col"])
            .reindex(columns=LAGS)
            .to_numpy(dtype=float)
        )
        t_p_values = []
        for ilag in range(len(LAGS)):
            result = stats.ttest_1samp(
                observed_by_case[:, ilag],
                popmean=float(null_mean[ilag]),
                nan_policy="omit",
                alternative="two-sided",
            )
            t_p_values.append(float(result.pvalue) if np.isfinite(result.pvalue) else np.nan)

        summary[spec["t_p_col"]] = t_p_values

    return summary, null_summary


def add_bootstrap_stability(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    summary = summary.copy()
    metric_order = [
        "blocking_frequency",
        "mean_intensity_all_gpm",
        "mean_active_duration_all_days",
    ]

    for metric_name in metric_order:
        spec = METRIC_SPECS[metric_name]
        values_by_case = (
            samples.pivot(index="case_id", columns="lag", values=spec["sample_col"])
            .reindex(columns=LAGS)
            .to_numpy(dtype=float)
        )
        n_cases = values_by_case.shape[0]
        sample_idx = rng.integers(0, n_cases, size=(n_bootstrap, n_cases))
        bootstrap_values = values_by_case[sample_idx].mean(axis=1)

        prefix = f"bootstrap_{metric_name}"
        summary[f"{prefix}_mean"] = bootstrap_values.mean(axis=0)
        summary[f"{prefix}_p025"] = np.percentile(bootstrap_values, 2.5, axis=0)
        summary[f"{prefix}_p975"] = np.percentile(bootstrap_values, 97.5, axis=0)
        summary[f"{prefix}_p05"] = np.percentile(bootstrap_values, 5, axis=0)
        summary[f"{prefix}_p95"] = np.percentile(bootstrap_values, 95, axis=0)

        if metric_name == "blocking_frequency":
            count_prefix = "bootstrap_n_blocking"
            summary[f"{count_prefix}_mean"] = summary[f"{prefix}_mean"] * n_cases
            summary[f"{count_prefix}_p025"] = summary[f"{prefix}_p025"] * n_cases
            summary[f"{count_prefix}_p975"] = summary[f"{prefix}_p975"] * n_cases
            summary[f"{count_prefix}_p05"] = summary[f"{prefix}_p05"] * n_cases
            summary[f"{count_prefix}_p95"] = summary[f"{prefix}_p95"] * n_cases
            anom_prefix = "bootstrap_blocking_count_anom"
            clim = summary["clim_n_blocking_expected"].to_numpy(dtype=float)
            source_prefix = count_prefix
        else:
            anom_prefix = f"bootstrap_{metric_name}_anom"
            clim = summary[spec["clim_mean_col"]].to_numpy(dtype=float)
            source_prefix = prefix

        summary[f"{anom_prefix}_mean"] = (
            summary[f"{source_prefix}_mean"].to_numpy(dtype=float) - clim
        )
        summary[f"{anom_prefix}_p025"] = (
            summary[f"{source_prefix}_p025"].to_numpy(dtype=float) - clim
        )
        summary[f"{anom_prefix}_p975"] = (
            summary[f"{source_prefix}_p975"].to_numpy(dtype=float) - clim
        )
        summary[f"{anom_prefix}_p05"] = (
            summary[f"{source_prefix}_p05"].to_numpy(dtype=float) - clim
        )
        summary[f"{anom_prefix}_p95"] = (
            summary[f"{source_prefix}_p95"].to_numpy(dtype=float) - clim
        )

        summary[spec["bootstrap_sig_p005_col"]] = (
            (summary[f"{anom_prefix}_p025"].to_numpy(dtype=float) > 0)
            | (summary[f"{anom_prefix}_p975"].to_numpy(dtype=float) < 0)
        )
        summary[spec["bootstrap_sig_p010_col"]] = (
            (summary[f"{anom_prefix}_p05"].to_numpy(dtype=float) > 0)
            | (summary[f"{anom_prefix}_p95"].to_numpy(dtype=float) < 0)
        )

    return summary


def add_anomaly_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["blocking_count_anom"] = (
        summary["n_blocking"] - summary["clim_n_blocking_expected"]
    )
    summary["intensity_anom_gpm"] = (
        summary["mean_intensity_all_gpm"] - summary["clim_mean_intensity_all_gpm"]
    )
    summary["duration_anom_days"] = (
        summary["mean_active_duration_all_days"]
        - summary["clim_mean_active_duration_all_days"]
    )
    summary["blocking_count_anom_5d"] = (
        summary["blocking_count_anom"].rolling(5, center=True, min_periods=1).mean()
    )
    summary["intensity_anom_5d_gpm"] = (
        summary["intensity_anom_gpm"].rolling(5, center=True, min_periods=1).mean()
    )
    summary["duration_anom_5d_days"] = (
        summary["duration_anom_days"].rolling(5, center=True, min_periods=1).mean()
    )
    return summary


def add_bootstrap_dots(
    ax: plt.Axes,
    summary: pd.DataFrame,
    sig_col: str,
    y_col: str,
    alpha: float,
    y_scale: float = 1.0,
) -> None:
    sig = summary[sig_col].astype(bool)
    if not sig.any():
        return
    ax.scatter(
        summary.loc[sig, "lag"],
        summary.loc[sig, y_col] * y_scale,
        s=24,
        color="black",
        edgecolor="white",
        linewidth=0.35,
        zorder=9,
    )


def set_anomaly_ylim(ax: plt.Axes, *series: pd.Series) -> None:
    values = np.concatenate([pd.Series(s).dropna().to_numpy(dtype=float) for s in series])
    ymin = min(float(values.min()), 0.0)
    ymax = max(float(values.max()), 0.0)
    span = ymax - ymin
    pad = span * 0.12 if span > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)


def draw_figure(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    significance_alpha: float | None = None,
    suffix: str = "",
) -> tuple[Path, Path]:
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

    heat = samples.pivot(index="case_id", columns="lag", values="daily_intensity_gpm")
    block = samples.pivot(index="case_id", columns="lag", values="is_blocking").astype(bool)
    heat_masked = np.ma.masked_where(~block.values, heat.values)

    peak_labels = [
        f"{i:02d}  {d.strftime('%Y-%m-%d')}"
        for i, d in enumerate(PEAK_DATES, start=1)
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
        hspace=0.36,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("white")
    vmax = np.nanpercentile(heat.values[block.values], 95)
    im = ax0.imshow(
        heat_masked,
        aspect="auto",
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=vmax),
        extent=[LAGS.min() - 0.5, LAGS.max() + 0.5, len(PEAK_DATES) + 0.5, 0.5],
        interpolation="nearest",
    )
    ax0.axvline(0, color="black", lw=1.2)
    ax0.set_xlim(-30.5, 30.5)
    ax0.set_xticks(np.arange(-30, 31, 10))
    ax0.set_yticks(np.arange(1, len(PEAK_DATES) + 1))
    ax0.set_yticklabels(peak_labels)
    ax0.set_xlabel("Lag relative to MHW peak day (days)")
    ax0.set_ylabel("MHW peak date")
    ax0.set_title("(a) Alaska blocking occurrence and daily intensity", loc="left", fontweight="bold")
    ax0.grid(axis="x", color="0.88", lw=0.7)
    cbar = fig.colorbar(im, ax=ax0, pad=0.014, fraction=0.045)
    cbar.set_label("Daily AB intensity (gpm)")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(
        summary["lag"],
        summary["blocking_count_anom"],
        color="#355C9A",
        lw=1.2,
        alpha=0.55,
        label="Daily anomaly",
    )
    ax1.plot(
        summary["lag"],
        summary["blocking_count_anom_5d"],
        color="#12355B",
        lw=2.6,
        label="5-day running mean anomaly",
    )
    ax1.axvline(0, color="black", lw=1)
    ax1.axhline(0, color="0.55", lw=1, ls="--")
    ax1.set_xlim(-30, 30)
    set_anomaly_ylim(ax1, summary["blocking_count_anom"], summary["blocking_count_anom_5d"])
    ax1.set_xlabel("Lag relative to MHW peak day (days)")
    ax1.set_ylabel("Blocking occurrence anomaly (count)")
    ax1.set_title("(b) Frequency", loc="left", fontweight="bold")
    ax1.grid(color="0.9", lw=0.8)
    if significance_alpha is not None:
        sig_key = (
            "bootstrap_sig_p005_col"
            if significance_alpha <= 0.05
            else "bootstrap_sig_p010_col"
        )
        add_bootstrap_dots(
            ax1,
            summary,
            METRIC_SPECS["blocking_frequency"][sig_key],
            "blocking_count_anom",
            significance_alpha,
        )
    ax1.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="upper right",
        fontsize=9,
    )
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        summary["lag"],
        summary["intensity_anom_gpm"],
        color="#C84630",
        lw=1.2,
        alpha=0.55,
        label="Daily anomaly",
    )
    ax2.plot(
        summary["lag"],
        summary["intensity_anom_5d_gpm"],
        color="#8F1D14",
        lw=2.6,
        label="5-day running mean anomaly",
    )
    ax2.axvline(0, color="black", lw=1)
    ax2.axhline(0, color="0.55", lw=1, ls="--")
    ax2.set_xlim(-30, 30)
    set_anomaly_ylim(ax2, summary["intensity_anom_gpm"], summary["intensity_anom_5d_gpm"])
    ax2.set_xlabel("Lag relative to MHW peak day (days)")
    ax2.set_ylabel("Daily intensity anomaly (gpm)")
    ax2.set_title("(c) Intensity", loc="left", fontweight="bold")
    ax2.grid(color="0.9", lw=0.8)
    if significance_alpha is not None:
        sig_key = (
            "bootstrap_sig_p005_col"
            if significance_alpha <= 0.05
            else "bootstrap_sig_p010_col"
        )
        add_bootstrap_dots(
            ax2,
            summary,
            METRIC_SPECS["mean_intensity_all_gpm"][sig_key],
            "intensity_anom_gpm",
            significance_alpha,
        )
    ax2.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="upper right",
        fontsize=9,
    )
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(
        summary["lag"],
        summary["duration_anom_days"],
        color="#83A88B",
        lw=1.2,
        alpha=0.55,
        label="Daily anomaly",
    )
    ax3.plot(
        summary["lag"],
        summary["duration_anom_5d_days"],
        color="#2F7D5F",
        lw=2.6,
        label="5-day running mean anomaly",
    )
    ax3.axvline(0, color="black", lw=1)
    ax3.axhline(0, color="0.55", lw=1, ls="--")
    ax3.set_xlim(-30, 30)
    set_anomaly_ylim(ax3, summary["duration_anom_days"], summary["duration_anom_5d_days"])
    ax3.set_xlabel("Lag relative to MHW peak day (days)")
    ax3.set_ylabel("Active AB event duration anomaly (days)")
    ax3.set_title("(d) Persistence", loc="left", fontweight="bold")
    ax3.grid(color="0.9", lw=0.8)
    if significance_alpha is not None:
        sig_key = (
            "bootstrap_sig_p005_col"
            if significance_alpha <= 0.05
            else "bootstrap_sig_p010_col"
        )
        add_bootstrap_dots(
            ax3,
            summary,
            METRIC_SPECS["mean_active_duration_all_days"][sig_key],
            "duration_anom_days",
            significance_alpha,
        )
    ax3.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        loc="upper right",
        fontsize=9,
    )
    if significance_alpha is not None:
        fig.text(
            0.965,
            0.025,
            f"Dots: case bootstrap CI excludes zero anomaly (p < {significance_alpha:.2f})",
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.25",
        )

    fig.suptitle(
        "Alaska blocking evolution from 30 days before to 30 days after 14 early-winter MHW peaks",
        fontsize=13,
        fontweight="bold",
    )

    out_png = OUT_DIR / f"Figure_AB_features_14MHW_peak_lag30_ONDJF{suffix}.png"
    out_pdf = OUT_DIR / f"Figure_AB_features_14MHW_peak_lag30_ONDJF{suffix}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    events = pd.read_csv(EVENT_FILE)
    daily = pd.read_csv(DAILY_INTENSITY_FILE)

    lookup = build_daily_lookup(events, daily)
    samples = build_lag_samples(lookup)
    summary = summarize(samples)
    summary_with_mc, null_summary = add_significance(samples, summary, lookup)
    summary_with_clim, clim_period = add_climatology_baseline_and_ttest(
        samples,
        summary_with_mc,
        lookup,
    )
    summary_with_clim_bootstrap = add_anomaly_metrics(
        add_bootstrap_stability(samples, summary_with_clim)
    )

    samples.to_csv(OUT_DIR / "AB_MHW_peak_lag_samples_14cases_ONDJF.csv", index=False)
    summary.to_csv(OUT_DIR / "AB_MHW_peak_lag_summary_14cases_ONDJF.csv", index=False)
    summary_with_mc.to_csv(
        OUT_DIR / "AB_MHW_peak_lag_summary_14cases_ONDJF_with_significance.csv",
        index=False,
    )
    anomaly_export = summary_with_clim_bootstrap.drop(
        columns=[
            col
            for col in summary_with_clim_bootstrap.columns
            if col.startswith("ttest_p_")
            or col.startswith("mc_p_")
            or col.startswith("mc_null_mean_")
        ],
        errors="ignore",
    )
    anomaly_export.to_csv(
        OUT_DIR / "AB_MHW_peak_lag_summary_14cases_ONDJF_anomaly_bootstrap.csv",
        index=False,
    )
    null_summary.to_csv(
        OUT_DIR / "AB_MHW_peak_monte_carlo_null_summary_ONDJF.csv",
        index=False,
    )

    out_png, out_pdf = draw_figure(
        samples,
        summary_with_clim_bootstrap,
        suffix="_anomaly",
    )
    out_sig_p005_png, out_sig_p005_pdf = draw_figure(
        samples,
        summary_with_clim_bootstrap,
        significance_alpha=0.05,
        suffix="_anomaly_bootstrap_p005",
    )
    out_sig_p010_png, out_sig_p010_pdf = draw_figure(
        samples,
        summary_with_clim_bootstrap,
        significance_alpha=0.10,
        suffix="_anomaly_bootstrap_p010",
    )

    print("Saved figure PNG:", out_png)
    print("Saved figure PDF:", out_pdf)
    print("Saved p<0.05 anomaly bootstrap figure PNG:", out_sig_p005_png)
    print("Saved p<0.05 anomaly bootstrap figure PDF:", out_sig_p005_pdf)
    print("Saved p<0.10 anomaly bootstrap figure PNG:", out_sig_p010_png)
    print("Saved p<0.10 anomaly bootstrap figure PDF:", out_sig_p010_pdf)
    print("Saved lag samples:", OUT_DIR / "AB_MHW_peak_lag_samples_14cases_ONDJF.csv")
    print("Saved lag summary:", OUT_DIR / "AB_MHW_peak_lag_summary_14cases_ONDJF.csv")
    print(
        "Saved lag summary with significance:",
        OUT_DIR / "AB_MHW_peak_lag_summary_14cases_ONDJF_with_significance.csv",
    )
    print(
        "Saved lag summary with anomaly bootstrap:",
        OUT_DIR / "AB_MHW_peak_lag_summary_14cases_ONDJF_anomaly_bootstrap.csv",
    )
    print(
        "Saved Monte Carlo null summary:",
        OUT_DIR / "AB_MHW_peak_monte_carlo_null_summary_ONDJF.csv",
    )
    print()
    print("Peak dates:")
    for d in PEAK_DATES:
        print(" ", d.strftime("%Y-%m-%d"))
    print()
    print("Basic diagnostics:")
    print("  Total cases:", len(PEAK_DATES))
    print("  Lag range:", int(LAGS.min()), "to", int(LAGS.max()))
    print("  Max blocking occurrence count:", int(summary["n_blocking"].max()))
    print(
        "  Max 5-day mean occurrence count:",
        f"{summary['n_blocking_5d'].max():.2f}",
    )
    print(
        "  Max 5-day mean intensity:",
        f"{summary['mean_intensity_all_5d_gpm'].max():.1f} gpm",
    )
    print("  AB climatology period:", f"{clim_period[0]}-{clim_period[1]}")
    print("  Monte Carlo iterations:", N_MONTE_CARLO)
    print("  Bootstrap iterations:", N_BOOTSTRAP)
    print("  Monte Carlo random seed:", RANDOM_SEED)
    for metric_name, spec in METRIC_SPECS.items():
        p_col = spec["mc_p_col"]
        p05_count = int((summary_with_mc[p_col] < 0.05).sum())
        p10_count = int(
            ((summary_with_mc[p_col] >= 0.05) & (summary_with_mc[p_col] < 0.10)).sum()
        )
        print(f"  {metric_name}: MC p<0.05 lags={p05_count}, 0.05<=p<0.10 lags={p10_count}")
    for metric_name, spec in METRIC_SPECS.items():
        p005_count = int(summary_with_clim_bootstrap[spec["bootstrap_sig_p005_col"]].sum())
        p010_count = int(summary_with_clim_bootstrap[spec["bootstrap_sig_p010_col"]].sum())
        print(
            f"  {metric_name}: anomaly bootstrap p<0.05-equivalent "
            f"lags={p005_count}, p<0.10-equivalent lags={p010_count}"
        )


if __name__ == "__main__":
    main()
