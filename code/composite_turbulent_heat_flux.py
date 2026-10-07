import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import ttest_1samp


BASE_DIR = Path(__file__).resolve().parents[1]

FILE_MSSHF = BASE_DIR / "ERA5_msshf_anom_NOLEAPDOY_1981_2024_200E-225E_40N-50N_detrended_fixed.nc"
FILE_MSLHF = BASE_DIR / "ERA5_mslhf_anom_NOLEAPDOY_1981_2024_200E-225E_40N-50N_detrended_fixed.nc"

GROUPS = {
    "group1_2020in": {
        "title": "Group 1 peak in early winter (n=13)",
        "expected_n": 13,
        "dates": [
            "1985-11-29",
            "1986-11-19",
            "1989-11-18",
            "1989-12-24",
            "1991-11-10",
            "1993-11-26",
            "2015-11-06",
            "2015-11-30",
            "2018-11-19",
            "2019-11-10",
            "2020-11-13",
            "2023-11-03",
            "2023-12-14",
        ],
    },
    "group2_2004only": {
        "title": "Group 2 peak in early winter (n=1)",
        "expected_n": 1,
        "dates": [
            "2004-12-24",
        ],
    },
}

LAGS = np.arange(-10, 11, 1)
ALPHA = 0.05


def extract_event_matrix(series, peak_dates, lags):
    data = np.full((len(peak_dates), len(lags)), np.nan, dtype=float)

    for i, peak in enumerate(peak_dates):
        for j, lag in enumerate(lags):
            target_day = peak + pd.Timedelta(days=int(lag))
            if target_day in series.index:
                data[i, j] = series.loc[target_day]

    return data


def composite_and_ttest(mat, alpha=0.05):
    mean = np.nanmean(mat, axis=0)
    pvals = np.full(mat.shape[1], np.nan, dtype=float)
    sig = np.zeros(mat.shape[1], dtype=bool)
    n_valid = np.sum(~np.isnan(mat), axis=0)

    for j in range(mat.shape[1]):
        x = mat[:, j]
        x = x[np.isfinite(x)]
        if len(x) >= 2:
            _, p = ttest_1samp(x, popmean=0.0, alternative="two-sided")
            pvals[j] = p
            sig[j] = p < alpha

    return mean, pvals, sig, n_valid


def make_group_dataframe(ser_lh, ser_sh, ser_hf, peak_dates):
    mat_lh = extract_event_matrix(ser_lh, peak_dates, LAGS)
    mat_sh = extract_event_matrix(ser_sh, peak_dates, LAGS)
    mat_hf = extract_event_matrix(ser_hf, peak_dates, LAGS)

    mean_lh, p_lh, sig_lh, n_lh = composite_and_ttest(mat_lh, alpha=ALPHA)
    mean_sh, p_sh, sig_sh, n_sh = composite_and_ttest(mat_sh, alpha=ALPHA)
    mean_hf, p_hf, sig_hf, n_hf = composite_and_ttest(mat_hf, alpha=ALPHA)

    return pd.DataFrame(
        {
            "lag_day": LAGS,
            "lh_mean": mean_lh,
            "lh_p": p_lh,
            "lh_sig_p_lt_0.05": sig_lh,
            "lh_n": n_lh,
            "sh_mean": mean_sh,
            "sh_p": p_sh,
            "sh_sig_p_lt_0.05": sig_sh,
            "sh_n": n_sh,
            "heat_flux_mean": mean_hf,
            "heat_flux_p": p_hf,
            "heat_flux_sig_p_lt_0.05": sig_hf,
            "heat_flux_n": n_hf,
        }
    )


def plot_group(df, title, out_png):
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(10.2, 7.2))

    ax.axvspan(-10, 0, color="0.93", zorder=0)
    ax.axvspan(0, 10, color="#f7eaea", zorder=0)

    ax.axhline(0, linestyle="--", linewidth=1.8, zorder=1)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    line_lh, = ax.plot(
        df["lag_day"], df["lh_mean"],
        linewidth=2.8, color="#1f77b4", label="LH anomaly", zorder=3
    )
    line_sh, = ax.plot(
        df["lag_day"], df["sh_mean"],
        linewidth=2.8, color="#d67a00", label="SH anomaly", zorder=3
    )
    line_hf, = ax.plot(
        df["lag_day"], df["heat_flux_mean"],
        linewidth=2.8, color="#c43c2b", label="heat flux anomaly", zorder=3
    )

    for mean_col, sig_col, line in [
        ("lh_mean", "lh_sig_p_lt_0.05", line_lh),
        ("sh_mean", "sh_sig_p_lt_0.05", line_sh),
        ("heat_flux_mean", "heat_flux_sig_p_lt_0.05", line_hf),
    ]:
        sig = df[sig_col].to_numpy(dtype=bool)
        ax.scatter(
            df.loc[sig, "lag_day"],
            df.loc[sig, mean_col],
            s=58,
            color=line.get_color(),
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )

    xticks = np.arange(-10, 11, 2)
    xticklabels = []
    for x in xticks:
        if x == 0:
            xticklabels.append("peak")
        elif x > 0:
            xticklabels.append(f"+{x}d")
        else:
            xticklabels.append(f"{x}d")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=16)

    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(axis="y", labelsize=15)

    ax.set_title(title, loc="left", fontsize=24)
    ax.text(1.0, 1.01, "W/m$^2$", transform=ax.transAxes, ha="right", va="bottom", fontsize=24)
    ax.legend(loc="lower left", ncol=2, frameon=True, fontsize=13)

    y_all = np.concatenate(
        [
            df["lh_mean"].to_numpy(dtype=float),
            df["sh_mean"].to_numpy(dtype=float),
            df["heat_flux_mean"].to_numpy(dtype=float),
        ]
    )
    y_all = y_all[np.isfinite(y_all)]
    pad = 6
    ymin_plot = 10 * np.floor((np.nanmin(y_all) - pad) / 10.0)
    ymax_plot = 10 * np.ceil((np.nanmax(y_all) + pad) / 10.0)
    ax.set_ylim(ymin_plot, ymax_plot)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_dates(time_index, groups):
    missing = []
    for group_name, config in groups.items():
        dates = pd.to_datetime(config["dates"])
        if len(dates) != config["expected_n"]:
            raise ValueError(f"{group_name} sample size does not match date count.")
        for date in dates:
            if date not in time_index:
                missing.append(f"{group_name}: {date.date()}")

    if missing:
        raise ValueError("Missing event dates in anomaly data: " + ", ".join(missing))


def main():
    ds_sh = xr.open_dataset(FILE_MSSHF)
    ds_lh = xr.open_dataset(FILE_MSLHF)

    da_sh = ds_sh["msshf"].transpose("time", "lat", "lon")
    da_lh = ds_lh["mslhf"].transpose("time", "lat", "lon")

    time_index = pd.DatetimeIndex(da_sh["time"].values)
    validate_dates(time_index, GROUPS)

    weights = np.cos(np.deg2rad(da_sh["lat"]))
    weights.name = "weights"

    ts_sh = da_sh.weighted(weights).mean(dim=("lat", "lon"))
    ts_lh = da_lh.weighted(weights).mean(dim=("lat", "lon"))
    ts_hf = ts_lh + ts_sh
    ts_hf.name = "heat_flux"

    ser_sh = ts_sh.to_series()
    ser_lh = ts_lh.to_series()
    ser_hf = ts_hf.to_series()

    for group_name, config in GROUPS.items():
        peak_dates = pd.to_datetime(config["dates"])
        df = make_group_dataframe(ser_lh, ser_sh, ser_hf, peak_dates)

        out_csv = BASE_DIR / f"early_winter_peak_heatflux_{group_name}_p005.csv"
        out_png = BASE_DIR / f"early_winter_peak_heatflux_{group_name}_p005.png"

        if out_csv.exists() or out_png.exists():
            raise FileExistsError(f"Refusing to overwrite existing output for {group_name}.")

        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        plot_group(df, config["title"], out_png)

        print("=" * 80)
        print(f"{group_name}: n={len(peak_dates)}")
        print("CSV saved to   :", out_csv)
        print("Figure saved to:", out_png)
        print(df)

    ds_sh.close()
    ds_lh.close()


if __name__ == "__main__":
    main()
