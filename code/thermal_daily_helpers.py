#!/usr/bin/env python3
"""Plot OISST tendency, ERA5 surface heating, and an inclusive residual."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "thermal_budget_helpers.py"
DEFAULT_OUTPUT_DIR = Path("/path/to/data/热收支分析/CFS_four_term/outputs_v1")
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_DIR / "cache/seasonal"
DEFAULT_OISST = Path("/path/to/data/MHW/NEP_MHW_daily_1981-2024_area_mean.nc")
DEFAULT_ERA5_ANNUAL = Path("/path/to/data/ERA5")
DEFAULT_ERA5_RADIATION = Path(
    "/path/to/data/热收支分析/ERA5_budget_flux_daily"
)
DEFAULT_ERA5_TURBULENT = Path("/path/to/data/draft/1")

RHO0 = 1025.0
CP0 = 3990.0
SECONDS_PER_DAY = 86_400.0
LAGS = np.arange(-10, 11, dtype=int)
TERMS = ("tendency", "surface", "residual")

PLOT_STYLE = {
    "tendency": {
        "label": r"$\Delta_+\mathrm{SST}'_{\mathrm{OISST}}/\Delta t$",
        "color": "#B33A3A",
        "linestyle": "-",
        "linewidth": 2.65,
    },
    "surface": {
        "label": r"$[Q_{\mathrm{net}}^{\mathrm{ERA5}}/(\rho_0 C_p h)]'$",
        "color": "#315A7D",
        "linestyle": (0, (6.0, 2.2)),
        "linewidth": 2.25,
    },
    "residual": {
        "label": r"$R^\ast$ (including advection)",
        "color": "#5A5148",
        "linestyle": (0, (1.2, 1.8)),
        "linewidth": 2.45,
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("cfs_budget_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot OISST SST-anomaly tendency, ERA5 net surface heating "
            "converted with CFS MLD, and the inclusive residual."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--oisst", type=Path, default=DEFAULT_OISST)
    parser.add_argument("--era5-annual", type=Path, default=DEFAULT_ERA5_ANNUAL)
    parser.add_argument(
        "--era5-radiation",
        type=Path,
        default=DEFAULT_ERA5_RADIATION,
    )
    parser.add_argument(
        "--era5-turbulent",
        type=Path,
        default=DEFAULT_ERA5_TURBULENT,
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.1,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.fontsize": 9.4,
            "savefig.facecolor": "white",
        }
    )


def output_path(output_dir: Path) -> Path:
    path = output_dir / (
        "early_winter_oisst_era5_qnet_three_term_anomaly_"
        "n13_20041224_ab_publication_v1.png"
    )
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite PNG: {path}")
    return path


def standardize_era5(array: xr.DataArray) -> xr.DataArray:
    rename = {}
    for target, candidates in {
        "time": ("time", "valid_time"),
        "lat": ("lat", "latitude"),
        "lon": ("lon", "longitude"),
    }.items():
        for candidate in candidates:
            if candidate in array.dims or candidate in array.coords:
                if candidate != target:
                    rename[candidate] = target
                break
    if rename:
        array = array.rename(rename)
    missing = {"time", "lat", "lon"} - set(array.dims)
    if missing:
        raise ValueError(f"{array.name!r} is missing dimensions {sorted(missing)}")
    array = array.assign_coords(lon=np.mod(array["lon"].astype(float), 360.0))
    array = array.sortby("time").sortby("lat").sortby("lon")
    array = array.sel(
        lat=slice(BASE.ANALYSIS_BOUNDS["lat_min"], BASE.ANALYSIS_BOUNDS["lat_max"]),
        lon=slice(BASE.ANALYSIS_BOUNDS["lon_min"], BASE.ANALYSIS_BOUNDS["lon_max"]),
    )
    if array.sizes.get("lat", 0) < 2 or array.sizes.get("lon", 0) < 2:
        raise ValueError(f"{array.name!r} does not cover the analysis region")
    return array.squeeze(drop=True)


def check_flux_units(array: xr.DataArray) -> None:
    units = str(array.attrs.get("units", "")).lower().replace(" ", "")
    if "w" not in units or "m" not in units:
        raise ValueError(
            f"Expected W m-2 for {array.name}, found {array.attrs.get('units')!r}"
        )


def load_early_radiation(
    annual_dir: Path,
    variable: str,
) -> xr.DataArray:
    pieces = []
    for year in range(1981, 1993):
        path = annual_dir / f"{year}.nc"
        if not path.is_file():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as dataset:
            if variable not in dataset:
                raise KeyError(f"{path} is missing {variable}")
            array = standardize_era5(dataset[variable])
            check_flux_units(array)
            pieces.append(array.astype("float32").load())
    return xr.concat(
        pieces,
        dim="time",
        coords="minimal",
        compat="override",
        join="exact",
    ).sortby("time")


def load_compact_flux(path: Path, variable: str) -> xr.DataArray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as dataset:
        if variable not in dataset:
            raise KeyError(f"{path} is missing {variable}")
        array = standardize_era5(dataset[variable])
        check_flux_units(array)
        return array.astype("float32").load()


def concatenate_without_duplicates(
    first: xr.DataArray,
    second: xr.DataArray,
) -> xr.DataArray:
    target_lat = first["lat"]
    target_lon = first["lon"]
    if not (
        np.array_equal(second["lat"].values, target_lat.values)
        and np.array_equal(second["lon"].values, target_lon.values)
    ):
        second = second.interp(lat=target_lat, lon=target_lon)
    combined = xr.concat(
        [first, second],
        dim="time",
        coords="minimal",
        compat="override",
        join="exact",
    ).sortby("time")
    index = pd.DatetimeIndex(combined["time"].values)
    keep = ~index.duplicated(keep="first")
    return combined.isel(time=np.flatnonzero(keep))


def cosine_area_mean(array: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(array["lat"]))
    return array.weighted(weights).mean(("lat", "lon"), skipna=True)


def load_era5_qnet(args: argparse.Namespace) -> pd.Series:
    radiation_paths = {
        "msnswrf": args.era5_radiation
        / "ERA5_msnswrf_daily_1993_2023_200E-225E_40N-50N.nc",
        "msnlwrf": args.era5_radiation
        / "ERA5_msnlwrf_daily_1993_2023_200E-225E_40N-50N.nc",
    }
    turbulent_paths = {
        "msshf": args.era5_turbulent
        / "ERA5_msshf_daily_1x1_200E-225E_40N-50N_1981_2024_merged.nc",
        "mslhf": args.era5_turbulent
        / "ERA5_mslhf_daily_1x1_200E-225E_40N-50N_1981_2024_merged.nc",
    }

    components = {}
    for variable, path in radiation_paths.items():
        early = load_early_radiation(args.era5_annual, variable)
        later = load_compact_flux(path, variable)
        components[variable] = concatenate_without_duplicates(early, later)
    for variable, path in turbulent_paths.items():
        components[variable] = load_compact_flux(path, variable)

    sw, lw, sh, lh = xr.align(
        components["msnswrf"],
        components["msnlwrf"],
        components["msshf"],
        components["mslhf"],
        join="inner",
    )
    qnet = (sw + lw + sh + lh).rename("qnet")
    qnet_mean = cosine_area_mean(qnet).load()
    index = pd.DatetimeIndex(qnet_mean["time"].values)
    series = pd.Series(
        qnet_mean.values.astype(float),
        index=index,
        name="era5_qnet_region_mean",
    ).sort_index()
    if not series.index.is_unique:
        raise ValueError("ERA5 Qnet time index contains duplicates")
    return series


def seasonal_cache_path(
    cache_root: Path,
    source: str,
    season_year: int,
) -> Path:
    return (
        cache_root
        / source
        / f"{source}_{season_year}_{season_year + 1}_regional.nc"
    )


def load_cfs_mld(cache_root: Path) -> dict[str, pd.Series]:
    output = {}
    for source, config in BASE.SOURCE_CONFIG.items():
        pieces = []
        for season_year in config["season_years"]:
            path = seasonal_cache_path(cache_root, source, season_year)
            if not path.is_file():
                raise FileNotFoundError(f"Missing CFS MLD cache: {path}")
            with xr.open_dataset(path) as dataset:
                if "mld" not in dataset:
                    raise KeyError(f"{path} is missing mld")
                pieces.append(dataset["mld"].load())
        combined = xr.concat(pieces, dim="time").sortby("time")
        index = pd.DatetimeIndex(combined["time"].values)
        series = pd.Series(
            combined.values.astype(float),
            index=index,
            name="cfs_region_mean_mld",
        )
        if not series.index.is_unique:
            raise ValueError(f"{source} MLD time index contains duplicates")
        if not np.isfinite(series).all() or (series <= 0.0).any():
            raise ValueError(f"{source} MLD contains invalid values")
        if float(series.max()) > BASE.MAX_MLD_M:
            raise ValueError(
                f"{source} MLD exceeds {BASE.MAX_MLD_M:g} m: "
                f"{float(series.max()):.2f} m"
            )
        output[source] = series
    return output


def source_anomaly(source: str, series: pd.Series) -> pd.Series:
    config = BASE.SOURCE_CONFIG[source]
    detrended = BASE.mean_preserving_detrend(series)
    index = pd.DatetimeIndex(detrended.index)
    baseline = (
        (index.year >= config["climatology_start"])
        & (index.year <= config["climatology_end"])
    )
    keys = BASE.month_day_key(index)
    climatology = (
        pd.Series(detrended.to_numpy(dtype=float)[baseline])
        .groupby(keys[baseline])
        .mean()
    )
    mapped = pd.Series(keys, index=index).map(climatology).to_numpy(dtype=float)
    return pd.Series(
        detrended.to_numpy(dtype=float) - mapped,
        index=index,
        name=series.name,
    )


def build_surface_anomalies(
    qnet: pd.Series,
    mld: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    output = {}
    for source, depth in mld.items():
        aligned_qnet = qnet.reindex(depth.index)
        raw = (
            aligned_qnet
            / (RHO0 * CP0 * depth)
            * SECONDS_PER_DAY
        ).rename("surface")
        output[source] = source_anomaly(source, raw)
    return output


def load_oisst_forward_tendency(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as dataset:
        if "sst_anomaly" not in dataset:
            raise KeyError(f"{path} is missing sst_anomaly")
        region = str(dataset.attrs.get("region", ""))
        if region and ("40-50N" not in region or "135-160W" not in region):
            raise ValueError(f"Unexpected OISST region: {region!r}")
        anomaly = dataset["sst_anomaly"].load()
    index = pd.DatetimeIndex(anomaly["time"].values)
    series = pd.Series(
        anomaly.values.astype(float),
        index=index,
        name="oisst_sst_anomaly",
    ).sort_index()
    tendency = series.shift(-1) - series
    consecutive = (
        np.diff(series.index.values).astype("timedelta64[D]").astype(int) == 1
    )
    valid = np.concatenate((consecutive, [False]))
    tendency.iloc[~valid] = np.nan
    return tendency.rename("tendency")


def assemble_terms(
    tendency: pd.Series,
    surface: dict[str, pd.Series],
) -> dict[str, dict[str, pd.Series]]:
    anomalies = {}
    for source, surface_series in surface.items():
        aligned_tendency = tendency.reindex(surface_series.index)
        residual = (aligned_tendency - surface_series).rename("residual")
        anomalies[source] = {
            "tendency": aligned_tendency.rename("tendency"),
            "surface": surface_series.rename("surface"),
            "residual": residual,
        }
    return anomalies


def validate_event_windows(
    anomalies: dict[str, dict[str, pd.Series]],
) -> None:
    missing = []
    for config in BASE.GROUPS.values():
        for text in config["dates"]:
            peak = pd.Timestamp(text)
            source = BASE.event_source(peak)
            for lag in LAGS:
                date = peak + pd.Timedelta(days=int(lag))
                for term in TERMS:
                    series = anomalies[source][term]
                    if date not in series.index or not np.isfinite(series.loc[date]):
                        missing.append(f"{source}:{date.date()}:{term}")
    if missing:
        raise ValueError(
            "Three-term event windows are incomplete:\n  - "
            + "\n  - ".join(sorted(set(missing)))
        )


def make_group_frame(
    anomalies: dict[str, dict[str, pd.Series]],
    peak_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = pd.DataFrame({"relative_day": LAGS})
    for term in TERMS:
        matrix = BASE.event_matrix(anomalies, term, peak_dates)
        means, counts, p_values, significant = BASE.summarize_matrix(matrix)
        frame[f"{term}_mean"] = means
        frame[f"{term}_n"] = counts
        frame[f"{term}_p"] = p_values
        frame[f"{term}_significant_p005"] = significant
    frame["closure_error"] = (
        frame["tendency_mean"]
        - frame["surface_mean"]
        - frame["residual_mean"]
    )
    return frame


def verify_group_frame(
    frame: pd.DataFrame,
    name: str,
    expected_count: int,
) -> None:
    if len(frame) != 21 or not np.array_equal(frame["relative_day"], LAGS):
        raise AssertionError(f"Unexpected rows for {name}")
    for term in TERMS:
        counts = frame[f"{term}_n"].to_numpy(dtype=int)
        if not np.array_equal(counts, np.full(21, expected_count)):
            raise AssertionError(f"Unexpected {term} counts for {name}: {counts}")
    closure = float(np.nanmax(np.abs(frame["closure_error"])))
    if closure > 1e-12:
        raise AssertionError(f"Three-term closure failed for {name}: {closure:.3e}")


def style_axis(axis: plt.Axes, panel_label: str) -> None:
    axis.axvspan(-10, 0, color="#EEF1F3", zorder=0)
    axis.axvspan(0, 10, color="#FAEEEE", zorder=0)
    axis.axhline(0, color="#6F7478", linewidth=0.8, zorder=1)
    axis.axvline(
        0,
        color="#4D5154",
        linewidth=1.0,
        linestyle=(0, (4.0, 3.0)),
        zorder=2,
    )
    axis.set_xlim(-10, 10)
    axis.margins(x=0)
    axis.set_xticks([-10, -5, 0, 5, 10])
    axis.set_xticklabels(["Lead 10", "Lead 5", "Peak", "Lag 5", "Lag 10"])
    axis.grid(axis="y", color="#B9BEC2", linewidth=0.5, alpha=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#242729")
    axis.spines["bottom"].set_color("#242729")
    axis.tick_params(colors="#242729", direction="out")
    axis.set_title(
        panel_label,
        loc="left",
        pad=8,
        fontsize=11.2,
        fontweight="semibold",
        color="#242729",
    )


def plot_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    show_significance: bool,
) -> None:
    for term, style in PLOT_STYLE.items():
        values = frame[f"{term}_mean"].to_numpy(dtype=float)
        axis.plot(
            LAGS,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=3,
        )
        if show_significance:
            significant = frame[
                f"{term}_significant_p005"
            ].to_numpy(dtype=bool)
            axis.scatter(
                LAGS[significant],
                values[significant],
                s=23,
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.65,
                zorder=5,
            )


def plot_figure(
    n13: pd.DataFrame,
    single: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    style_axis(axes[0], "(a)")
    style_axis(axes[1], "(b)")
    plot_panel(axes[0], n13, show_significance=True)
    plot_panel(axes[1], single, show_significance=False)

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
    axes[0].set_ylabel(r"Temperature-budget anomaly ($^\circ$C day$^{-1}$)")

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        for style in PLOT_STYLE.values()
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
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = output_path(args.output_dir)

    qnet = load_era5_qnet(args)
    mld = load_cfs_mld(args.cache_root)
    surface = build_surface_anomalies(qnet, mld)
    tendency = load_oisst_forward_tendency(args.oisst)
    anomalies = assemble_terms(tendency, surface)
    validate_event_windows(anomalies)

    n13_dates = pd.DatetimeIndex(
        pd.to_datetime(BASE.GROUPS["cfs_n13_four_term"]["dates"])
    )
    single_dates = pd.DatetimeIndex(
        pd.to_datetime(BASE.GROUPS["cfs_20041224_four_term"]["dates"])
    )
    n13 = make_group_frame(anomalies, n13_dates)
    single = make_group_frame(anomalies, single_dates)
    verify_group_frame(n13, "n13", 13)
    verify_group_frame(single, "2004", 1)
    plot_figure(n13, single, output)

    max_closure = max(
        float(np.nanmax(np.abs(n13["closure_error"]))),
        float(np.nanmax(np.abs(single["closure_error"]))),
    )
    counts = {
        term: int(n13[f"{term}_significant_p005"].sum())
        for term in TERMS
    }
    peak_window = single.loc[
        single["relative_day"].between(-1, 3),
        [
            "relative_day",
            "tendency_mean",
            "surface_mean",
            "residual_mean",
        ],
    ]
    print(
        "ERA5 Qnet coverage used: "
        f"{qnet.index.min().date()} to {qnet.index.max().date()}"
    )
    print(
        "Maximum CFS regional-mean MLD: "
        f"{max(float(series.max()) for series in mld.values()):.2f} m"
    )
    print(f"Maximum composite closure error: {max_closure:.3e} degree_C day-1")
    print(f"n=13 significant-point counts: {counts}")
    print("2004 three-term anomalies near peak:")
    print(peak_window.to_string(index=False))
    print(f"Saved PNG: {output}")


if __name__ == "__main__":
    main()
