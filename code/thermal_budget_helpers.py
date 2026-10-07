#!/usr/bin/env python3
"""Compute the four-term CFS mixed-layer temperature budget for all events."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import xarray as xr


DEFAULT_ROOT = Path("/path/to/data/热收支分析/CFS_four_term")
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "outputs_v1"

GROUPS = {
    "cfs_n13_four_term": {
        "title": "n=13",
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
    "cfs_20041224_four_term": {
        "title": "2004-12-24",
        "dates": ["2004-12-24"],
    },
}

SOURCE_CONFIG = {
    "cfsr": {
        "season_years": range(1980, 2011),
        "climatology_start": 1981,
        "climatology_end": 2010,
    },
    "cfsv2": {
        "season_years": range(2011, 2024),
        "climatology_start": 2011,
        "climatology_end": 2020,
    },
}

ANALYSIS_BOUNDS = {"lat_min": 40.0, "lat_max": 50.0, "lon_min": 200.0, "lon_max": 225.0}
MAX_MLD_M = 300.0
RHO0 = 1025.0
CP0 = 3990.0
EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_DAY = 86_400.0
ALPHA = 0.05
LAGS = np.arange(-10, 11, dtype=int)
TERMS = ("tendency", "surface", "hadv", "residual")

PLOT_STYLE = {
    "tendency": {
        "label": r"$\partial T/\partial t$",
        "color": "#202020",
        "linestyle": "-",
        "linewidth": 2.8,
    },
    "surface": {
        "label": r"$Q_{\mathrm{net}}/(\rho_0 C_p h)$",
        "color": "#c85c2b",
        "linestyle": "--",
        "linewidth": 2.3,
    },
    "hadv": {
        "label": r"$-\mathbf{u}\cdot\nabla T$",
        "color": "#2b6ea6",
        "linestyle": "-.",
        "linewidth": 2.3,
    },
    "residual": {
        "label": r"$R$",
        "color": "#65762f",
        "linestyle": ":",
        "linewidth": 2.6,
    },
}

VARIABLE_MATCHERS = {
    "theta": ("potential temperature", "potential_temperature", "pot"),
    "u": ("u-component of current", "u_component_of_current", "uogrd"),
    "v": ("v-component of current", "v_component_of_current", "vogrd"),
    "mld": ("bottom of ocean mixed layer", "ocean mixed layer", "obml", "dbss"),
    "qnet": ("total downward heat flux", "total_downward_heat_flux", "thflx"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot the CFSR/CFSv2 four-term mixed-layer temperature budget."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-inputs", action="store_true")
    return parser.parse_args()


def candidate_netcdf_files(directory: Path) -> list[Path]:
    suffixes = {".nc", ".nc4", ".cdf", ".netcdf"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and path.stat().st_size > 0
        and path.suffix.lower() in suffixes
    )


def season_directory(data_root: Path, source: str, season_year: int) -> Path:
    return data_root / source / f"{season_year}_{season_year + 1}"


def verify_input_tree(data_root: Path) -> dict[tuple[str, int], list[Path]]:
    files_by_season = {}
    missing = []
    for source, config in SOURCE_CONFIG.items():
        for season_year in config["season_years"]:
            directory = season_directory(data_root, source, season_year)
            files = candidate_netcdf_files(directory) if directory.is_dir() else []
            if not files:
                missing.append(str(directory))
            else:
                files_by_season[(source, season_year)] = files
    if missing:
        preview = "\n  - ".join(missing[:12])
        raise FileNotFoundError(
            f"Missing CFS NetCDF data for {len(missing)} cold seasons:\n  - "
            + preview
            + ("\n  - ..." if len(missing) > 12 else "")
        )
    return files_by_season


def variable_text(name: str, da: xr.DataArray) -> str:
    attrs = " ".join(
        str(da.attrs.get(key, ""))
        for key in (
            "long_name",
            "standard_name",
            "GRIB2_Parameter_Name",
            "Grib2_Parameter_Name",
            "abbreviation",
            "description",
        )
    )
    return f"{name} {attrs}".lower().replace("-", "_").replace(" ", "_")


def find_variable(ds: xr.Dataset, physical_name: str) -> xr.DataArray:
    matchers = tuple(item.lower().replace("-", "_").replace(" ", "_") for item in VARIABLE_MATCHERS[physical_name])
    candidates = []
    for name, da in ds.data_vars.items():
        text = variable_text(name, da)
        score = sum(matcher in text for matcher in matchers)
        if physical_name == "mld":
            score += 3 * int("mixed_layer" in text or "obml" in text)
            score -= 4 * int("isothermal" in text or "isotherm" in text)
        if score > 0:
            candidates.append((score, name, da))
    if not candidates:
        raise KeyError(
            f"Could not find {physical_name!r}; variables were {list(ds.data_vars)}"
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def coordinate_kind(name: str, coord: xr.DataArray) -> str | None:
    text = " ".join(
        [
            name,
            str(coord.attrs.get("standard_name", "")),
            str(coord.attrs.get("long_name", "")),
            str(coord.attrs.get("axis", "")),
            str(coord.attrs.get("units", "")),
        ]
    ).lower()
    if "time" in text or np.issubdtype(coord.dtype, np.datetime64):
        return "time"
    if "latitude" in text or name.lower() in {"lat", "y"}:
        return "lat"
    if "longitude" in text or name.lower() in {"lon", "x"}:
        return "lon"
    if "depth below sea" in text or name.lower() in {"depth", "depth_below_sea"}:
        return "depth"
    return None


def standardize_dataarray(da: xr.DataArray, needs_depth: bool) -> xr.DataArray:
    rename = {}
    for dim in da.dims:
        coord = da.coords.get(dim)
        if coord is None:
            continue
        kind = coordinate_kind(dim, coord)
        if kind and kind not in rename.values() and dim != kind:
            rename[dim] = kind
    if rename:
        da = da.rename(rename)

    required = {"time", "lat", "lon"} | ({"depth"} if needs_depth else set())
    missing = required - set(da.dims)
    if missing:
        raise ValueError(f"Variable {da.name!r} is missing dimensions {sorted(missing)}: {da.dims}")

    extra = [dim for dim in da.dims if dim not in required]
    for dim in extra:
        if da.sizes[dim] != 1:
            raise ValueError(f"Unexpected non-singleton dimension {dim!r} in {da.name!r}")
        da = da.isel({dim: 0}, drop=True)

    da = da.assign_coords(lon=np.mod(da["lon"].astype(float), 360.0)).sortby("lon")
    da = da.sortby("lat").sortby("time")
    if needs_depth:
        da = da.assign_coords(depth=np.abs(da["depth"].astype(float))).sortby("depth")
        da = da.sel(depth=slice(0.0, 303.0))
    order = ("time", "depth", "lat", "lon") if needs_depth else ("time", "lat", "lon")
    return da.transpose(*order).astype("float64")


def open_season(files: Iterable[Path]) -> tuple[xr.Dataset, xr.Dataset]:
    raw = xr.open_mfdataset(
        [str(path) for path in files],
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
        parallel=False,
    )
    fields = xr.Dataset(
        {
            "theta": standardize_dataarray(find_variable(raw, "theta"), needs_depth=True),
            "u": standardize_dataarray(find_variable(raw, "u"), needs_depth=True),
            "v": standardize_dataarray(find_variable(raw, "v"), needs_depth=True),
            "mld": standardize_dataarray(find_variable(raw, "mld"), needs_depth=False),
            "qnet": standardize_dataarray(find_variable(raw, "qnet"), needs_depth=False),
        }
    )
    return fields, raw


def verify_units(ds: xr.Dataset) -> None:
    expected = {
        "u": ("m/s", "m s-1", "m s**-1", "m s^-1"),
        "v": ("m/s", "m s-1", "m s**-1", "m s^-1"),
        "mld": ("m", "meter", "metre"),
        "qnet": ("w/m2", "w/m^2", "w m-2", "w m**-2", "w m^-2"),
    }
    for name, accepted in expected.items():
        units = str(ds[name].attrs.get("units", "")).lower().strip()
        compact = units.replace(" ", "")
        if not units:
            raise ValueError(f"CFS variable {name!r} has no units attribute")
        if not any(
            token in units or token.replace(" ", "") in compact
            for token in accepted
        ):
            raise ValueError(
                f"Unexpected units for CFS variable {name!r}: {units!r}; "
                f"expected one of {accepted}"
            )


def target_season_bounds(source: str, season_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    target_start = pd.Timestamp(year=season_year, month=10, day=23)
    if source == "cfsr" and season_year == 2010:
        target_end = pd.Timestamp("2010-12-31")
    else:
        target_end = pd.Timestamp(year=season_year + 1, month=1, day=4)
    return target_start, target_end


def verify_six_hourly_and_daily_mean(
    ds: xr.Dataset,
    source: str,
    season_year: int,
) -> xr.Dataset:
    index = pd.DatetimeIndex(ds["time"].values)
    if index.has_duplicates:
        keep = ~index.duplicated(keep="first")
        ds = ds.isel(time=np.flatnonzero(keep))
        index = pd.DatetimeIndex(ds["time"].values)
    counts = pd.Series(1, index=index).resample("1D").sum()
    target_start, target_end = target_season_bounds(source, season_year)
    target_counts = counts.loc[target_start:target_end]
    expected_dates = pd.date_range(target_start, target_end, freq="1D")
    target_counts = target_counts.reindex(expected_dates, fill_value=0)
    bad = target_counts[target_counts != 4]
    if not bad.empty:
        preview = ", ".join(
            f"{date.date()}={int(count)}" for date, count in bad.iloc[:8].items()
        )
        raise ValueError(f"Expected four CFS samples per day; found {preview}")
    daily = ds.resample(time="1D").mean(keep_attrs=True)
    return daily.sel(time=slice(target_start, target_end))


def depth_edges(depth_values: np.ndarray) -> np.ndarray:
    depth_values = np.asarray(depth_values, dtype=float)
    if np.any(np.diff(depth_values) <= 0):
        raise ValueError("CFS depth levels must be strictly increasing")
    edges = np.empty(depth_values.size + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (depth_values[:-1] + depth_values[1:])
    edges[-1] = depth_values[-1] + 0.5 * (depth_values[-1] - depth_values[-2])
    return edges


def mixed_layer_thickness(depth: xr.DataArray, mld: xr.DataArray) -> xr.DataArray:
    edges = depth_edges(depth.values)
    top = xr.DataArray(edges[:-1], dims="depth", coords={"depth": depth})
    bottom = xr.DataArray(edges[1:], dims="depth", coords={"depth": depth})
    return (xr.where(mld < bottom, mld, bottom) - top).clip(min=0.0)


def mixed_layer_average(da: xr.DataArray, thickness: xr.DataArray) -> xr.DataArray:
    valid = thickness.where(da.notnull())
    denominator = valid.sum("depth", skipna=True)
    numerator = (da * valid).sum("depth", skipna=True)
    return (numerator / denominator).where(denominator > 0.0)


def centered_daily_derivative(da: xr.DataArray) -> xr.DataArray:
    seconds = da["time"].values.astype("datetime64[s]").astype("int64").astype(float)
    seconds = xr.DataArray(seconds, dims="time", coords={"time": da["time"]})
    denominator = seconds.shift(time=-1) - seconds.shift(time=1)
    derivative = (da.shift(time=-1) - da.shift(time=1)) / denominator
    return derivative.where((denominator > 0.0) & (denominator <= 3.0 * SECONDS_PER_DAY))


def area_mean(da: xr.DataArray, ocean_mask: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da["lat"]))
    weights.name = "cos_latitude"
    return da.where(ocean_mask).weighted(weights).mean(("lat", "lon"), skipna=True)


def calculate_season_regional(
    files: list[Path],
    source: str,
    season_year: int,
) -> tuple[xr.Dataset, float, int]:
    fields, raw = open_season(files)
    verify_units(fields)
    daily = verify_six_hourly_and_daily_mean(fields, source, season_year)
    daily = daily.sel(
        lat=slice(ANALYSIS_BOUNDS["lat_min"], ANALYSIS_BOUNDS["lat_max"]),
        lon=slice(ANALYSIS_BOUNDS["lon_min"], ANALYSIS_BOUNDS["lon_max"]),
    )
    if daily.sizes.get("lat", 0) < 3 or daily.sizes.get("lon", 0) < 3:
        raise ValueError(f"CFS data do not cover the analysis region {ANALYSIS_BOUNDS}")

    mld = daily["mld"].where(daily["mld"] > 0.0)
    max_mld = float(mld.max(skipna=True).compute())
    if max_mld > MAX_MLD_M:
        raise ValueError(
            f"Maximum CFS MLD is {max_mld:.2f} m, exceeding the allowed {MAX_MLD_M:.0f} m"
        )

    thickness = mixed_layer_thickness(daily["depth"], mld)
    tm = mixed_layer_average(daily["theta"], thickness)
    um = mixed_layer_average(daily["u"], thickness)
    vm = mixed_layer_average(daily["v"], thickness)

    tendency = centered_daily_derivative(tm) * SECONDS_PER_DAY
    surface = daily["qnet"] / (RHO0 * CP0 * mld) * SECONDS_PER_DAY
    lat_radians = np.deg2rad(tm["lat"])
    meters_per_lon_degree = EARTH_RADIUS_M * np.cos(lat_radians) * np.pi / 180.0
    meters_per_lat_degree = EARTH_RADIUS_M * np.pi / 180.0
    d_t_dx = tm.differentiate("lon") / meters_per_lon_degree
    d_t_dy = tm.differentiate("lat") / meters_per_lat_degree
    hadv = -(um * d_t_dx + vm * d_t_dy) * SECONDS_PER_DAY
    residual = tendency - surface - hadv

    ocean_mask = (
        tm.notnull()
        & mld.notnull()
        & daily["qnet"].notnull()
        & um.notnull()
        & vm.notnull()
    ).all("time").compute()
    ocean_cells = int(ocean_mask.sum())
    if ocean_cells == 0:
        raise ValueError("The common CFS ocean mask is empty")

    result = xr.Dataset(
        {
            "tendency": area_mean(tendency, ocean_mask),
            "surface": area_mean(surface, ocean_mask),
            "hadv": area_mean(hadv, ocean_mask),
            "residual": area_mean(residual, ocean_mask),
            "mld": area_mean(mld, ocean_mask),
            "tm": area_mean(tm, ocean_mask),
        }
    ).compute()
    result["closure_error"] = (
        result["tendency"] - result["surface"] - result["hadv"] - result["residual"]
    )
    result = result.dropna("time", how="any")
    result.attrs.update(
        {
            "rho0_kg_m-3": RHO0,
            "cp0_J_kg-1_K-1": CP0,
            "units_budget_terms": "degree_C day-1",
            "max_mld_m": max_mld,
            "ocean_cells": ocean_cells,
            "qnet_sign": "positive downward into ocean",
        }
    )
    raw.close()
    return result, max_mld, ocean_cells


def write_new_netcdf(dataset: xr.Dataset, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite NetCDF: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary NetCDF already exists: {temporary}")
    dataset.to_netcdf(temporary)
    if path.exists():
        raise FileExistsError(f"NetCDF target appeared during write: {path}")
    temporary.rename(path)


def load_or_calculate_regional(
    data_root: Path,
    output_dir: Path,
    files_by_season: dict[tuple[str, int], list[Path]],
) -> dict[str, xr.Dataset]:
    source_datasets = {}
    for source, config in SOURCE_CONFIG.items():
        pieces = []
        for index, season_year in enumerate(config["season_years"], start=1):
            cache = (
                output_dir
                / "cache"
                / "seasonal"
                / source
                / f"{source}_{season_year}_{season_year + 1}_regional.nc"
            )
            if cache.exists():
                with xr.open_dataset(cache) as cached:
                    piece = cached.load()
                print(f"Loaded cache: {cache}", flush=True)
            else:
                print(
                    f"Calculating {source} {season_year}/{season_year + 1} "
                    f"({index}/{len(config['season_years'])})",
                    flush=True,
                )
                piece, _, _ = calculate_season_regional(
                    files_by_season[(source, season_year)],
                    source,
                    season_year,
                )
                write_new_netcdf(piece, cache)
                print(f"Saved cache: {cache}", flush=True)
            pieces.append(piece)
        source_datasets[source] = xr.concat(pieces, dim="time").sortby("time")
    return source_datasets


def mean_preserving_detrend(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    valid = np.isfinite(values)
    dates = pd.DatetimeIndex(series.index)
    x = (dates - dates[0]).total_seconds().to_numpy(dtype=float) / SECONDS_PER_DAY
    slope, intercept = np.polyfit(x[valid], values[valid], 1)
    detrended = values - (slope * x + intercept) + np.nanmean(values[valid])
    return pd.Series(detrended, index=series.index, name=series.name)


def month_day_key(index: pd.DatetimeIndex) -> np.ndarray:
    return index.month.to_numpy() * 100 + index.day.to_numpy()


def source_anomalies(source: str, regional: xr.Dataset) -> dict[str, pd.Series]:
    config = SOURCE_CONFIG[source]
    index = pd.DatetimeIndex(regional["time"].values)
    baseline = (
        (index.year >= config["climatology_start"])
        & (index.year <= config["climatology_end"])
    )
    keys = month_day_key(index)
    anomalies = {}
    for term in TERMS:
        series = mean_preserving_detrend(
            pd.Series(regional[term].values, index=index, name=term)
        )
        climatology = pd.Series(series.to_numpy()[baseline]).groupby(keys[baseline]).mean()
        mapped = pd.Series(keys, index=index).map(climatology).to_numpy(dtype=float)
        anomalies[term] = pd.Series(
            series.to_numpy(dtype=float) - mapped,
            index=index,
            name=term,
        )
    return anomalies


def event_source(date: pd.Timestamp) -> str:
    return "cfsr" if date < pd.Timestamp("2011-04-01") else "cfsv2"


def validate_event_windows(anomalies: dict[str, dict[str, pd.Series]]) -> None:
    missing = []
    for config in GROUPS.values():
        for text in config["dates"]:
            peak = pd.Timestamp(text)
            source = event_source(peak)
            for lag in LAGS:
                date = peak + pd.Timedelta(days=int(lag))
                for term in TERMS:
                    series = anomalies[source][term]
                    if date not in series.index or not np.isfinite(series.loc[date]):
                        missing.append(f"{source}:{date.date()}:{term}")
    if missing:
        raise ValueError(
            "Event windows are incomplete:\n  - " + "\n  - ".join(sorted(set(missing)))
        )


def event_matrix(
    anomalies: dict[str, dict[str, pd.Series]],
    term: str,
    peak_dates: pd.DatetimeIndex,
) -> np.ndarray:
    matrix = np.full((len(peak_dates), len(LAGS)), np.nan)
    for row, peak in enumerate(peak_dates):
        series = anomalies[event_source(peak)][term]
        for col, lag in enumerate(LAGS):
            matrix[row, col] = series.loc[peak + pd.Timedelta(days=int(lag))]
    return matrix


def summarize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(matrix, axis=0)
    counts = np.sum(np.isfinite(matrix), axis=0)
    p_values = np.full(matrix.shape[1], np.nan)
    significant = np.zeros(matrix.shape[1], dtype=bool)
    if matrix.shape[0] >= 2:
        for column in range(matrix.shape[1]):
            values = matrix[:, column]
            values = values[np.isfinite(values)]
            if values.size >= 2 and not np.allclose(values, values[0]):
                _, p_value = ttest_1samp(values, popmean=0.0, alternative="two-sided")
                p_values[column] = p_value
                significant[column] = p_value < ALPHA
    return means, counts, p_values, significant


def make_group_frame(
    anomalies: dict[str, dict[str, pd.Series]],
    peak_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = pd.DataFrame({"relative_day": LAGS})
    for term in TERMS:
        matrix = event_matrix(anomalies, term, peak_dates)
        means, counts, p_values, significant = summarize_matrix(matrix)
        frame[f"{term}_mean"] = means
        frame[f"{term}_n"] = counts
        frame[f"{term}_p"] = p_values
        frame[f"{term}_significant_p005"] = significant
    frame["closure_error"] = (
        frame["tendency_mean"]
        - frame["surface_mean"]
        - frame["hadv_mean"]
        - frame["residual_mean"]
    )
    return frame


def plot_group(
    frame: pd.DataFrame,
    title: str,
    output: Path,
    show_significance: bool,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite PNG: {output}")
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    ax.axvspan(-10, 0, color="#e5e7e9", alpha=0.72, zorder=0)
    ax.axvspan(0, 10, color="#f8dedd", alpha=0.62, zorder=0)
    ax.axhline(0, color="#777777", linewidth=1.0, zorder=1)
    ax.axvline(0, color="#555555", linewidth=1.2, linestyle="--", zorder=1)

    for term in TERMS:
        style = PLOT_STYLE[term]
        values = frame[f"{term}_mean"].to_numpy()
        ax.plot(
            LAGS,
            values,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=3,
        )
        if show_significance:
            significant = frame[f"{term}_significant_p005"].to_numpy(dtype=bool)
            ax.scatter(
                LAGS[significant],
                values[significant],
                s=38,
                facecolors="white",
                edgecolors=style["color"],
                linewidths=1.5,
                zorder=5,
            )

    ax.set_xlim(-10, 10)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels(["Lead 10", "Lead 5", "Peak", "Lag 5", "Lag 10"])
    ax.set_ylabel(r"Temperature tendency ($^\circ$C day$^{-1}$)")
    ax.set_title(title, fontsize=16, pad=12)
    ax.grid(axis="y", color="#bfc3c5", linewidth=0.7, alpha=0.55)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=True,
        framealpha=0.96,
        ncol=4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def output_paths(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    outputs = {}
    for name, config in GROUPS.items():
        suffix = "_p005" if len(config["dates"]) >= 2 else ""
        csv_path = output_dir / f"early_winter_{name}{suffix}.csv"
        png_path = output_dir / f"early_winter_{name}{suffix}.png"
        if csv_path.exists() or png_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite output for {name}: {csv_path} or {png_path}"
            )
        outputs[name] = (csv_path, png_path)
    return outputs


def main() -> None:
    args = parse_args()
    files_by_season = verify_input_tree(args.data_root)
    if args.check_inputs:
        print(f"All {len(files_by_season)} CFS cold-season directories contain NetCDF files.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(args.output_dir)
    regional = load_or_calculate_regional(args.data_root, args.output_dir, files_by_season)
    anomalies = {
        source: source_anomalies(source, dataset)
        for source, dataset in regional.items()
    }
    validate_event_windows(anomalies)

    max_daily_closure = max(
        float(np.nanmax(np.abs(dataset["closure_error"].values)))
        for dataset in regional.values()
    )
    if max_daily_closure > 1e-12:
        raise AssertionError(
            f"Daily four-term closure failed: {max_daily_closure:.3e} degree_C day-1"
        )

    for name, config in GROUPS.items():
        peak_dates = pd.DatetimeIndex(pd.to_datetime(config["dates"]))
        frame = make_group_frame(anomalies, peak_dates)
        if len(frame) != 21 or not np.array_equal(frame["relative_day"], LAGS):
            raise AssertionError(f"Unexpected output rows for {name}")
        if len(peak_dates) >= 2 and not all(
            np.array_equal(frame[f"{term}_n"].to_numpy(), np.full(21, len(peak_dates)))
            for term in TERMS
        ):
            raise AssertionError(f"Unexpected event count for {name}")
        composite_closure = float(np.nanmax(np.abs(frame["closure_error"])))
        if composite_closure > 1e-12:
            raise AssertionError(
                f"Composite closure failed for {name}: {composite_closure:.3e}"
            )

        csv_path, png_path = outputs[name]
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot_group(
            frame,
            title=config["title"],
            output=png_path,
            show_significance=len(peak_dates) >= 2,
        )
        print(f"Saved CSV: {csv_path}")
        print(f"Saved PNG: {png_path}")

    print(f"Maximum daily closure error: {max_daily_closure:.3e} degree_C day-1")
    print("CFSR and CFSv2 were detrended and climatologized separately across the 2011 transition.")


if __name__ == "__main__":
    main()
