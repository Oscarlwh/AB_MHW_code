#!/usr/bin/env python3
"""Plot two-panel ERA5 EGR maps for centered lag-2 five-day means."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

import compute_plot_era5_eady_lag0_25_center5day as base


DATA_FILE = base.DATA_FILE


def member_fields(
    ds: xr.Dataset,
    level: int,
    lag_window: list[int],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    events = ds["eady_event_anomaly"].sel(lag=lag_window, lev=level).mean("lag", skipna=True)
    n13_samples = events.isel(event=base.N13_INDICES)
    n13 = n13_samples.mean("event", skipna=True)
    case_2004 = events.isel(event=base.CASE_2004_INDEX)
    p_value = xr.DataArray(
        stats.ttest_1samp(n13_samples.values, popmean=0.0, axis=0, nan_policy="omit").pvalue.astype(np.float32),
        dims=("lat", "lon"),
        coords={"lat": n13_samples.lat.values, "lon": n13_samples.lon.values},
        name="p_value",
    )
    scale = base.SECONDS_PER_DAY * 100.0
    return n13 * scale, case_2004 * scale, p_value


def plot_level(
    ds: xr.Dataset,
    level: int,
    center_lag: int,
    lag_window: list[int],
    out_root: Path,
    rectangular_colorbar: bool = False,
) -> tuple[Path, list[dict[str, object]]]:
    n13, case_2004, p_value = member_fields(ds, level, lag_window)
    time_title = "peak day" if center_lag == 0 else f"lag {center_lag} days"
    fields = [
        (n13, f"(a) {time_title}", "N=13"),
        (case_2004, f"(b) {time_title}", "2004-12-24"),
    ]
    projection = ccrs.PlateCarree(central_longitude=180) if HAS_CARTOPY else None
    data_crs = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig = plt.figure(figsize=(10.0, 3.9))
    rows: list[dict[str, object]] = []

    for i, (source, left_title, right_title) in enumerate(fields):
        field = source.sel(
            lat=slice(base.DISPLAY_LAT_MIN, base.DISPLAY_LAT_MAX),
            lon=slice(base.DISPLAY_LON_MIN, base.DISPLAY_LON_MAX),
        )
        vmax = base.nice_vmax(float(np.nanpercentile(np.abs(field.values), 98.0)))
        levels = np.linspace(-vmax, vmax, 17)
        ax = fig.add_subplot(1, 2, i + 1, projection=projection) if HAS_CARTOPY else fig.add_subplot(1, 2, i + 1)
        lon = field.lon.values
        lat = field.lat.values
        if HAS_CARTOPY:
            ax.set_extent(
                [base.DISPLAY_LON_MIN, base.DISPLAY_LON_MAX, base.DISPLAY_LAT_MIN, base.DISPLAY_LAT_MAX],
                crs=data_crs,
            )
            plot_values = np.clip(field.values, -vmax, vmax) if rectangular_colorbar else field.values
            contour = ax.contourf(
                lon,
                lat,
                plot_values,
                levels=levels,
                cmap=base.paper_cmap(),
                extend="neither" if rectangular_colorbar else "both",
                transform=data_crs,
            )
            ax.coastlines(resolution="110m", linewidth=0.65, color="black", zorder=8)
            ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25, edgecolor="black", zorder=8)
            ax.set_xticks([150, 180, 210, 240], crs=data_crs)
            ax.set_yticks([20, 40, 60, 80], crs=data_crs)
            base.add_box(ax, data_crs)
        else:
            ax.set_xlim(base.DISPLAY_LON_MIN, base.DISPLAY_LON_MAX)
            ax.set_ylim(base.DISPLAY_LAT_MIN, base.DISPLAY_LAT_MAX)
            plot_values = np.clip(field.values, -vmax, vmax) if rectangular_colorbar else field.values
            contour = ax.contourf(
                lon,
                lat,
                plot_values,
                levels=levels,
                cmap=base.paper_cmap(),
                extend="neither" if rectangular_colorbar else "both",
            )
            ax.set_xticks([150, 180, 210, 240])
            ax.set_yticks([20, 40, 60, 80])
            base.add_box(ax)

        ax.set_xticklabels([base.lon_label(x) for x in [150, 180, 210, 240]], fontsize=8.5)
        ax.set_yticklabels([f"{y}°N" for y in [20, 40, 60, 80]] if i == 0 else [""] * 4, fontsize=8.5)
        xlabels = ax.get_xticklabels()
        if xlabels:
            xlabels[0].set_ha("left")
            xlabels[-1].set_ha("right")
        ax.tick_params(length=3, width=0.8, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.set_title(left_title, loc="left", fontsize=9.5, pad=3)
        ax.set_title("EGR", loc="center", fontsize=9.5, pad=3)
        ax.set_title(right_title, loc="right", fontsize=9.3, pad=3)

        significant_fraction = np.nan
        if i == 0:
            p_plot = p_value.sel(
                lat=slice(base.DISPLAY_LAT_MIN, base.DISPLAY_LAT_MAX),
                lon=slice(base.DISPLAY_LON_MIN, base.DISPLAY_LON_MAX),
            )
            significant = p_plot.values[::2, ::2] < 0.05
            yy, xx = np.meshgrid(lat[::2], lon[::2], indexing="ij")
            scatter_kwargs = {
                "s": 6.0,
                "color": "black",
                "marker": "o",
                "linewidths": 0,
                "zorder": 10,
            }
            if HAS_CARTOPY:
                scatter_kwargs["transform"] = data_crs
            ax.scatter(xx[significant], yy[significant], **scatter_kwargs)
            significant_fraction = float((p_plot.values < 0.05).mean())

        colorbar = fig.colorbar(
            contour,
            ax=ax,
            orientation="vertical",
            fraction=0.047,
            pad=0.040,
            ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax],
        )
        colorbar.ax.tick_params(labelsize=8.2, length=2.5, pad=2)
        colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        colorbar.outline.set_linewidth(0.7)
        rows.append(
            {
                "level_hPa": level,
                "member": right_title,
                "center_lag": center_lag,
                "window_lags": ",".join(str(lag) for lag in lag_window),
                "vmax_1e-2_day-1": vmax,
                "min_1e-2_day-1": float(np.nanmin(field.values)),
                "max_1e-2_day-1": float(np.nanmax(field.values)),
                "mean_1e-2_day-1": float(np.nanmean(field.values)),
                "nan_count": int(np.isnan(field.values).sum()),
                "inf_count": int(np.isinf(field.values).sum()),
                "significant_fraction_p005": significant_fraction,
            }
        )

    fig.text(0.5, 0.055, rf"EGR ($10^{{-2}}$ day$^{{-1}}$), {level} hPa", ha="center", va="center", fontsize=9.2)
    fig.subplots_adjust(left=0.055, right=0.950, bottom=0.16, top=0.91, wspace=0.17)
    out_root.mkdir(parents=True, exist_ok=True)
    time_tag = "peak" if center_lag == 0 else f"lag{center_lag}"
    colorbar_tag = "_rectangular_colorbar" if rectangular_colorbar else ""
    output = out_root / (
        f"ERA5_EGR_{level}hPa_N13_and_2004_{time_tag}_center5day_two_panel"
        f"{colorbar_tag}.png"
    )
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"[FIG] {output}", flush=True)
    return output, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-lag", type=int, default=2, help="Center lag for the five-day mean")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=base.PLOT_LEVELS,
        help="Pressure levels to plot (default: all project levels)",
    )
    parser.add_argument(
        "--rectangular-colorbar",
        action="store_true",
        help="Use clipped rectangular colorbars without pointed extensions",
    )
    args = parser.parse_args()
    center_lag = int(args.center_lag)
    plot_levels = [int(level) for level in args.levels]
    invalid_levels = [level for level in plot_levels if level not in base.PLOT_LEVELS]
    if invalid_levels:
        raise ValueError(f"Unsupported pressure levels: {invalid_levels}")
    lag_window = list(range(center_lag - 2, center_lag + 3))
    time_tag = "peak" if center_lag == 0 else f"lag{center_lag}"
    colorbar_dir_tag = "_rectangular_colorbar" if args.rectangular_colorbar else ""
    out_root = base.OUT_ROOT / f"figures_{time_tag}_center5day_two_panel{colorbar_dir_tag}"
    check_file = out_root / f"ERA5_EGR_{time_tag}_center5day_two_panel_check.csv"
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)
    with xr.open_dataset(DATA_FILE) as source:
        ds = source.load()
    missing_lags = [lag for lag in lag_window if lag not in ds.lag.values]
    if missing_lags:
        ds.close()
        raise KeyError(f"Missing lags required by centered five-day mean: {missing_lags}")
    paths = []
    records = []
    for level in plot_levels:
        path, rows = plot_level(
            ds,
            level,
            center_lag,
            lag_window,
            out_root,
            rectangular_colorbar=args.rectangular_colorbar,
        )
        paths.append(path)
        records.extend(rows)
    ds.close()
    pd.DataFrame(records).to_csv(check_file, index=False, encoding="utf-8-sig")
    if len(paths) != len(plot_levels):
        raise RuntimeError(f"Expected {len(plot_levels)} figures, generated {len(paths)}")
    print(f"[CHECK] {check_file}", flush=True)
    print(f"[DONE] Generated {len(paths)} two-panel {time_tag} EGR figures", flush=True)


if __name__ == "__main__":
    main()
