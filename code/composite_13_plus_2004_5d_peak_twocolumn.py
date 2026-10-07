import os
import string
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ttest_1samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths and events
# ============================================================
ANOM_FILE = (
    "/path/to/data/ABH_170E240E_1981_2024_5N90N/anomaly/"
    "ERA5_Z500_anom_NOLEAP_1981_2024_vs1981_2020_gpm_170E240E_5N90N_detrended.nc"
)

OUT_DIR = (
    "/path/to/data/ABH_170E240E_1981_2024_5N90N/composite/"
    "composite_13_plus_2004_5d_peak_twocolumn"
)

OUTPUT_STEM = (
    "Z500_anomaly_5dmean_group13_vs_2004_peak_lead15_lag15_"
    "170E240E_20N75N_p005"
)

ALL_PEAK_DATES = pd.to_datetime(
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
).normalize()

SPECIAL_PEAK_DATE = pd.Timestamp("2004-12-24")
GROUP13_PEAK_DATES = ALL_PEAK_DATES[ALL_PEAK_DATES != SPECIAL_PEAK_DATE]


# ============================================================
# 2. Plot and analysis settings
# ============================================================
LON_MIN = 170
LON_MAX = 240
LAT_MIN = 20
LAT_MAX = 75

RELATIVE_DAYS = np.array([-15, -10, -5, 0, 5, 10, 15], dtype=np.int16)
ROLL_HALF_WIDTH = 2
ALPHA = 0.05

VMIN = -120
VMAX = 120
FILL_LEVELS = np.arange(VMIN, VMAX + 10, 10)
LINE_LEVELS = np.arange(VMIN, VMAX + 20, 20)
LINE_LEVELS = LINE_LEVELS[LINE_LEVELS != 0]

DOT_STRIDE_LAT = 2
DOT_STRIDE_LON = 2
DOT_SIZE = 8
DOT_ALPHA = 0.95

PANEL_TITLES = [
    "lead 15 days",
    "lead 10 days",
    "lead 5 days",
    "peak",
    "lag 5 days",
    "lag 10 days",
    "lag 15 days",
]


def date_label(value):
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def lon_labels():
    return ["180°", "160°W", "140°W", "120°W"]


def target_window(peak_date, relative_day):
    center = pd.to_datetime(peak_date) + pd.to_timedelta(int(relative_day), unit="D")
    dates = pd.date_range(
        center - pd.Timedelta(days=ROLL_HALF_WIDTH),
        center + pd.Timedelta(days=ROLL_HALF_WIDTH),
        freq="D",
    )
    return center, dates


def require_dates(dates, time_set, context):
    missing = [date_label(value) for value in dates if pd.to_datetime(value) not in time_set]
    if missing:
        raise ValueError(f"Missing dates for {context}: {', '.join(missing)}")


def compute_event_fields(z_region, time_set, peak_date):
    fields = []
    centers = []
    window_starts = []
    window_ends = []

    for relative_day in RELATIVE_DAYS:
        center, dates = target_window(peak_date, relative_day)
        context = f"peak={date_label(peak_date)}, relative_day={int(relative_day):+d}"
        require_dates(dates, time_set, context)

        field = z_region.sel(time=dates).mean(dim="time", skipna=True)
        if int(field.isnull().sum().values) != 0:
            raise ValueError(f"Unexpected NaN in 5-day mean for {context}")

        fields.append(field)
        centers.append(center)
        window_starts.append(dates[0])
        window_ends.append(dates[-1])

    data = xr.concat(fields, dim="relative_day")
    data = data.assign_coords(relative_day=RELATIVE_DAYS)

    return (
        data,
        np.array(centers, dtype="datetime64[ns]"),
        np.array(window_starts, dtype="datetime64[ns]"),
        np.array(window_ends, dtype="datetime64[ns]"),
    )


def compute_output_dataset(z_region, time_set):
    group_fields = []
    group_centers = []
    group_window_starts = []
    group_window_ends = []

    for peak_date in GROUP13_PEAK_DATES:
        fields, centers, starts, ends = compute_event_fields(z_region, time_set, peak_date)
        group_fields.append(fields)
        group_centers.append(centers)
        group_window_starts.append(starts)
        group_window_ends.append(ends)

    samples = xr.concat(group_fields, dim="case")
    samples = samples.assign_coords(
        case=np.arange(1, len(GROUP13_PEAK_DATES) + 1, dtype=np.int16),
        case_peak_date=("case", GROUP13_PEAK_DATES.to_numpy(dtype="datetime64[ns]")),
    )

    group_composite = samples.mean(dim="case", skipna=True)
    group_composite.name = "z500_anomaly_5dmean_group13"

    _, p_values = ttest_1samp(samples.values, popmean=0.0, axis=0, nan_policy="omit")
    p_value = xr.DataArray(
        p_values,
        coords={
            "relative_day": RELATIVE_DAYS,
            "lat": z_region["lat"],
            "lon": z_region["lon"],
        },
        dims=("relative_day", "lat", "lon"),
        name="p_value",
    )
    significant = (p_value < ALPHA).astype("int8")
    significant.name = "significant_p005"

    sample_count = xr.DataArray(
        np.full(len(RELATIVE_DAYS), len(GROUP13_PEAK_DATES), dtype=np.int16),
        coords={"relative_day": RELATIVE_DAYS},
        dims=("relative_day",),
        name="sample_count",
    )

    special_fields, special_centers, special_starts, special_ends = compute_event_fields(
        z_region,
        time_set,
        SPECIAL_PEAK_DATE,
    )
    special_fields.name = "z500_anomaly_5dmean_special2004"

    out_ds = xr.Dataset(
        {
            "z500_anomaly_5dmean_group13": group_composite.astype("float32"),
            "z500_anomaly_5dmean_special2004": special_fields.astype("float32"),
            "p_value": p_value.astype("float32"),
            "significant_p005": significant,
            "sample_count": sample_count,
        }
    )

    out_ds = out_ds.assign_coords(
        group13_peak_date=("case", GROUP13_PEAK_DATES.to_numpy(dtype="datetime64[ns]")),
        group13_target_date=(
            ("case", "relative_day"),
            np.array(group_centers, dtype="datetime64[ns]"),
        ),
        group13_window_start=(
            ("case", "relative_day"),
            np.array(group_window_starts, dtype="datetime64[ns]"),
        ),
        group13_window_end=(
            ("case", "relative_day"),
            np.array(group_window_ends, dtype="datetime64[ns]"),
        ),
        special2004_peak_date=np.datetime64(SPECIAL_PEAK_DATE, "ns"),
        special2004_target_date=("relative_day", special_centers),
        special2004_window_start=("relative_day", special_starts),
        special2004_window_end=("relative_day", special_ends),
    )

    out_ds.attrs.update(
        {
            "description": "Peak-centered Z500 anomaly comparison: 13-event composite and 2004 special case",
            "anomaly_file": ANOM_FILE,
            "domain": "170E-240E, 20N-75N",
            "relative_days": ", ".join(map(str, RELATIVE_DAYS)),
            "running_mean": "centered 5-day mean, target date +/- 2 days",
            "group13_definition": "All 14 peak events except 2004-12-24; includes 2020-11-13",
            "special_case": date_label(SPECIAL_PEAK_DATE),
            "significance_test": "One-sample Student t-test against zero for group13 only",
            "alpha": str(ALPHA),
            "climatology_period": "1981-2020",
            "detrended": "True",
        }
    )

    return out_ds


def style_map_axis(ax, row_index, col_index, data_crs):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=data_crs)
    ax.coastlines(resolution="50m", linewidth=0.85)
    ax.add_feature(cfeature.BORDERS, linewidth=0.30, alpha=0.45)
    ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.25)

    ax.set_xticks([180, 200, 220, 240], crs=data_crs)
    ax.set_yticks([20, 30, 40, 50, 60, 70], crs=data_crs)
    if col_index == 0:
        ax.set_yticklabels(
            ["20°N", "30°N", "40°N", "50°N", "60°N", "70°N"],
            fontsize=9,
        )
    else:
        ax.set_yticklabels([])

    if row_index == len(RELATIVE_DAYS) - 1:
        ax.set_xticklabels(lon_labels(), fontsize=9)
    else:
        ax.set_xticklabels([])

    ax.gridlines(
        crs=data_crs,
        linewidth=0.32,
        color="gray",
        alpha=0.32,
        linestyle="--",
        draw_labels=False,
    )

    panel_index = row_index * 2 + col_index
    ax.text(
        0.01,
        1.025,
        f"({string.ascii_lowercase[panel_index]}) {PANEL_TITLES[row_index]}",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_comparison(out_ds, out_png, out_pdf):
    map_proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10.4, 18.5))
    grid = fig.add_gridspec(
        nrows=7,
        ncols=2,
        width_ratios=[1.0, 1.0],
        left=0.055,
        right=0.900,
        bottom=0.052,
        top=0.910,
        wspace=0.055,
        hspace=0.135,
    )

    group_data = out_ds["z500_anomaly_5dmean_group13"]
    special_data = out_ds["z500_anomaly_5dmean_special2004"]
    significance = out_ds["significant_p005"]
    lats = out_ds["lat"].values
    lons = out_ds["lon"].values

    contour_fill = None
    axes = np.empty((len(RELATIVE_DAYS), 2), dtype=object)

    for row_index, relative_day in enumerate(RELATIVE_DAYS):
        for col_index, field_all in enumerate([group_data, special_data]):
            ax = fig.add_subplot(grid[row_index, col_index], projection=map_proj)
            axes[row_index, col_index] = ax
            field = field_all.sel(relative_day=relative_day)

            contour_fill = ax.contourf(
                lons,
                lats,
                field.values,
                levels=FILL_LEVELS,
                cmap="RdBu_r",
                extend="both",
                transform=data_crs,
            )

            contours = ax.contour(
                lons,
                lats,
                field.values,
                levels=LINE_LEVELS,
                colors="black",
                linewidths=0.42,
                alpha=0.55,
                transform=data_crs,
            )
            ax.contour(
                lons,
                lats,
                field.values,
                levels=[0],
                colors="black",
                linewidths=0.80,
                alpha=0.75,
                transform=data_crs,
            )
            ax.clabel(contours, inline=True, fontsize=5.7, fmt="%d", inline_spacing=2)

            if col_index == 0:
                sig_now = significance.sel(relative_day=relative_day)
                sig_sub = sig_now.values[::DOT_STRIDE_LAT, ::DOT_STRIDE_LON].astype(bool)
                lat_sub = lats[::DOT_STRIDE_LAT]
                lon_sub = lons[::DOT_STRIDE_LON]
                xx, yy = np.meshgrid(lon_sub, lat_sub)
                ax.scatter(
                    xx[sig_sub],
                    yy[sig_sub],
                    s=DOT_SIZE,
                    c="black",
                    marker=".",
                    alpha=DOT_ALPHA,
                    linewidths=0,
                    transform=data_crs,
                    zorder=6,
                )

            style_map_axis(ax, row_index, col_index, data_crs)
            ax.set_anchor("E" if col_index == 0 else "W")

            if row_index == 0:
                annotation = "N = 13" if col_index == 0 else "2004-12-24"
                ax.text(
                    0.99,
                    1.025,
                    annotation,
                    transform=ax.transAxes,
                    fontsize=10.5,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                )

    fig.canvas.draw()
    right_top_pos = axes[0, 1].get_position()
    right_bottom_pos = axes[-1, 1].get_position()

    cbar_x0 = right_top_pos.x1 + 0.022
    cbar_y0 = right_bottom_pos.y0
    cbar_height = right_top_pos.y1 - right_bottom_pos.y0
    cbar_ax = fig.add_axes([cbar_x0, cbar_y0, 0.027, cbar_height])
    colorbar = fig.colorbar(
        contour_fill,
        cax=cbar_ax,
        orientation="vertical",
        ticks=[-120, -60, 0, 60, 120],
    )
    colorbar.set_label("Z500 anomaly (gpm)", fontsize=13)
    colorbar.ax.tick_params(labelsize=10)

    fig.suptitle("Z500 anomaly centered 5-day mean around MHW peak", fontsize=18, y=0.982)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_check_file(out_ds, out_nc, out_png, out_pdf, out_check):
    lines = [
        "Z500 13+1 PEAK-CENTERED 5-DAY MEAN TWO-COLUMN CHECK",
        "=" * 88,
        f"Anomaly file: {ANOM_FILE}",
        f"Output NC: {out_nc}",
        f"Output PNG: {out_png}",
        f"Output PDF: {out_pdf}",
        "Domain: 170E-240E, 20N-75N",
        "Running mean: centered 5-day mean, target date +/- 2 days",
        f"Group13 events: {len(GROUP13_PEAK_DATES)}",
        "Group13 peak dates: " + ", ".join(date_label(value) for value in GROUP13_PEAK_DATES),
        f"Special event: {date_label(SPECIAL_PEAK_DATE)}",
        "",
    ]

    for relative_day, panel_title in zip(RELATIVE_DAYS, PANEL_TITLES):
        group_field = out_ds["z500_anomaly_5dmean_group13"].sel(relative_day=relative_day)
        special_field = out_ds["z500_anomaly_5dmean_special2004"].sel(relative_day=relative_day)
        p_value = out_ds["p_value"].sel(relative_day=relative_day)
        significant = out_ds["significant_p005"].sel(relative_day=relative_day)
        sample_count = int(out_ds["sample_count"].sel(relative_day=relative_day).values)
        special_target = pd.to_datetime(
            out_ds["special2004_target_date"].sel(relative_day=relative_day).values
        )

        lines.extend(
            [
                f"{panel_title} (relative day {int(relative_day):+d}):",
                f"  group13 sample_count = {sample_count}",
                (
                    "  group13 mean/min/max/NaN = "
                    f"{float(group_field.mean().values):.3f} / "
                    f"{float(group_field.min().values):.3f} / "
                    f"{float(group_field.max().values):.3f} / "
                    f"{int(group_field.isnull().sum().values)}"
                ),
                f"  p_value NaN = {int(p_value.isnull().sum().values)}",
                f"  significant grids p<{ALPHA} = {int(significant.sum().values)} / {significant.size}",
                (
                    "  special2004 target/window = "
                    f"{date_label(special_target)} / "
                    f"{date_label(special_target - pd.Timedelta(days=ROLL_HALF_WIDTH))} to "
                    f"{date_label(special_target + pd.Timedelta(days=ROLL_HALF_WIDTH))}"
                ),
                (
                    "  special2004 mean/min/max/NaN = "
                    f"{float(special_field.mean().values):.3f} / "
                    f"{float(special_field.min().values):.3f} / "
                    f"{float(special_field.max().values):.3f} / "
                    f"{int(special_field.isnull().sum().values)}"
                ),
                "",
            ]
        )

    with open(out_check, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines).rstrip() + "\n")


def main():
    if len(GROUP13_PEAK_DATES) != 13:
        raise ValueError(f"Expected 13 regular events, found {len(GROUP13_PEAK_DATES)}")
    if SPECIAL_PEAK_DATE not in set(ALL_PEAK_DATES):
        raise ValueError("The 2004 special event is not present in ALL_PEAK_DATES")

    os.makedirs(OUT_DIR, exist_ok=True)

    out_nc = os.path.join(OUT_DIR, OUTPUT_STEM + ".nc")
    out_png = os.path.join(OUT_DIR, OUTPUT_STEM + ".png")
    out_pdf = os.path.join(OUT_DIR, OUTPUT_STEM + ".pdf")
    out_check = os.path.join(
        OUT_DIR,
        "check_Z500_anomaly_5dmean_group13_vs_2004_peak_lead15_lag15.txt",
    )

    print("Reading anomaly data:")
    print(ANOM_FILE)
    ds = xr.open_dataset(ANOM_FILE)
    if "z" not in ds.variables:
        raise ValueError(f"Variable z not found. Variables: {list(ds.variables)}")

    z_region = ds["z"].sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )
    if z_region.sizes["lat"] != 56 or z_region.sizes["lon"] != 71:
        raise ValueError(
            f"Unexpected regional shape: lat={z_region.sizes['lat']}, lon={z_region.sizes['lon']}"
        )

    time_set = set(pd.to_datetime(ds["time"].values))
    out_ds = compute_output_dataset(z_region, time_set)

    encoding = {
        "z500_anomaly_5dmean_group13": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "z500_anomaly_5dmean_special2004": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "p_value": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "significant_p005": {"zlib": True, "complevel": 4, "dtype": "int8"},
        "sample_count": {"zlib": True, "complevel": 4, "dtype": "int16"},
    }

    out_ds.to_netcdf(out_nc, encoding=encoding)
    plot_comparison(out_ds, out_png, out_pdf)
    write_check_file(out_ds, out_nc, out_png, out_pdf, out_check)

    print("Saved:")
    print(out_nc)
    print(out_png)
    print(out_pdf)
    print(out_check)

    ds.close()
    print("Done.")


if __name__ == "__main__":
    main()
