from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize


EVENT_FILE = Path(
    "/path/to/user/Documents/DesktopOrganizer/Folders/Paper/论文/表格文件/"
    "NEP_MHW_events_1981-2024_area_mean_smoothed30.csv"
)
DAILY_FILE = Path(
    "/path/to/data/MHW/NEP_MHW_daily_1981-2024_area_mean_smoothed30.nc"
)
OUT_DIR = Path(
    "/path/to/user/Documents/Codex/ABH/figures/MHW14_and_2004_intensity_GRL"
)

EVENT_ID = 39
SPECIAL_PEAK = pd.Timestamp("2004-12-24")
PEAK_DATES = pd.DatetimeIndex(
    pd.to_datetime(
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
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
RED = "#FF0000"
BLACK = "#1A1A1A"
GRID = "#D9D9D9"


def read_table_s1_event_intensities() -> pd.DataFrame:
    """Read the same 14 events used in the updated Table S1."""
    source = pd.read_csv(EVENT_FILE, encoding="utf-8-sig")
    source["peak_date"] = pd.to_datetime(source["peak_date"])
    indexed = source.set_index("peak_date", drop=False)
    missing = PEAK_DATES.difference(indexed.index)
    if len(missing):
        raise ValueError(f"Table S1 peak dates missing from CSV: {missing.tolist()}")
    events = (
        indexed.loc[PEAK_DATES]
        .reset_index(drop=True)
        .rename(columns={"max_intensity": "maximum_intensity"})
    )
    events = events[["peak_date", "maximum_intensity", "mean_intensity"]]
    if len(events) != 14:
        raise ValueError(f"Expected 14 MHW events, found {len(events)}.")
    if events["peak_date"].eq(SPECIAL_PEAK).sum() != 1:
        raise ValueError("The 2004-12-24 event is missing or duplicated.")
    return events


def read_2004_daily_intensity() -> pd.DataFrame:
    with xr.open_dataset(DAILY_FILE) as dataset:
        variables = ["sst", "sst_clim", "sst_q90", "sst_anomaly", "is_mhw_day"]
        event = (
            dataset[variables]
            .where(dataset["event_id"] == EVENT_ID, drop=True)
            .to_dataframe()
            .reset_index()
        )

    event["time"] = pd.to_datetime(event["time"]).dt.normalize()
    event = event.sort_values("time").reset_index(drop=True)
    expected_dates = pd.date_range("2004-12-20", "2004-12-28", freq="D")
    if not np.array_equal(event["time"].to_numpy(), expected_dates.to_numpy()):
        raise ValueError("Event 39 does not cover 2004-12-20 to 2004-12-28.")
    if event.loc[event["sst_anomaly"].idxmax(), "time"] != SPECIAL_PEAK:
        raise ValueError("The daily-intensity peak is not 2004-12-24.")
    return event


def apply_grl_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(events: pd.DataFrame, daily: pd.DataFrame) -> tuple[Path, Path]:
    apply_grl_style()
    fig = plt.figure(figsize=(7.4, 6.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[28, 0.8],
        height_ratios=[2.35, 1.0],
        left=0.085,
        right=0.94,
        bottom=0.095,
        top=0.965,
        hspace=0.72,
        wspace=0.12,
    )
    ax_top = fig.add_subplot(grid[0, :])
    ax_bar = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[1, 1])

    x = np.arange(len(events))
    maximum = events["maximum_intensity"].to_numpy()
    mean = events["mean_intensity"].to_numpy()

    ax_top.fill_between(x, mean, maximum, color="#B8C9D6", alpha=0.28, zorder=1)
    ax_top.plot(
        x,
        maximum,
        color=BLUE,
        marker="o",
        markersize=4.2,
        linewidth=1.6,
        label="Maximum intensity",
        zorder=3,
    )
    ax_top.plot(
        x,
        mean,
        color=ORANGE,
        marker="s",
        markersize=4.0,
        linewidth=1.5,
        label="Mean intensity",
        zorder=3,
    )

    for xi, high, avg in zip(x, maximum, mean):
        ax_top.text(
            xi,
            high + 0.055,
            f"{high:.2f}",
            color=BLUE,
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
        ax_top.text(
            xi,
            avg - 0.065,
            f"{avg:.2f}",
            color=ORANGE,
            ha="center",
            va="top",
            fontsize=6.2,
        )

    ax_top.set_xlim(-0.55, len(events) - 0.45)
    ax_top.set_ylim(0.72, 2.80)
    ax_top.set_ylabel("MHW intensity (°C)")
    ax_top.set_xlabel("Peak date", labelpad=5)
    ax_top.set_xticks(x)
    date_labels = [date.strftime("%Y/%m/%d") for date in events["peak_date"]]
    ax_top.set_xticklabels(date_labels, rotation=48, ha="right", rotation_mode="anchor")
    for label, date in zip(ax_top.get_xticklabels(), events["peak_date"]):
        label.set_color(BLACK if date == SPECIAL_PEAK else RED)
        label.set_fontweight("bold" if date == SPECIAL_PEAK else "normal")

    ax_top.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75)
    ax_top.set_axisbelow(True)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.2)
    ax_top.set_title(
        "(a) Intensity of the 14 early-winter MHW events",
        loc="left",
        fontweight="bold",
        pad=7,
    )

    dates = daily["time"].to_numpy()
    intensity = daily["sst_anomaly"].to_numpy(dtype=float)
    date_numbers = mdates.date2num(dates)
    x_edges = np.r_[date_numbers - 0.5, date_numbers[-1] + 0.5]
    norm = Normalize(vmin=0.85, vmax=1.00)
    mesh = ax_bar.pcolormesh(
        x_edges,
        [0.0, 1.0],
        intensity[np.newaxis, :],
        cmap="YlOrRd",
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=1.0,
    )

    peak_index = int(np.argmax(intensity))
    ax_bar.scatter(
        date_numbers[peak_index],
        1.06,
        marker="v",
        s=26,
        color=BLACK,
        clip_on=False,
        zorder=5,
    )
    ax_bar.text(
        date_numbers[peak_index],
        1.16,
        "Peak",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )
    ax_bar.set_xlim(x_edges[0], x_edges[-1])
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.set_yticks([])
    ax_bar.set_xticks(date_numbers)
    ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_bar.tick_params(axis="x", length=0, pad=5)
    ax_bar.set_xlabel("Date in 2004", labelpad=5)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.set_title(
        "(b) Daily intensity of the 2004 Northeast Pacific MHW",
        loc="left",
        fontweight="bold",
        y=1.34,
        pad=0,
    )

    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_label("MHW intensity (°C)", labelpad=6)
    colorbar.set_ticks([0.85, 0.90, 0.95, 1.00])
    colorbar.outline.set_linewidth(0.6)
    cax.tick_params(width=0.6, length=3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "Figure_MHW14_and_2004_daily_intensity_GRL_TableS1_updated.png"
    pdf_path = OUT_DIR / "Figure_MHW14_and_2004_daily_intensity_GRL_TableS1_updated.pdf"
    # Include the complete vertical colorbar label in the exported canvas.
    fig.savefig(
        png_path,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    events = read_table_s1_event_intensities()
    daily = read_2004_daily_intensity()
    png_path, pdf_path = plot_figure(events, daily)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(
        OUT_DIR / "MHW14_TableS1_intensity_values.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    daily.to_csv(
        OUT_DIR / "MHW_2004_daily_intensity.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    special = events.loc[events["peak_date"].eq(SPECIAL_PEAK)].iloc[0]
    peak = daily.loc[daily["sst_anomaly"].idxmax()]
    print(f"Top-panel events: {len(events)}")
    print(
        "2004 event in panel (a): "
        f"maximum={special['maximum_intensity']:.2f} °C, "
        f"mean={special['mean_intensity']:.2f} °C"
    )
    print(
        f"Panel (b) peak: {peak['time']:%Y-%m-%d}, "
        f"{peak['sst_anomaly']:.6f} °C"
    )
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
