from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_era5_2024_jan_12lev as prep2024  # noqa: E402
import compute_era5_qd_peak_lag0_7_author_sor_17lev as qdgt  # noqa: E402
import compute_era5_qe_fe_geopotential_tendency as qefegt  # noqa: E402


DATA_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推")
PREP12_ROOT = DATA_ROOT / "era5_daily_1deg_monthly_12lev"
QD_OUT = DATA_ROOT / "era5_qd_peak_lagm2_27_author_sor_12lev"
QEFE_OUT = DATA_ROOT / "era5_qe_fe_geopotential_tendency_lagm2_27"
FIG_OUT = DATA_ROOT / "fig4_style_own_region_N13_plus_2004_lag0_25_center5day_lon180_225E"

LAGS_FULL = list(range(-2, 28))
PANEL_LAGS = [0, 5, 10, 15, 20, 25]
WINDOW = 5
LEVELS12 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]

LON_MIN, LON_MAX = 180.0, 225.0
LAT_MIN, LAT_MAX = 20.0, 80.0
LEV_MIN, LEV_MAX = 100.0, 1000.0

MEMBER_LABELS = {
    "N13_plus_2020": "N=13",
    "event06_2004-12-24": "2004-12-24",
}
MEMBER_FILES = {
    "N13_plus_2020": "N13",
    "event06_2004-12-24": "2004-12-24",
}
DIRECT_MEMBERS = {
    "N13_plus_2020": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13],
    "event06_2004-12-24": [6],
}


def paper_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "hu2025_like_blue_red",
        ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff", "#fff7bc", "#fee391", "#fdae61", "#f46d43", "#b30000"],
        N=256,
    )


def nice_vmax(value: float) -> float:
    for candidate in [2, 4, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 125, 160]:
        if value <= candidate:
            return float(candidate)
    return float(np.ceil(value / 50.0) * 50.0)


def format_lat(x: float) -> str:
    return f"{int(round(x))}°N"


def configure_qd() -> None:
    qdgt.ERA5_FILLED_17LEV_ROOT = PREP12_ROOT
    qdgt.OUT_ROOT = QD_OUT
    qdgt.LEVELS = LEVELS12
    qdgt.SOR_LEVELS = LEVELS12
    qdgt.P_PA = np.array(LEVELS12, dtype=np.float64) * 100.0
    qdgt.LAGS = LAGS_FULL
    qdgt.PLOT_MEMBERS = ["group1_N12", "event06_2004-12-24", "event11_2020-11-13"]


def load_season_with_2024_jan(season_year: int, truncate_2023: bool = False) -> xr.Dataset:
    del truncate_2023
    qefe = qefegt.qefe
    if season_year == 2023:
        months = [(2023, 10), (2023, 11), (2023, 12), (2024, 1)]
        dates = pd.date_range("2023-10-01", "2024-01-31", freq="D")
    else:
        months = qefe.season_months(season_year, truncate_2023=False)
        dates = pd.date_range(f"{season_year}-10-01", f"{season_year + 1}-02-28", freq="D")
    parts = [qefe.load_month(year, month) for year, month in months]
    try:
        ds = xr.concat(parts, dim="time", coords="minimal", compat="override").sortby("time").load()
        return ds.sel(time=dates).load()
    finally:
        for part in parts:
            part.close()


def configure_qefe(qd_forcing_file: Path) -> None:
    qefe = qefegt.qefe
    qefe.ERA5_FILLED_8LEV_ROOT = PREP12_ROOT
    qefe.OUT_ROOT = QEFE_OUT / "forcing_12lev"
    qefe.CACHE_TAG = "12lev"
    qefe.LEVELS = LEVELS12
    qefe.LAGS = LAGS_FULL
    qefe.LAT_MIN, qefe.LAT_MAX = 10.0, 85.0
    qefe.LON_MIN, qefe.LON_MAX = 120.0, 260.0
    qefe.DISPLAY_LAT_MIN, qefe.DISPLAY_LAT_MAX = 10.0, 85.0
    qefe.DISPLAY_LON_MIN, qefe.DISPLAY_LON_MAX = 120.0, 260.0
    qefe.load_season = load_season_with_2024_jan

    qefegt.PREP12_ROOT = PREP12_ROOT
    qefegt.OUT_ROOT = QEFE_OUT
    qefegt.FORCING_ROOT = QEFE_OUT / "forcing_12lev"
    qefegt.SIGMA_FILE = qd_forcing_file
    qefegt.SOR_LEVELS = LEVELS12
    qefegt.LAGS = LAGS_FULL
    qefegt.MEMBERS = {
        "group1_N12": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13],
        "event06_2004-12-24": [6],
        "event11_2020-11-13": [11],
    }


def compute_inputs(overwrite: bool) -> tuple[Path, Path, Path]:
    prep2024.build_daily(overwrite=False)
    configure_qd()
    qd_forcing = qdgt.build_forcing(QD_OUT, overwrite=overwrite)
    try:
        qd_phi = qdgt.invert_author_sor(qd_forcing, QD_OUT, mode="sigma1", overwrite=overwrite, sc=1.0)
        qd_path = Path(qd_phi.encoding.get("source", QD_OUT / "ERA5_Qd_peak_lag-2_27_phi_tendency_author_SOR_1000-100hPa_sigma1_SC1p0.nc"))
        qd_phi.close()
    finally:
        forcing_file = Path(qd_forcing.encoding.get("source", QD_OUT / "ERA5_Qd_peak_lag-2_27_author_smoothvars_17lev_forcing.nc"))
        qd_forcing.close()
        gc.collect()

    configure_qefe(forcing_file)
    event_ds = qefegt.build_forcing(overwrite)
    try:
        member_ds = qefegt.qefe.build_composites(event_ds, overwrite)
        try:
            qefile = invert_qefe_direct_members("Qe", member_ds, overwrite)
            fefile = invert_qefe_direct_members("Fe", member_ds, overwrite)
        finally:
            member_ds.close()
    finally:
        event_ds.close()
        gc.collect()

    return qd_path, qefile, fefile


def direct_qefe_output_file(factor: str) -> Path:
    return QEFE_OUT / f"ERA5_GT_by_{factor}_lagm2_27_direct_N13_plus_2004_author_SOR_1000-100hPa.nc"


def direct_qefe_cache_file(factor: str, member: str, lag: int) -> Path:
    return QEFE_OUT / "sor_cache_direct" / factor / member / f"lag{lag}.npz"


def direct_member_forcing(member_ds: xr.Dataset, factor: str, member: str, lag: int) -> np.ndarray:
    var = f"{factor}_member_lag_anomaly"
    if member == "N13_plus_2020":
        field = (12.0 * member_ds[var].sel(member="group1_N12", lag=lag) + member_ds[var].sel(member="event11_2020-11-13", lag=lag)) / 13.0
    else:
        field = member_ds[var].sel(member=member, lag=lag)
    return field.sel(lev=LEVELS12).values.astype(np.float32)


def load_or_invert_direct_qefe(
    factor: str,
    member: str,
    lag: int,
    member_ds: xr.Dataset,
    sigma: np.ndarray,
    overwrite: bool,
) -> np.ndarray:
    path = direct_qefe_cache_file(factor, member, lag)
    if path.exists() and not overwrite:
        with np.load(path) as cached:
            return cached["phi"].astype(np.float32)
    print(f"[SOR-DIRECT] factor={factor} member={member} lag={lag}", flush=True)
    field = direct_member_forcing(member_ds, factor, member, lag)
    arr = qefegt.invert_one(field, sigma, member_ds.lat.values.astype(np.float64), factor)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, phi=arr)
    tmp.replace(path)
    return arr


def invert_qefe_direct_members(factor: str, member_ds: xr.Dataset, overwrite: bool) -> Path:
    out_file = direct_qefe_output_file(factor)
    if out_file.exists() and not overwrite:
        print(f"[CACHE] {out_file}", flush=True)
        return out_file
    if not qefegt.SIGMA_FILE.exists():
        raise FileNotFoundError(qefegt.SIGMA_FILE)
    target = qefegt.target_table()
    members = list(DIRECT_MEMBERS)
    lat = member_ds.lat.values
    lon = member_ds.lon.values
    phi = np.full((len(members), len(LAGS_FULL), len(LEVELS12), len(lat), len(lon)), np.nan, dtype=np.float32)
    rows: list[dict[str, object]] = []
    with xr.open_dataset(qefegt.SIGMA_FILE) as sigma_ds:
        for mi, member in enumerate(members):
            indices = DIRECT_MEMBERS[member]
            for li, lag in enumerate(LAGS_FULL):
                sigma = qefegt.sigma_for_member_lag(sigma_ds, target, indices, lag)
                arr = load_or_invert_direct_qefe(factor, member, lag, member_ds, sigma, overwrite)
                phi[mi, li] = arr
                vals925 = arr[LEVELS12.index(925)] * 1.0e4
                rows.append(
                    {
                        "factor": factor,
                        "member": member,
                        "lag": lag,
                        "n_events_for_sigma": len(indices),
                        "phi925_min_1e4": float(np.nanmin(vals925)),
                        "phi925_max_1e4": float(np.nanmax(vals925)),
                        "phi925_mean_1e4": float(np.nanmean(vals925)),
                        "nan_count": int(np.isnan(arr).sum()),
                        "inf_count": int(np.isinf(arr).sum()),
                    }
                )
    ds = xr.Dataset(
        {
            "phi_tendency_member": (("member", "lag", "lev", "lat", "lon"), phi),
            "n_events_for_sigma": ("member", np.asarray([len(DIRECT_MEMBERS[m]) for m in members], dtype=np.int16)),
        },
        coords={
            "member": np.asarray(members, dtype=object),
            "lag": np.asarray(LAGS_FULL, dtype=np.int16),
            "lev": np.asarray(LEVELS12, dtype=np.int16),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": f"ERA5 GT by {factor}, direct SOR on N13 and 2004 member forcing, lag-2..27",
            "units": "m2 s-3",
            "method": "Direct inversion of member-composite Qe/Fe forcing; no event-level significance fields.",
            "N13_definition": "(12 * group1_N12 + event11_2020-11-13) / 13",
            "sigma": "1981-2020 no-leap event-matched sigma1 area-mean profile",
        },
    )
    ds["phi_tendency_member"].attrs["units"] = "m2 s-3"
    encoding = {"phi_tendency_member": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    QEFE_OUT.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_file, encoding=encoding)
    ds.close()
    check = out_file.with_name(out_file.stem + "_check.csv")
    pd.DataFrame(rows).to_csv(check, index=False, encoding="utf-8-sig")
    print(f"[NC] {out_file}", flush=True)
    print(f"[CSV] {check}", flush=True)
    return out_file


def select_member(da: xr.DataArray, member: str) -> xr.DataArray:
    if "member" in da.dims and member in set(str(x) for x in da.member.values):
        return da.sel(member=member)
    if member == "N13_plus_2020":
        return (12.0 * da.sel(member="group1_N12") + da.sel(member="event11_2020-11-13")) / 13.0
    return da.sel(member=member)


def load_gt(factor: str, path: Path, member: str) -> xr.DataArray:
    var = "phi_tendency_qd" if factor == "Qd" else "phi_tendency_member"
    with xr.open_dataset(path) as ds:
        da = ds[var].load()
    da = select_member(da, member)
    if float(da.lon.max()) <= 180.0:
        da = da.assign_coords(lon=da.lon % 360).sortby("lon")
    da = da.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    da = da.where((da.lev >= LEV_MIN) & (da.lev <= LEV_MAX), drop=True)
    return da.sortby("lat").sortby("lev")


def five_day_section(da: xr.DataArray, panel_lag: int) -> xr.DataArray:
    half_width = WINDOW // 2
    lags = list(range(panel_lag - half_width, panel_lag + half_width + 1))
    missing = [lag for lag in lags if lag not in da.lag.values]
    if missing:
        raise KeyError(f"Missing lags for panel lag{panel_lag}: {missing}")
    return da.sel(lag=lags).mean("lag", skipna=True).mean("lon", skipna=True) * 1.0e4


def plot_factor_member(factor: str, da: xr.DataArray, member: str, out_root: Path) -> tuple[Path, list[dict[str, object]]]:
    panels = [five_day_section(da, lag) for lag in PANEL_LAGS]
    vmax = nice_vmax(float(np.nanpercentile(np.abs(np.stack([p.values for p in panels])), 98.0)))
    levels = np.linspace(-vmax, vmax, 17)
    fig, axes = plt.subplots(3, 2, figsize=(7.45, 6.55), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    rows: list[dict[str, object]] = []
    last = None
    for i, panel_lag in enumerate(PANEL_LAGS):
        ax = axes_flat[i]
        field = panels[i]
        lat = field.lat.values.astype(float)
        lev = field.lev.values.astype(float)
        values = field.transpose("lev", "lat").values
        last = ax.contourf(lat, lev, values, levels=levels, cmap=paper_cmap(), extend="both")
        ax.axhline(500, color="0.35", linewidth=0.35, alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylim(1000, 100)
        ax.set_xlim(LAT_MIN, LAT_MAX)
        ax.set_title(f"({chr(97 + i)}) lag {panel_lag} day", loc="left", fontsize=9.3, pad=3)
        ax.set_title(f"GT by {factor}", loc="center", fontsize=9.3, pad=3)
        ax.set_title(MEMBER_LABELS[member], loc="right", fontsize=9.0 if member != "N13_plus_2020" else 9.3, pad=3)
        pressure_ticks = [1000, 850, 700, 500, 300, 200, 100]
        ax.yaxis.set_major_locator(FixedLocator(pressure_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(["1000", "850", "700", "500", "300", "200", "100"] if i % 2 == 0 else [""] * len(pressure_ticks)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        lat_ticks = [20, 40, 60, 80]
        ax.xaxis.set_major_locator(FixedLocator(lat_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([format_lat(x) for x in lat_ticks] if i >= 4 else [""] * len(lat_ticks)))
        ax.tick_params(length=3, width=0.7, pad=1.5, labelsize=7.6)
        labels = ax.get_xticklabels()
        if labels:
            labels[0].set_ha("left")
            labels[-1].set_ha("right")
        if i % 2 == 0:
            ax.set_ylabel("hPa", fontsize=8.5, labelpad=2)
        rows.append(
            {
                "factor": factor,
                "member": MEMBER_LABELS[member],
                "member_key": member,
                "panel_lag": panel_lag,
                "window_lags": ",".join(str(lag) for lag in range(panel_lag - WINDOW // 2, panel_lag + WINDOW // 2 + 1)),
                "lon_average_min_E": LON_MIN,
                "lon_average_max_E": LON_MAX,
                "vmax_1e4_m2_s3": vmax,
                "min_1e4_m2_s3": float(np.nanmin(values)),
                "max_1e4_m2_s3": float(np.nanmax(values)),
                "mean_1e4_m2_s3": float(np.nanmean(values)),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
            }
        )
    for ax in axes_flat[len(PANEL_LAGS):]:
        ax.axis("off")
    if last is None:
        raise RuntimeError("No panels plotted")
    cax = fig.add_axes([0.22, 0.060, 0.56, 0.023])
    cb = fig.colorbar(last, cax=cax, orientation="horizontal", ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.ax.tick_params(labelsize=8, length=2.5, pad=1)
    cb.set_label(r"$10^{-4}$ m$^2$ s$^{-3}$", fontsize=8.5, labelpad=2)
    cb.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.078, right=0.970, bottom=0.135, top=0.955, hspace=0.245, wspace=0.105)
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / f"ERA5_Hu2025_fig4_style_GT_by_{factor}_{MEMBER_FILES[member]}_lag0_25_center5day_lat_pressure.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out, rows


def plot_all(paths: tuple[Path, Path, Path]) -> None:
    factor_paths = {"Qd": paths[0], "Qe": paths[1], "Fe": paths[2]}
    rows: list[dict[str, object]] = []
    for factor, path in factor_paths.items():
        for member in ["N13_plus_2020", "event06_2004-12-24"]:
            da = load_gt(factor, path, member)
            out, subrows = plot_factor_member(factor, da, member, FIG_OUT)
            for row in subrows:
                row["source_file"] = str(path)
                row["png"] = str(out)
            rows.extend(subrows)
            print(f"[FIG] {out}", flush=True)
    check = FIG_OUT / "ERA5_Hu2025_fig4_style_N13_plus_2004_lag0_25_center5day_check.csv"
    pd.DataFrame(rows).to_csv(check, index=False, encoding="utf-8-sig")
    print(f"[CSV] {check}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        paths = (
            QD_OUT / "ERA5_Qd_peak_lag-2_27_phi_tendency_author_SOR_1000-100hPa_sigma1_SC1p0.nc",
            direct_qefe_output_file("Qe"),
            direct_qefe_output_file("Fe"),
        )
    else:
        paths = compute_inputs(args.overwrite)
    plot_all(paths)


if __name__ == "__main__":
    main()
