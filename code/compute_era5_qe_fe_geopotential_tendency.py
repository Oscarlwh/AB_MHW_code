from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy import stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    import compute_era5_qe_fe_peak_lag0_7 as qefe
except ModuleNotFoundError:
    from . import compute_era5_qe_fe_peak_lag0_7 as qefe


AUTHOR_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTHOR_CODE))
from Laplace_SOR import Laplac_SOR  # noqa: E402


PREP12_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_daily_1deg_monthly_12lev")
OUT_ROOT = Path("/path/to/data/非绝热加热异常_ERA5反推/era5_qe_fe_geopotential_tendency")
FORCING_ROOT = OUT_ROOT / "forcing_12lev"
SIGMA_FILE = Path(
    "/path/to/data/非绝热加热异常_ERA5反推/era5_qd_peak_lag0_7_author_sor_17lev/"
    "ERA5_Qd_peak_lag0_7_author_smoothvars_17lev_forcing.nc"
)

SOR_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
PLOT_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]
LAGS = list(range(8))
MEMBERS = {
    "group1_N12": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13],
    "event06_2004-12-24": [6],
    "event11_2020-11-13": [11],
}
MEMBER_LABELS = {
    "group1_N12": "N=12",
    "event06_2004-12-24": "2004-12-24",
    "event11_2020-11-13": "2020-11-13",
}
FILE_LABELS = {
    "group1_N12": "N12",
    "event06_2004-12-24": "2004-12-24",
    "event11_2020-11-13": "2020-11-13",
}
FACTOR_CONFIG = {
    # The author's larger relaxation factors diverge on this daily ERA5 domain;
    # SC=1.0 is the validated stable setting also used by the project Qd run.
    "Qe": {"forcing_type": 1, "sc": 1.0},
    "Fe": {"forcing_type": 2, "sc": 1.0},
}


def lag_tag() -> str:
    if LAGS == list(range(int(LAGS[0]), int(LAGS[-1]) + 1)):
        return f"lag{int(LAGS[0])}_{int(LAGS[-1])}"
    return "lag" + "-".join(str(int(x)) for x in LAGS)

DISPLAY_LAT_MIN, DISPLAY_LAT_MAX = 20.0, 80.0
DISPLAY_LON_MIN, DISPLAY_LON_MAX = 150.0, 250.0
BOX_LON_MIN, BOX_LON_MAX = 200.0, 225.0
BOX_LAT_MIN, BOX_LAT_MAX = 40.0, 50.0


def configure_qefe_module() -> None:
    qefe.ERA5_FILLED_8LEV_ROOT = PREP12_ROOT
    qefe.OUT_ROOT = FORCING_ROOT
    qefe.CACHE_TAG = "12lev"
    qefe.LEVELS = SOR_LEVELS
    qefe.LAT_MIN, qefe.LAT_MAX = 10.0, 85.0
    qefe.LON_MIN, qefe.LON_MAX = 120.0, 260.0
    # Keep the larger domain in forcing caches; plotting is cropped later.
    qefe.DISPLAY_LAT_MIN, qefe.DISPLAY_LAT_MAX = 10.0, 85.0
    qefe.DISPLAY_LON_MIN, qefe.DISPLAY_LON_MAX = 120.0, 260.0


def build_forcing(overwrite: bool) -> xr.Dataset:
    configure_qefe_module()
    target = qefe.target_table()
    var_clim = qefe.build_variable_climatology(overwrite)
    try:
        forcing_clim = qefe.build_qe_fe_daily_climatology(var_clim, target, overwrite)
        try:
            events = qefe.build_event_anomalies(var_clim, forcing_clim, target, overwrite)
        finally:
            forcing_clim.close()
    finally:
        var_clim.close()
    return events


def target_table() -> pd.DataFrame:
    return qefe.target_table()


def sigma_for_member_lag(sigma_ds: xr.Dataset, target: pd.DataFrame, indices: list[int], lag: int) -> np.ndarray:
    doys = target[(target.event.isin(indices)) & (target.lag == lag)].doy_nl.astype(int).tolist()
    sigma = sigma_ds["sigma1_area_mean_doy"].sel(doy_nl=doys, lev=SOR_LEVELS).mean("doy_nl")
    return np.maximum(sigma.values.astype(np.float64), 1.0e-12)


def cache_file(factor: str, member: str, event: int, lag: int) -> Path:
    return OUT_ROOT / "sor_cache" / factor / member / f"event{event:02d}_lag{lag}.npz"


def invert_one(
    field: np.ndarray,
    sigma: np.ndarray,
    lat: np.ndarray,
    factor: str,
) -> np.ndarray:
    cfg = FACTOR_CONFIG[factor]
    forcing = np.transpose(field, (2, 1, 0)).astype(np.float64)
    gt = laplac_sor_precomputed(
        forcing,
        sigma,
        cfg["sc"],
        1.0e-12,
        np.pi / 180.0,
        np.pi / 180.0,
        np.asarray(SOR_LEVELS, dtype=np.float64) * 100.0,
        lat,
        cfg["forcing_type"],
    )
    return np.transpose(gt, (2, 1, 0)).astype(np.float32)


def laplac_sor_precomputed(
    forcing: np.ndarray,
    sigma: np.ndarray,
    sc: float,
    cri: float,
    dx: float,
    dy: float,
    p: np.ndarray,
    lat: np.ndarray,
    forcing_type: int,
) -> np.ndarray:
    """Numerically equivalent author SOR with invariant coefficients cached."""
    del cri  # The convergence test is disabled in the author's implementation.
    radius = 6.371e6
    gas_constant = 287.0
    nx, ny, np0 = forcing.shape
    dp = -2500.0
    pp = np.arange(100000.0, 9900.0, dp)
    nz = len(pp)

    lat_rad = np.deg2rad(lat.astype(np.float64))
    cos_lat = np.cos(lat_rad)[None, :, None]
    sin_lat = np.sin(lat_rad)[None, :, None]
    coriolis = (2.0 * (2.0 * np.pi / 86400.0) * np.sin(lat_rad))[None, :, None]

    sigma_interp = np.flip(CubicSpline(np.flip(p), np.flip(sigma))(np.flip(pp)))
    reciprocal_sigma = (1.0 / sigma_interp)[None, None, :]
    forcing_interp = np.flip(
        CubicSpline(np.flip(p), np.flip(forcing, axis=2), axis=2)(np.flip(pp)),
        axis=2,
    )

    if forcing_type == 1:
        s1 = forcing_interp / (pp[None, None, :] * sigma_interp[None, None, :])
        frc1 = -coriolis * (np.gradient(s1, 1, axis=2) / dp) * gas_constant
    elif forcing_type == 2:
        frc1 = forcing_interp
    else:
        raise ValueError(f"Unknown forcing_type: {forcing_type}")
    frc = radius * radius * frc1 * coriolis * cos_lat * cos_lat

    cos_i = cos_lat[:, 1:-1, :]
    sin_i = sin_lat[:, 1:-1, :]
    f_i = coriolis[:, 1:-1, :]
    r_mid = reciprocal_sigma[:, :, 1:-1]
    r_grad = reciprocal_sigma[:, :, 2:] - reciprocal_sigma[:, :, :-2]
    coe011 = 1.0 / (dx * dx)
    coe211 = coe011
    coe101 = cos_i * sin_i / (2.0 * dy) + cos_i * cos_i / (dy * dy)
    coe121 = -cos_i * sin_i / (2.0 * dy) + cos_i * cos_i / (dy * dy)
    vertical_base = radius * radius * cos_i * cos_i * f_i * f_i
    coe110 = -vertical_base * r_grad / (2.0 * dp) / (2.0 * dp) + vertical_base * r_mid / (dp * dp)
    coe112 = vertical_base * r_grad / (2.0 * dp) / (2.0 * dp) + vertical_base * r_mid / (dp * dp)
    coe111 = -2.0 / (dx * dx) - 2.0 * cos_i * cos_i / (dx * dx) - 2.0 * vertical_base * r_mid / (dp * dp)

    gt = np.zeros((nx, ny, nz), dtype=np.float64)
    for _ in range(3000):
        residual = frc[1:-1, 1:-1, 1:-1] - (
            coe011 * gt[:-2, 1:-1, 1:-1]
            + coe111 * gt[1:-1, 1:-1, 1:-1]
            + coe211 * gt[2:, 1:-1, 1:-1]
            + coe101 * gt[1:-1, :-2, 1:-1]
            + coe121 * gt[1:-1, 2:, 1:-1]
            + coe110 * gt[1:-1, 1:-1, :-2]
            + coe112 * gt[1:-1, 1:-1, 2:]
        )
        gt[1:-1, 1:-1, 1:-1] += sc / coe111 * residual
        if forcing_type == 2:
            gt[:, :, 0] = gt[:, :, 1]
            gt[:, :, -1] = gt[:, :, -2]
        else:
            gt[:, :, 0] = gt[:, :, 1] + gas_constant * dp / pp[0] * forcing_interp[:, :, 1]
            gt[:, :, -1] = gt[:, :, -2] - gas_constant * dp / pp[-1] * forcing_interp[:, :, -1]

    return np.flip(
        CubicSpline(np.flip(pp), np.flip(gt, axis=2), axis=2)(np.flip(p)),
        axis=2,
    ).reshape(nx, ny, np0)


def load_or_invert(
    factor: str,
    member: str,
    event: int,
    lag: int,
    event_ds: xr.Dataset,
    sigma: np.ndarray,
    overwrite: bool,
) -> np.ndarray:
    path = cache_file(factor, member, event, lag)
    if path.exists() and not overwrite:
        with np.load(path) as cached:
            return cached["phi"].astype(np.float32)
    print(f"[SOR] factor={factor} member={member} event={event} lag={lag}", flush=True)
    field = event_ds[f"{factor}_event_lag_anomaly"].sel(event=event, lag=lag, lev=SOR_LEVELS).values
    arr = invert_one(field, sigma, event_ds.lat.values.astype(np.float64), factor)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, phi=arr)
    tmp.replace(path)
    return arr


def output_file(factor: str) -> Path:
    return OUT_ROOT / f"ERA5_GT_by_{factor}_{lag_tag()}_12plus1plus1_author_SOR_1000-100hPa.nc"


def fill_sor_cache(factor: str, event_ds: xr.Dataset, lags: list[int], overwrite: bool) -> None:
    target = target_table()
    with xr.open_dataset(SIGMA_FILE) as sigma_ds:
        for member, indices in MEMBERS.items():
            for lag in lags:
                sigma = sigma_for_member_lag(sigma_ds, target, indices, lag)
                for event in indices:
                    load_or_invert(factor, member, event, lag, event_ds, sigma, overwrite)


def invert_factor(factor: str, event_ds: xr.Dataset, overwrite: bool) -> xr.Dataset:
    out_file = output_file(factor)
    if out_file.exists() and not overwrite:
        print(f"[CACHE] {out_file}", flush=True)
        return xr.open_dataset(out_file)
    if not SIGMA_FILE.exists():
        raise FileNotFoundError(SIGMA_FILE)
    target = target_table()
    lat = event_ds.lat.values
    lon = event_ds.lon.values
    members = list(MEMBERS)
    member_phi = np.full((len(members), len(LAGS), len(SOR_LEVELS), len(lat), len(lon)), np.nan, dtype=np.float32)
    p_value = np.full_like(member_phi, np.nan)
    n12_phi = np.full((len(MEMBERS["group1_N12"]), len(LAGS), len(SOR_LEVELS), len(lat), len(lon)), np.nan, dtype=np.float32)
    rows = []
    with xr.open_dataset(SIGMA_FILE) as sigma_ds:
        for mi, (member, indices) in enumerate(MEMBERS.items()):
            for lag in LAGS:
                sigma = sigma_for_member_lag(sigma_ds, target, indices, lag)
                samples = []
                for slot, event in enumerate(indices):
                    arr = load_or_invert(factor, member, event, lag, event_ds, sigma, overwrite)
                    samples.append(arr)
                    if member == "group1_N12":
                        n12_phi[slot, lag] = arr
                stack = np.stack(samples, axis=0)
                member_phi[mi, lag] = np.mean(stack, axis=0, dtype=np.float64).astype(np.float32)
                if len(indices) > 1:
                    _, p = stats.ttest_1samp(stack, popmean=0.0, axis=0, nan_policy="omit")
                    p_value[mi, lag] = p.astype(np.float32)
                p925 = member_phi[mi, lag, SOR_LEVELS.index(925)] * 1.0e4
                rows.append(
                    {
                        "factor": factor,
                        "member": member,
                        "lag": lag,
                        "n_events": len(indices),
                        "phi925_min_1e4": float(np.nanmin(p925)),
                        "phi925_max_1e4": float(np.nanmax(p925)),
                        "phi925_mean_1e4": float(np.nanmean(p925)),
                        "nan_count": int(np.isnan(member_phi[mi, lag]).sum()),
                        "inf_count": int(np.isinf(member_phi[mi, lag]).sum()),
                    }
                )

    n12_indices = MEMBERS["group1_N12"]
    ds = xr.Dataset(
        {
            "phi_tendency_member": (("member", "lag", "lev", "lat", "lon"), member_phi),
            "p_value": (("member", "lag", "lev", "lat", "lon"), p_value),
            "significant_p005": (("member", "lag", "lev", "lat", "lon"), p_value < 0.05),
            "phi_tendency_N12_event": (("n12_event", "lag", "lev", "lat", "lon"), n12_phi),
            "n_events": ("member", np.asarray([len(x) for x in MEMBERS.values()], dtype=np.int16)),
            "peak_date": ("event", np.asarray([np.datetime64(x) for x in qefe.PEAK_DATES])),
        },
        coords={
            "member": np.asarray(members, dtype=object),
            "n12_event": np.asarray(n12_indices, dtype=np.int16),
            "event": np.arange(len(qefe.PEAK_DATES), dtype=np.int16),
            "lag": np.asarray(LAGS, dtype=np.int16),
            "lev": np.asarray(SOR_LEVELS, dtype=np.int16),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": f"ERA5 geopotential tendency induced by {factor}, {lag_tag()}",
            "units": "m2 s-3",
            "forcing_type": FACTOR_CONFIG[factor]["forcing_type"],
            "relaxation_factor_SC": FACTOR_CONFIG[factor]["sc"],
            "sigma": "1981-2020 no-leap event-matched sigma1 area-mean profile",
            "horizontal_boundary": "fixed zero in author Laplace_SOR",
            "vertical_grid": "author internal 25 hPa grid, 1000-100 hPa",
        },
    )
    ds["phi_tendency_member"].attrs["units"] = "m2 s-3"
    ds["phi_tendency_N12_event"].attrs["units"] = "m2 s-3"
    encoding = {
        "phi_tendency_member": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "p_value": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "significant_p005": {"zlib": True, "complevel": 4},
        "phi_tendency_N12_event": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_file, encoding=encoding)
    pd.DataFrame(rows).to_csv(out_file.with_name(out_file.stem + "_check.csv"), index=False, encoding="utf-8-sig")
    ds.close()
    print(f"[NC] {out_file}", flush=True)
    return xr.open_dataset(out_file)


def paper_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_blue_red",
        ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff", "#fff7bc", "#fee391", "#fdae61", "#f46d43", "#b30000"],
        N=256,
    )


def lon_label(x: float) -> str:
    if int(x) == 180:
        return "180°"
    return f"{int(x)}°E" if x < 180 else f"{int(360 - x)}°W"


def nice_vmax(value: float) -> float:
    for candidate in [2, 4, 8, 12, 18, 25, 40, 60, 80, 120, 160, 250, 400, 800]:
        if value <= candidate:
            return float(candidate)
    return float(np.ceil(value / 500.0) * 500.0)


def add_box(ax, transform=None) -> None:
    xs = [BOX_LON_MIN, BOX_LON_MAX, BOX_LON_MAX, BOX_LON_MIN, BOX_LON_MIN]
    ys = [BOX_LAT_MIN, BOX_LAT_MIN, BOX_LAT_MAX, BOX_LAT_MAX, BOX_LAT_MIN]
    kwargs = {"color": "green", "linewidth": 1.5, "zorder": 9}
    if transform is not None:
        kwargs["transform"] = transform
    ax.plot(xs, ys, **kwargs)


def plot_member_level(ds: xr.Dataset, factor: str, member: str, plot_level: int, vmax: float) -> Path:
    da = (
        ds["phi_tendency_member"].sel(member=member, lev=plot_level, lat=slice(20, 80), lon=slice(150, 250))
        * 1.0e4
    )
    pval = ds["p_value"].sel(member=member, lev=plot_level, lat=slice(20, 80), lon=slice(150, 250))
    levels = np.linspace(-vmax, vmax, 17)
    data_crs = ccrs.PlateCarree() if HAS_CARTOPY else None
    proj = ccrs.PlateCarree(central_longitude=180) if HAS_CARTOPY else None
    fig = plt.figure(figsize=(7.45, 8.0))
    last = None
    for i, lag in enumerate(LAGS):
        ax = fig.add_subplot(4, 2, i + 1, projection=proj) if HAS_CARTOPY else fig.add_subplot(4, 2, i + 1)
        field = da.sel(lag=lag)
        lon, lat = field.lon.values, field.lat.values
        kwargs = {"transform": data_crs} if HAS_CARTOPY else {}
        last = ax.contourf(lon, lat, field.values, levels=levels, cmap=paper_cmap(), extend="both", **kwargs)
        if HAS_CARTOPY:
            ax.set_extent([150, 250, 20, 80], crs=data_crs)
            ax.coastlines(resolution="110m", linewidth=0.65, color="black", zorder=8)
            ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25, edgecolor="black", zorder=8)
            ax.set_xticks([150, 180, 210, 240], crs=data_crs)
            ax.set_yticks([20, 40, 60, 80], crs=data_crs)
            add_box(ax, data_crs)
        else:
            ax.set_xlim(150, 250)
            ax.set_ylim(20, 80)
            ax.set_xticks([150, 180, 210, 240])
            ax.set_yticks([20, 40, 60, 80])
            add_box(ax)
        if member == "group1_N12":
            sig = pval.sel(lag=lag).values < 0.05
            sig_sub = sig[::2, ::2]
            yy, xx = np.meshgrid(lat[::2], lon[::2], indexing="ij")
            scatter_kwargs = {"transform": data_crs} if HAS_CARTOPY else {}
            ax.scatter(xx[sig_sub], yy[sig_sub], s=6.0, color="black", marker="o", linewidths=0, zorder=10, **scatter_kwargs)
        ax.set_xticklabels([lon_label(x) for x in [150, 180, 210, 240]] if i >= 6 else [""] * 4, fontsize=8)
        ax.set_yticklabels([f"{int(y)}°N" for y in [20, 40, 60, 80]] if i % 2 == 0 else [""] * 4, fontsize=8)
        ax.tick_params(length=3, width=0.8, pad=2)
        ax.text(0.00, 1.02, f"({chr(97 + i)}) lag {lag} days", transform=ax.transAxes, ha="left", va="bottom", fontsize=9.3)
        factor_x = 0.58 if member == "group1_N12" else 0.52
        factor_size = 9.3 if member == "group1_N12" else 8.7
        ax.text(factor_x, 1.02, f"GT by {factor}", transform=ax.transAxes, ha="center", va="bottom", fontsize=factor_size)
        member_size = 9.3 if member == "group1_N12" else 8.0
        ax.text(0.98, 1.02, MEMBER_LABELS[member], transform=ax.transAxes, ha="right", va="bottom", fontsize=member_size)
    cax = fig.add_axes([0.22, 0.071, 0.56, 0.023])
    cb = fig.colorbar(last, cax=cax, orientation="horizontal", ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.ax.tick_params(labelsize=8, length=2.5, pad=1)
    unit = r"$10^{-4}$ m$^2$ s$^{-3}$"
    cb.set_label(f"{unit}\n{plot_level} hPa", fontsize=8, labelpad=2)
    cb.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.145, top=0.965, hspace=0.20, wspace=0.005)
    fig_dir = OUT_ROOT / "figures" / f"GT_by_{factor}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"ERA5_GT_by_{factor}_{plot_level}hPa_{FILE_LABELS[member]}_lag0_7_autoscale_vmax{int(vmax)}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[FIG] {out}", flush=True)
    return out


def plot_factor(ds: xr.Dataset, factor: str) -> None:
    rows = []
    for level in PLOT_LEVELS:
        for member in MEMBERS:
            values = np.abs((ds["phi_tendency_member"].sel(member=member, lev=level) * 1.0e4).values)
            vmax = nice_vmax(float(np.nanpercentile(values, 98.0)))
            out = plot_member_level(ds, factor, member, level, vmax)
            rows.append(
                {
                    "factor": factor,
                    "member": MEMBER_LABELS[member],
                    "level": level,
                    "vmax": vmax,
                    "significance": "p<0.05 for N=12 only",
                    "file": str(out),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_ROOT / f"ERA5_GT_by_{factor}_plot_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", nargs="+", choices=["Qe", "Fe"], default=["Qe", "Fe"])
    parser.add_argument("--overwrite-forcing", action="store_true")
    parser.add_argument("--overwrite-inversion", action="store_true")
    parser.add_argument("--forcing-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--lags", nargs="+", type=int, choices=LAGS, default=LAGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_qefe_module()
    if args.plot_only:
        for factor in args.factors:
            with xr.open_dataset(output_file(factor)) as ds:
                plot_factor(ds, factor)
        return
    event_ds = build_forcing(args.overwrite_forcing)
    try:
        if args.forcing_only:
            return
        if args.cache_only:
            for factor in args.factors:
                fill_sor_cache(factor, event_ds, args.lags, args.overwrite_inversion)
            return
        for factor in args.factors:
            result = invert_factor(factor, event_ds, args.overwrite_inversion)
            try:
                plot_factor(result, factor)
            finally:
                result.close()
    finally:
        event_ds.close()


if __name__ == "__main__":
    main()
