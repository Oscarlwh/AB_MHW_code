import os
import numpy as np
import xarray as xr
import pandas as pd

# =========================================================
# 0. 路径设置
# =========================================================
workdir = "/path/to/data/1"

file_msshf = os.path.join(
    workdir,
    "ERA5_msshf_daily_1x1_200E-225E_40N-50N_1981_2024_NOLEAP_detrended.nc"
)
file_mslhf = os.path.join(
    workdir,
    "ERA5_mslhf_daily_1x1_200E-225E_40N-50N_1981_2024_NOLEAP_detrended.nc"
)

clim_start = "1991-01-01"
clim_end   = "2020-12-31"

# =========================================================
# 1. 构造严格的 NOLEAP dayofyear（1~365）
# =========================================================
def make_noleap_doy(time_coord: xr.DataArray) -> xr.DataArray:
    time_index = pd.DatetimeIndex(time_coord.values)

    doy = np.array(time_index.dayofyear, dtype=np.int32)
    is_leap = np.array(time_index.is_leap_year, dtype=bool)
    after_feb28 = np.array(time_index.month > 2, dtype=bool)

    noleap_doy = doy.copy()
    mask = is_leap & after_feb28
    noleap_doy[mask] -= 1

    return xr.DataArray(
        noleap_doy,
        dims=["time"],
        coords={"time": time_coord},
        name="noleap_doy"
    )

# =========================================================
# 2. 单变量处理函数
# =========================================================
def calc_clim_anom_from_detrended_fixed(infile: str, varname: str):
    ds = xr.open_dataset(infile)

    print("\n" + "=" * 90)
    print(f"Processing {varname}")
    print("=" * 90)
    print(ds)

    if varname not in ds.data_vars:
        raise ValueError(f"{varname} not found in {infile}")

    da = ds[varname].transpose("time", "lat", "lon")

    # -------------------------
    # 构造 noleap_doy
    # -------------------------
    noleap_doy_all = make_noleap_doy(da["time"])

    print("\nNOLEAP DOY check:")
    print("min =", int(noleap_doy_all.min().values))
    print("max =", int(noleap_doy_all.max().values))
    print("unique count =", len(np.unique(noleap_doy_all.values)))

    # -------------------------
    # 基准时段
    # -------------------------
    da_base = da.sel(time=slice(clim_start, clim_end))
    noleap_doy_base = noleap_doy_all.sel(time=slice(clim_start, clim_end))

    print(f"\nBase period for {varname}:")
    print(str(da_base.time.values[0]), "to", str(da_base.time.values[-1]))
    print("Length =", da_base.sizes["time"])

    # -------------------------
    # 气候态
    # 注意：这里先保留分组维名为 noleap_doy
    # -------------------------
    clim_core = da_base.groupby(noleap_doy_base).mean("time").astype("float32")

    # 用于异常计算的气候态（维度名保留为 noleap_doy）
    clim_for_anom = clim_core.copy()

    # 用于输出的气候态（把 noleap_doy 改成 dayofyear）
    clim_out_da = clim_core.rename({"noleap_doy": "dayofyear"}).sortby("dayofyear")
    clim = xr.Dataset({varname: clim_out_da})

    clim[varname].attrs = da.attrs.copy()
    clim.attrs = {
        "Title": f"Daily climatology of detrended {varname}",
        "Base_Period": "1991-2020",
        "Calendar": "NOLEAP",
        "Region": "40-50N, 200-225E",
        "Grid": "1x1 degree"
    }

    # -------------------------
    # 异常
    # 这里右边必须还是 noleap_doy 维
    # -------------------------
    anom_da = (da.groupby(noleap_doy_all) - clim_for_anom).astype("float32")

    # groupby 减法后一般会保留一个 noleap_doy 坐标，删掉即可
    if "noleap_doy" in anom_da.coords:
        anom_da = anom_da.drop_vars("noleap_doy")

    anom_da = anom_da.transpose("time", "lat", "lon")
    anom = xr.Dataset({varname: anom_da})

    anom[varname].attrs = da.attrs.copy()
    anom[varname].attrs["long_name"] = da.attrs.get("long_name", varname) + " anomaly"
    anom.attrs = {
        "Title": f"Daily anomaly of detrended {varname}",
        "Anomaly_Base_Period": "1991-2020",
        "Calendar": "NOLEAP",
        "Region": "40-50N, 200-225E",
        "Grid": "1x1 degree"
    }

    # -------------------------
    # 输出
    # -------------------------
    clim_file = os.path.join(
        workdir,
        f"ERA5_{varname}_clim_NOLEAPDOY_1991_2020_200E-225E_40N-50N_detrended_fixed.nc"
    )
    anom_file = os.path.join(
        workdir,
        f"ERA5_{varname}_anom_NOLEAPDOY_1981_2024_200E-225E_40N-50N_detrended_fixed.nc"
    )

    encoding_clim = {
        varname: {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat": {"dtype": "float32"},
        "lon": {"dtype": "float32"},
        "dayofyear": {"dtype": "int32"},
    }
    encoding_anom = {
        varname: {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat": {"dtype": "float32"},
        "lon": {"dtype": "float32"},
    }

    clim.to_netcdf(clim_file, encoding=encoding_clim)
    anom.to_netcdf(anom_file, encoding=encoding_anom)

    print(f"\n[OK] Saved climatology: {clim_file}")
    print(f"[OK] Saved anomaly   : {anom_file}")

    print("\nClimatology dataset:")
    print(clim)

    print("\nAnomaly dataset:")
    print(anom)

    print("\nChecks:")
    print("clim dayofyear len     =", clim.sizes["dayofyear"])
    print("clim dayofyear min/max =", int(clim["dayofyear"].min().values), int(clim["dayofyear"].max().values))
    print("anom time len          =", anom.sizes["time"])
    print("anom min               =", float(anom[varname].min(skipna=True).values))
    print("anom max               =", float(anom[varname].max(skipna=True).values))
    print("anom mean              =", float(anom[varname].mean(skipna=True).values))

    return clim, anom

# =========================================================
# 3. 主程序
# =========================================================
clim_msshf, anom_msshf = calc_clim_anom_from_detrended_fixed(file_msshf, "msshf")
clim_mslhf, anom_mslhf = calc_clim_anom_from_detrended_fixed(file_mslhf, "mslhf")

print("\nDone.")