import os
import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore")

# ============================================================
# 1. paths and settings
# ============================================================
IN_FILE = (
    "/path/to/data/ABH_170E240E_1981_2024/"
    "ERA5_Z500_daily_170E240E_35N90N_1981_2024_merged_gpm_no_leap.nc"
)

OUT_DIR = "/path/to/data/ABH_170E240E_1981_2024/anomaly"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_DETRENDED = os.path.join(
    OUT_DIR,
    "ERA5_Z500_daily_170E240E_35N90N_1981_2024_merged_gpm_no_leap_detrended.nc"
)

OUT_CLIM = os.path.join(
    OUT_DIR,
    "ERA5_Z500_clim_NOLEAP_1981_2020_gpm_170E240E_35N90N_detrended.nc"
)

OUT_ANOM = os.path.join(
    OUT_DIR,
    "ERA5_Z500_anom_NOLEAP_1981_2024_vs1981_2020_gpm_170E240E_35N90N_detrended.nc"
)

OUT_CHECK = os.path.join(
    OUT_DIR,
    "check_detrend_clim_anom_1981_2024_vs1981_2020.txt"
)

CLIM_START = "1981-01-01"
CLIM_END = "2020-12-31"

# 初始读取 chunk
CHUNKS_READ = {
    "time": 365,
    "lat": 56,
    "lon": 71,
}

# 去趋势 chunk：time 必须是单个 chunk
# 如果内存压力大，把 lat/lon 改成 4
CHUNKS_DETREND = {
    "time": -1,
    "lat": 8,
    "lon": 8,
}

# 保存前后检查用 chunk
CHUNKS_REOPEN = {
    "time": 365,
    "lat": 56,
    "lon": 71,
}


# ============================================================
# 2. helper functions
# ============================================================
def add_noleap_doy(ds):
    """
    给 no-leap 数据添加 doy_nl 坐标。

    因为 Feb 29 已经删除，不能直接用闰年的 dayofyear。
    这里用非闰年 2001 把每个日期映射到 1–365。
    """
    time = pd.to_datetime(ds["time"].values)

    doy_nl = []
    for t in time:
        ref = pd.Timestamp(year=2001, month=t.month, day=t.day)
        doy_nl.append(ref.dayofyear)

    doy_nl = np.array(doy_nl, dtype=np.int16)
    ds = ds.assign_coords(doy_nl=("time", doy_nl))

    return ds


def detrend_1d(y):
    """
    对单个格点的完整时间序列做线性去趋势。

    输入：
      y(time)

    输出：
      y - linear_trend

    注意：
      输出不是 anomaly，只是去掉线性趋势后的原场残差。
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)

    valid = np.isfinite(y)

    if valid.sum() == 0:
        return np.full(y.shape, np.nan, dtype=np.float32)

    if valid.sum() < 3:
        out = y - np.nanmean(y)
        return out.astype(np.float32)

    coef = np.polyfit(x[valid], y[valid], 1)
    trend = coef[0] * x + coef[1]
    out = y - trend

    return out.astype(np.float32)


def quick_stats(da, name, compute=True):
    """
    打印 DataArray 的基本统计。
    """
    print(f"\n--- {name} stats ---")
    print("shape:", da.shape)
    print("dims :", da.dims)

    if compute:
        print("NaN count:", int(da.isnull().sum().compute().values))
        print("mean:", float(da.mean(skipna=True).compute().values))
        print("std :", float(da.std(skipna=True).compute().values))
        print("min :", float(da.min(skipna=True).compute().values))
        print("max :", float(da.max(skipna=True).compute().values))
    else:
        print("NaN count:", int(da.isnull().sum().values))
        print("mean:", float(da.mean(skipna=True).values))
        print("std :", float(da.std(skipna=True).values))
        print("min :", float(da.min(skipna=True).values))
        print("max :", float(da.max(skipna=True).values))


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


# ============================================================
# 3. read input
# ============================================================
print("=" * 100)
print("READ INPUT")
print("=" * 100)

ds = xr.open_dataset(IN_FILE, chunks=CHUNKS_READ)

if "z" not in ds.variables:
    raise ValueError(f"Variable z not found. Variables: {list(ds.variables)}")

z = ds["z"]

print(ds)
print("\nInput shape:", z.shape)
print("Input chunks:", z.chunks)
print("time:", str(ds.time.values[0]), "to", str(ds.time.values[-1]))
print("lat :", float(ds.lat.min()), "to", float(ds.lat.max()), "n =", ds.sizes["lat"])
print("lon :", float(ds.lon.min()), "to", float(ds.lon.max()), "n =", ds.sizes["lon"])

time = pd.to_datetime(ds["time"].values)
feb29_count = int(((time.month == 2) & (time.day == 29)).sum())
duplicate_count = int(pd.Index(time).duplicated().sum())

print("Feb29 count:", feb29_count)
print("Duplicate time count:", duplicate_count)

if feb29_count != 0:
    raise ValueError("Input still contains Feb 29. Stop.")

if duplicate_count != 0:
    raise ValueError("Input contains duplicate time values. Stop.")

quick_stats(z, "input z", compute=True)


# ============================================================
# 4. detrend gridpoint by gridpoint
# ============================================================
print("\n" + "=" * 100)
print("DETREND")
print("=" * 100)

# 关键修正：
# time 是 detrend_1d 的 core dimension，必须是单个 chunk。
# 空间维度分块，避免一次性把全部格点塞进内存。
z_for_detrend = z.chunk(CHUNKS_DETREND)

print("Chunks for detrending:")
print(z_for_detrend.chunks)

z_detrended = xr.apply_ufunc(
    detrend_1d,
    z_for_detrend,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.float32],
)

# apply_ufunc 输出维度可能变为 lat, lon, time
z_detrended = z_detrended.transpose("time", "lat", "lon")
z_detrended.name = "z_detrended"

z_detrended.attrs["long_name"] = "Detrended 500 hPa geopotential height"
z_detrended.attrs["units"] = "gpm"
z_detrended.attrs["detrend_method"] = (
    "Linear trend removed independently at each grid point over 1981-2024."
)

ds_detrended = z_detrended.to_dataset(name="z_detrended")
ds_detrended.attrs["source"] = IN_FILE
ds_detrended.attrs["period"] = "1981-2024"
ds_detrended.attrs["domain"] = "170E-240E, 35N-90N"
ds_detrended.attrs["grid"] = "1 degree"
ds_detrended.attrs["processing"] = "Linear detrending at each grid point."

print("\nWriting detrended file:")
print(OUT_DETRENDED)

encoding_det = {
    "z_detrended": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(np.nan),
    }
}

safe_remove(OUT_DETRENDED)

with ProgressBar():
    ds_detrended.to_netcdf(OUT_DETRENDED, encoding=encoding_det)

print("[OK] detrended file saved.")

# 释放部分对象
del z_detrended, ds_detrended, z_for_detrend
gc.collect()


# ============================================================
# 5. reopen detrended for climatology/anomaly
# ============================================================
print("\n" + "=" * 100)
print("REOPEN DETRENDED")
print("=" * 100)

ds_det = xr.open_dataset(OUT_DETRENDED, chunks=CHUNKS_REOPEN)
ds_det = add_noleap_doy(ds_det)

zd = ds_det["z_detrended"]

print(ds_det)
quick_stats(zd, "z_detrended", compute=True)

# 检查 no-leap doy
doy_vals = ds_det["doy_nl"].values

print("\nNOLEAP DOY check:")
print("unique doy count:", len(np.unique(doy_vals)))
print("doy min:", int(doy_vals.min()))
print("doy max:", int(doy_vals.max()))

if len(np.unique(doy_vals)) != 365:
    raise ValueError("doy_nl does not have 365 unique values.")

if int(doy_vals.min()) != 1 or int(doy_vals.max()) != 365:
    raise ValueError("doy_nl range is not 1–365.")


# ============================================================
# 6. compute 1981–2020 no-leap climatology
# ============================================================
print("\n" + "=" * 100)
print("COMPUTE 1981-2020 NOLEAP CLIMATOLOGY")
print("=" * 100)

zd_clim_period = zd.sel(time=slice(CLIM_START, CLIM_END))
doy_clim_period = ds_det["doy_nl"].sel(time=slice(CLIM_START, CLIM_END))

time_clim = pd.to_datetime(zd_clim_period["time"].values)

print("Clim period start:", time_clim[0])
print("Clim period end  :", time_clim[-1])
print("Clim period days :", len(time_clim))
print("Expected days    :", 40 * 365)

if len(time_clim) != 40 * 365:
    raise ValueError("Climatology period length is not 40*365.")

# groupby no-leap day-of-year
z_clim = zd_clim_period.groupby(doy_clim_period).mean("time")

# groupby 后维度名在不同 xarray 版本中可能不同
if "group" in z_clim.dims:
    z_clim = z_clim.rename({"group": "doy"})
elif "doy_nl" in z_clim.dims:
    z_clim = z_clim.rename({"doy_nl": "doy"})

z_clim = z_clim.assign_coords(doy=np.arange(1, 366, dtype=np.int16))
z_clim = z_clim.transpose("doy", "lat", "lon")
z_clim = z_clim.astype("float32")
z_clim.name = "z_clim"

z_clim.attrs["long_name"] = "NOLEAP daily climatology of detrended 500 hPa geopotential height"
z_clim.attrs["units"] = "gpm"
z_clim.attrs["climatology_period"] = "1981-2020"
z_clim.attrs["note"] = "Climatology computed from linearly detrended Z500, no-leap calendar."

ds_clim = z_clim.to_dataset(name="z_clim")
ds_clim.attrs["source"] = OUT_DETRENDED
ds_clim.attrs["climatology_period"] = "1981-2020"
ds_clim.attrs["domain"] = "170E-240E, 35N-90N"
ds_clim.attrs["calendar"] = "no-leap"
ds_clim.attrs["grid"] = "1 degree"

print("\nWriting climatology file:")
print(OUT_CLIM)

encoding_clim = {
    "z_clim": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(np.nan),
    }
}

safe_remove(OUT_CLIM)

with ProgressBar():
    ds_clim.to_netcdf(OUT_CLIM, encoding=encoding_clim)

print("[OK] climatology file saved.")
quick_stats(z_clim, "z_clim", compute=True)

del ds_clim
gc.collect()


# ============================================================
# 7. compute anomaly
# ============================================================
print("\n" + "=" * 100)
print("COMPUTE ANOMALY")
print("=" * 100)

# 为了 groupby 对齐，clim 维度改回 doy_nl
z_clim_for_anom = z_clim.rename({"doy": "doy_nl"})

z_anom = zd.groupby(ds_det["doy_nl"]) - z_clim_for_anom
z_anom = z_anom.astype("float32")
z_anom.name = "z"

z_anom.attrs["long_name"] = "Z500 anomaly relative to 1981-2020 NOLEAP climatology after detrending"
z_anom.attrs["units"] = "gpm"
z_anom.attrs["climatology_period"] = "1981-2020"
z_anom.attrs["detrend"] = "Linear trend removed at each grid point over 1981-2024 before climatology."

ds_anom = z_anom.to_dataset(name="z")
ds_anom.attrs["source_detrended"] = OUT_DETRENDED
ds_anom.attrs["source_climatology"] = OUT_CLIM
ds_anom.attrs["period"] = "1981-2024"
ds_anom.attrs["climatology_period"] = "1981-2020"
ds_anom.attrs["domain"] = "170E-240E, 35N-90N"
ds_anom.attrs["calendar"] = "no-leap"
ds_anom.attrs["grid"] = "1 degree"

print("\nWriting anomaly file:")
print(OUT_ANOM)

encoding_anom = {
    "z": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(np.nan),
    }
}

safe_remove(OUT_ANOM)

with ProgressBar():
    ds_anom.to_netcdf(OUT_ANOM, encoding=encoding_anom)

print("[OK] anomaly file saved.")
quick_stats(z_anom, "z_anom", compute=True)

del z_anom, ds_anom
gc.collect()


# ============================================================
# 8. final reopen check
# ============================================================
print("\n" + "=" * 100)
print("FINAL REOPEN CHECK")
print("=" * 100)

ds_check_det = xr.open_dataset(OUT_DETRENDED)
ds_check_clim = xr.open_dataset(OUT_CLIM)
ds_check_anom = xr.open_dataset(OUT_ANOM)

print("\nDetrended:")
print(ds_check_det)

print("\nClimatology:")
print(ds_check_clim)

print("\nAnomaly:")
print(ds_check_anom)

det = ds_check_det["z_detrended"]
clm = ds_check_clim["z_clim"]
anm = ds_check_anom["z"]

det_nan = int(det.isnull().sum().values)
clm_nan = int(clm.isnull().sum().values)
anm_nan = int(anm.isnull().sum().values)

det_mean = float(det.mean(skipna=True).values)
clm_mean = float(clm.mean(skipna=True).values)
anm_mean = float(anm.mean(skipna=True).values)

det_min = float(det.min(skipna=True).values)
det_max = float(det.max(skipna=True).values)

clm_min = float(clm.min(skipna=True).values)
clm_max = float(clm.max(skipna=True).values)

anm_min = float(anm.min(skipna=True).values)
anm_max = float(anm.max(skipna=True).values)

check_lines = []
check_lines.append("DETREND / CLIMATOLOGY / ANOMALY CHECK")
check_lines.append("=" * 80)
check_lines.append(f"Input: {IN_FILE}")
check_lines.append(f"Detrended: {OUT_DETRENDED}")
check_lines.append(f"Climatology: {OUT_CLIM}")
check_lines.append(f"Anomaly: {OUT_ANOM}")
check_lines.append("")
check_lines.append("Basic settings:")
check_lines.append("Input period: 1981-2024")
check_lines.append("Climatology period: 1981-2020")
check_lines.append("Domain: 170E-240E, 35N-90N")
check_lines.append("Calendar: no-leap")
check_lines.append("Grid: 1 degree")
check_lines.append("")
check_lines.append("Shapes:")
check_lines.append(f"Detrended z_detrended: {det.shape}")
check_lines.append(f"Climatology z_clim: {clm.shape}")
check_lines.append(f"Anomaly z: {anm.shape}")
check_lines.append("")
check_lines.append("NaN counts:")
check_lines.append(f"Detrended: {det_nan}")
check_lines.append(f"Climatology: {clm_nan}")
check_lines.append(f"Anomaly: {anm_nan}")
check_lines.append("")
check_lines.append("Means:")
check_lines.append(f"Detrended mean: {det_mean}")
check_lines.append(f"Climatology mean: {clm_mean}")
check_lines.append(f"Anomaly mean: {anm_mean}")
check_lines.append("")
check_lines.append("Ranges:")
check_lines.append(f"Detrended min/max: {det_min} / {det_max}")
check_lines.append(f"Climatology min/max: {clm_min} / {clm_max}")
check_lines.append(f"Anomaly min/max: {anm_min} / {anm_max}")

with open(OUT_CHECK, "w", encoding="utf-8") as f:
    f.write("\n".join(check_lines))

print("\n".join(check_lines))
print("\nSaved check file:")
print(OUT_CHECK)

# 基础硬检查
if det.shape != (16060, 56, 71):
    raise ValueError("Detrended shape is not expected.")

if clm.shape != (365, 56, 71):
    raise ValueError("Climatology shape is not expected.")

if anm.shape != (16060, 56, 71):
    raise ValueError("Anomaly shape is not expected.")

if det_nan != 0 or clm_nan != 0 or anm_nan != 0:
    raise ValueError("NaN found in detrended/clim/anom output.")

if not (-1.0 < anm_mean < 1.0):
    print("[WARNING] anomaly mean is not close to 0. Please inspect.")

print("\n[OK] detrend / climatology / anomaly pipeline completed.")

# close
ds.close()
ds_det.close()
ds_check_det.close()
ds_check_clim.close()
ds_check_anom.close()

gc.collect()

print("\nDone.")