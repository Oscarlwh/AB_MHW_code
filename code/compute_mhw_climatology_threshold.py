import os
import numpy as np
import xarray as xr
from tqdm import tqdm

# =========================================================
# 1. 基本参数
# =========================================================
DATA_DIR = r"E:\NOAA\noaa.oisst.v2.highres"
OUT_DIR  = r"E:\NOAA\MHW"
os.makedirs(OUT_DIR, exist_ok=True)

# 基准期
YEAR_START = 1983
YEAR_END   = 2012

# NEP 区域（Chen 那篇用的区域）
LAT_MIN, LAT_MAX = 40.0, 50.0
LON_MIN_W, LON_MAX_W = -160.0, -135.0   # 西经，统一转到 -180~180

# 输出文件
OUT_CLIM = os.path.join(OUT_DIR, "NEP_MHW_clim_1983-2012.nc")
OUT_Q90  = os.path.join(OUT_DIR, "NEP_MHW_q90_1983-2012.nc")

# Hobday: 11-day window centered on each day
HALF_WINDOW = 5   # 前后各 5 天，加当天共 11 天


# =========================================================
# 2. 读取基准期数据
# =========================================================
years = range(YEAR_START, YEAR_END + 1)
files = [os.path.join(DATA_DIR, f"sst.day.mean.{y}.nc") for y in years]

print("即将读取以下文件：")
for f in files:
    print("  ", f)

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    parallel=True,
    chunks={"time": 365}
)

print("\n原始数据：")
print(ds)


# =========================================================
# 3. 经度转为 -180~180，并排序
# =========================================================
if float(ds.lon.max()) > 180:
    lon_new = (((ds.lon + 180) % 360) - 180)
    ds = ds.assign_coords(lon=lon_new)
    ds = ds.sortby("lon")

print("\n经度范围：", float(ds.lon.min()), "~", float(ds.lon.max()))
print("纬度范围：", float(ds.lat.min()), "~", float(ds.lat.max()))


# =========================================================
# 4. 裁剪 NEP 区域
# =========================================================
ds_nep = ds.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN_W, LON_MAX_W)
)

print("\n裁剪后区域：")
print(ds_nep)


# =========================================================
# 5. 去掉 2 月 29 日
# =========================================================
time = ds_nep["time"]
is_feb29 = (time.dt.month == 2) & (time.dt.day == 29)
ds_nep = ds_nep.sel(time=~is_feb29)

print("\n去掉 2 月 29 日后：")
print(ds_nep)
print("time length =", ds_nep.sizes["time"])


# =========================================================
# 6. 区域面积加权平均（cos(lat)）
# =========================================================
weights = np.cos(np.deg2rad(ds_nep["lat"]))
sst_area_mean = ds_nep["sst"].weighted(weights).mean(dim=("lat", "lon"))

print("\n区域平均后的 SST：")
print(sst_area_mean)

# 转成 numpy，避免 dask quantile 的兼容性问题
print("\n正在将区域平均时间序列载入内存...")
sst_area_mean = sst_area_mean.compute()

# 1D 时间序列
sst = sst_area_mean.values.astype(np.float64)
time = sst_area_mean["time"]
doy = time.dt.dayofyear.values   # 1..365

print("区域平均序列长度：", len(sst))


# =========================================================
# 7. 按 Hobday 方法计算 climatology 和 q90
#    对每个 dayofyear，取中心±5天共11天窗口
# =========================================================
clim = np.full(365, np.nan, dtype=np.float64)
q90  = np.full(365, np.nan, dtype=np.float64)

print("\n开始计算 11 天窗口 climatology 和 q90 ...")
for d in tqdm(range(1, 366), desc="Calculating daily climatology/q90", ncols=100):
    # 循环窗口，例如 doy=1 时窗口应包含 361..365 和 1..6
    window_days = [((d - 1 + offset) % 365) + 1 for offset in range(-HALF_WINDOW, HALF_WINDOW + 1)]

    mask = np.isin(doy, window_days)
    vals = sst[mask]

    clim[d - 1] = np.nanmean(vals)
    q90[d - 1]  = np.nanpercentile(vals, 90)

print("计算完成。")


# =========================================================
# 8. 保存 climatology 文件
# =========================================================
dayofyear = np.arange(1, 366)

ds_clim = xr.Dataset(
    {
        "sst_clim": (("dayofyear",), clim.astype(np.float32))
    },
    coords={
        "dayofyear": dayofyear
    }
)

ds_clim["sst_clim"].attrs["long_name"] = "Daily climatological mean SST for NEP MHW detection"
ds_clim["sst_clim"].attrs["units"] = "degC"

ds_clim.attrs["title"] = "NEP daily climatology for MHW detection"
ds_clim.attrs["baseline_period"] = f"{YEAR_START}-{YEAR_END}"
ds_clim.attrs["region"] = "40-50N, 135-160W"
ds_clim.attrs["source"] = "NOAA OISST v2.1 daily mean SST"
ds_clim.attrs["method"] = "Hobday et al. (2016): 11-day moving window climatology"
ds_clim.attrs["note"] = (
    "Area-mean SST over NE Pacific; leap day removed; "
    "for each dayofyear, climatology is calculated from all data within an 11-day window centered on that day"
)


# =========================================================
# 9. 保存 q90 文件
# =========================================================
ds_q90 = xr.Dataset(
    {
        "sst_q90": (("dayofyear",), q90.astype(np.float32))
    },
    coords={
        "dayofyear": dayofyear
    }
)

ds_q90["sst_q90"].attrs["long_name"] = "Daily 90th percentile SST threshold for NEP MHW detection"
ds_q90["sst_q90"].attrs["units"] = "degC"

ds_q90.attrs["title"] = "NEP daily q90 threshold for MHW detection"
ds_q90.attrs["baseline_period"] = f"{YEAR_START}-{YEAR_END}"
ds_q90.attrs["region"] = "40-50N, 135-160W"
ds_q90.attrs["source"] = "NOAA OISST v2.1 daily mean SST"
ds_q90.attrs["method"] = "Hobday et al. (2016): 11-day moving window percentile threshold"
ds_q90.attrs["note"] = (
    "Area-mean SST over NE Pacific; leap day removed; "
    "for each dayofyear, q90 is calculated from all data within an 11-day window centered on that day"
)


# =========================================================
# 10. 写出文件
# =========================================================
encoding_clim = {
    "sst_clim": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    }
}

encoding_q90 = {
    "sst_q90": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    }
}

print("\n正在保存 climatology 文件...")
ds_clim.to_netcdf(OUT_CLIM, encoding=encoding_clim)

print("正在保存 q90 文件...")
ds_q90.to_netcdf(OUT_Q90, encoding=encoding_q90)

print("\n全部完成。")
print("climatology 文件：", OUT_CLIM)
print("q90 文件：", OUT_Q90)