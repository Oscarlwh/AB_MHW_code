import os
import numpy as np
import xarray as xr

WORK_DIR = r"E:\NOAA\MHW"

CLIM_FILE = os.path.join(WORK_DIR, "NEP_MHW_clim_1983-2012.nc")
Q90_FILE  = os.path.join(WORK_DIR, "NEP_MHW_q90_1983-2012.nc")

OUT_CLIM_SMOOTH = os.path.join(WORK_DIR, "NEP_MHW_clim_1983-2012_smoothed30.nc")
OUT_Q90_SMOOTH  = os.path.join(WORK_DIR, "NEP_MHW_q90_1983-2012_smoothed30.nc")

WINDOW = 30  # Hobday 建议的 30-day moving window


def circular_rolling_mean(arr, window):
    """
    对 1..365 的 dayofyear 序列做循环滑动平均
    让年初和年末能连起来平滑
    """
    half_left = window // 2
    half_right = window - 1 - half_left

    arr_pad = np.concatenate([arr[-half_left:], arr, arr[:half_right]])
    kernel = np.ones(window, dtype=float) / window
    arr_smooth = np.convolve(arr_pad, kernel, mode="valid")
    return arr_smooth


# 读取原始基准文件
ds_clim = xr.open_dataset(CLIM_FILE)
ds_q90 = xr.open_dataset(Q90_FILE)

clim = ds_clim["sst_clim"].values.astype(float)
q90 = ds_q90["sst_q90"].values.astype(float)
dayofyear = ds_clim["dayofyear"].values

# 平滑
clim_smooth = circular_rolling_mean(clim, WINDOW)
q90_smooth = circular_rolling_mean(q90, WINDOW)

# 保存平滑后的 climatology
ds_clim_s = xr.Dataset(
    {"sst_clim": (("dayofyear",), clim_smooth.astype(np.float32))},
    coords={"dayofyear": dayofyear}
)
ds_clim_s["sst_clim"].attrs["long_name"] = "Daily climatological mean SST for NEP MHW detection (30-day smoothed)"
ds_clim_s["sst_clim"].attrs["units"] = "degC"
ds_clim_s.attrs["title"] = "NEP daily climatology for MHW detection"
ds_clim_s.attrs["baseline_period"] = "1983-2012"
ds_clim_s.attrs["region"] = "40-50N, 135-160W"
ds_clim_s.attrs["source"] = "NOAA OISST v2.1 daily mean SST"
ds_clim_s.attrs["method"] = "11-day window climatology + 30-day circular smoothing"

# 保存平滑后的 q90
ds_q90_s = xr.Dataset(
    {"sst_q90": (("dayofyear",), q90_smooth.astype(np.float32))},
    coords={"dayofyear": dayofyear}
)
ds_q90_s["sst_q90"].attrs["long_name"] = "Daily 90th percentile SST threshold for NEP MHW detection (30-day smoothed)"
ds_q90_s["sst_q90"].attrs["units"] = "degC"
ds_q90_s.attrs["title"] = "NEP daily q90 threshold for MHW detection"
ds_q90_s.attrs["baseline_period"] = "1983-2012"
ds_q90_s.attrs["region"] = "40-50N, 135-160W"
ds_q90_s.attrs["source"] = "NOAA OISST v2.1 daily mean SST"
ds_q90_s.attrs["method"] = "11-day window percentile threshold + 30-day circular smoothing"

encoding_clim = {"sst_clim": {"zlib": True, "complevel": 4, "dtype": "float32"}}
encoding_q90  = {"sst_q90":  {"zlib": True, "complevel": 4, "dtype": "float32"}}

ds_clim_s.to_netcdf(OUT_CLIM_SMOOTH, encoding=encoding_clim)
ds_q90_s.to_netcdf(OUT_Q90_SMOOTH, encoding=encoding_q90)

print("已保存：")
print(OUT_CLIM_SMOOTH)
print(OUT_Q90_SMOOTH)