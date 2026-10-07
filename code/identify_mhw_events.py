import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# =========================================================
# 1. 基本参数
# =========================================================
DATA_DIR = r"E:\NOAA\noaa.oisst.v2.highres"
WORK_DIR = r"E:\NOAA\MHW"
os.makedirs(WORK_DIR, exist_ok=True)

# 目标年份
YEAR_START = 1981
YEAR_END   = 2024

# NEP 区域
LAT_MIN, LAT_MAX = 40.0, 50.0
LON_MIN_W, LON_MAX_W = -160.0, -135.0

# Hobday 事件参数
MIN_DURATION = 5
MAX_GAP = 2

# 已有基准文件
CLIM_FILE = os.path.join(WORK_DIR, "NEP_MHW_clim_1983-2012_smoothed30.nc")
Q90_FILE  = os.path.join(WORK_DIR, "NEP_MHW_q90_1983-2012_smoothed30.nc")

# 输出文件
DAILY_OUT = os.path.join(WORK_DIR, "NEP_MHW_daily_1981-2024_area_mean_smoothed30.nc")
EVENT_OUT = os.path.join(WORK_DIR, "NEP_MHW_events_1981-2024_area_mean_smoothed30.csv")


# =========================================================
# 2. 读取 1981–2024 OISST 并做区域平均
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

# 经度转为 -180~180
if float(ds.lon.max()) > 180:
    lon_new = (((ds.lon + 180) % 360) - 180)
    ds = ds.assign_coords(lon=lon_new)
    ds = ds.sortby("lon")

# 裁剪 NEP 区域
ds_nep = ds.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN_W, LON_MAX_W)
)

print("\n裁剪后区域：")
print(ds_nep)

# 去掉 2 月 29 日
time = ds_nep["time"]
is_feb29 = (time.dt.month == 2) & (time.dt.day == 29)
ds_nep = ds_nep.sel(time=~is_feb29)

print("\n去掉 2 月 29 日后：")
print(ds_nep)
print("time length =", ds_nep.sizes["time"])

# 区域面积加权平均
weights = np.cos(np.deg2rad(ds_nep["lat"]))
sst_area_mean = ds_nep["sst"].weighted(weights).mean(dim=("lat", "lon"))

print("\n区域平均后的 SST：")
print(sst_area_mean)

print("\n正在将区域平均时间序列载入内存...")
sst_area_mean = sst_area_mean.compute()


# =========================================================
# 3. 读取 climatology 和 q90
# =========================================================
ds_clim = xr.open_dataset(CLIM_FILE)
ds_q90  = xr.open_dataset(Q90_FILE)

clim_doy = ds_clim["sst_clim"]   # (dayofyear,)
q90_doy  = ds_q90["sst_q90"]     # (dayofyear,)

print("\n已读取基准文件：")
print(CLIM_FILE)
print(Q90_FILE)

# 按 noleap dayofyear 映射到完整时间序列
time_full = sst_area_mean["time"]

month = time_full.dt.month.values
day = time_full.dt.day.values
year = time_full.dt.year.values
doy_raw = time_full.dt.dayofyear.values

# 判断闰年
is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)

# 去闰日后的 dayofyear
doy_nl = doy_raw.copy()
doy_nl[is_leap & (month > 2)] -= 1

print("noleap doy range:", doy_nl.min(), "~", doy_nl.max())

doy_nl_da = xr.DataArray(
    doy_nl,
    coords={"time": time_full},
    dims="time",
    name="dayofyear"
)

sst_clim_full = clim_doy.sel(dayofyear=doy_nl_da)
sst_q90_full  = q90_doy.sel(dayofyear=doy_nl_da)

# 计算 anomaly 和 MHW 日掩膜
sst_anomaly = sst_area_mean - sst_clim_full
is_mhw_day = (sst_area_mean > sst_q90_full)

# 保存日尺度文件
ds_daily = xr.Dataset(
    {
        "sst": sst_area_mean.astype(np.float32),
        "sst_clim": sst_clim_full.astype(np.float32),
        "sst_q90": sst_q90_full.astype(np.float32),
        "sst_anomaly": sst_anomaly.astype(np.float32),
        "is_mhw_day": is_mhw_day.astype(np.int8),
    }
)

ds_daily["sst"].attrs["long_name"] = "Area-mean SST over NE Pacific"
ds_daily["sst"].attrs["units"] = "degC"

ds_daily["sst_clim"].attrs["long_name"] = "Daily climatological mean SST"
ds_daily["sst_clim"].attrs["units"] = "degC"

ds_daily["sst_q90"].attrs["long_name"] = "Daily 90th percentile SST threshold"
ds_daily["sst_q90"].attrs["units"] = "degC"

ds_daily["sst_anomaly"].attrs["long_name"] = "SST anomaly relative to climatology"
ds_daily["sst_anomaly"].attrs["units"] = "degC"

ds_daily["is_mhw_day"].attrs["long_name"] = "Marine heatwave day flag (1=yes, 0=no)"

ds_daily.attrs["title"] = "NEP area-mean daily MHW diagnostics"
ds_daily.attrs["region"] = "40-50N, 135-160W"
ds_daily.attrs["baseline_period"] = "1983-2012"
ds_daily.attrs["mhw_definition"] = "Hobday et al. style; daily SST > q90"
ds_daily.attrs["intensity_definition"] = "sst_anomaly = SST - climatology"
ds_daily.attrs["note"] = "Leap day removed"

encoding = {
    "sst": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "sst_clim": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "sst_q90": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "sst_anomaly": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "is_mhw_day": {"zlib": True, "complevel": 4, "dtype": "int8"},
}

print("\n正在保存日尺度文件...")
ds_daily.to_netcdf(DAILY_OUT, encoding=encoding)
print("已保存：", DAILY_OUT)


# =========================================================
# 4. 识别 MHW 事件
# =========================================================
def detect_mhw_events(time, sst, clim, q90, anomaly,
                      min_duration=5, max_gap=2):
    """
    基于 1D 区域平均序列识别 MHW 事件
    判定：sst > q90
    强度：anomaly = sst - clim
    允许中间最多 max_gap 天低于阈值
    """
    time = pd.to_datetime(time)
    sst = np.asarray(sst, dtype=float)
    clim = np.asarray(clim, dtype=float)
    q90 = np.asarray(q90, dtype=float)
    anomaly = np.asarray(anomaly, dtype=float)

    above = sst > q90
    n = len(sst)

    events = []
    event_id = np.full(n, np.nan)

    i = 0
    eid = 0

    while i < n:
        if above[i]:
            start = i
            last_hot = i
            gap_count = 0
            j = i + 1

            while j < n:
                if above[j]:
                    last_hot = j
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > max_gap:
                        break
                j += 1

            end = last_hot
            duration = end - start + 1

            if duration >= min_duration:
                eid += 1
                event_id[start:end+1] = eid

                sl = slice(start, end + 1)
                anom_evt = anomaly[sl]
                sst_evt = sst[sl]
                q90_evt = q90[sl]

                peak_rel_idx = int(np.nanargmax(anom_evt))
                peak_idx = start + peak_rel_idx

                events.append({
                    "event_id": eid,
                    "start_date": time[start],
                    "end_date": time[end],
                    "peak_date": time[peak_idx],
                    "duration_days": duration,
                    "mean_intensity": float(np.nanmean(anom_evt)),
                    "max_intensity": float(np.nanmax(anom_evt)),
                    "cum_intensity": float(np.nansum(anom_evt)),
                    "mean_sst": float(np.nanmean(sst_evt)),
                    "mean_threshold_exceedance": float(np.nanmean(sst_evt - q90_evt)),
                    "start_index": int(start),
                    "end_index": int(end),
                    "peak_index": int(peak_idx),
                })

            i = end + 1
        else:
            i += 1

    return pd.DataFrame(events), event_id


print("\n开始识别 MHW 事件...")
df_evt, event_id = detect_mhw_events(
    time=ds_daily["time"].values,
    sst=ds_daily["sst"].values,
    clim=ds_daily["sst_clim"].values,
    q90=ds_daily["sst_q90"].values,
    anomaly=ds_daily["sst_anomaly"].values,
    min_duration=MIN_DURATION,
    max_gap=MAX_GAP,
)

print("总事件数 =", len(df_evt))


# =========================================================
# 5. 给日尺度文件补充 event_id
# =========================================================
ds_daily["event_id"] = xr.DataArray(
    event_id.astype(np.float32),
    coords={"time": ds_daily["time"]},
    dims=("time",)
)
ds_daily["event_id"].attrs["long_name"] = "Marine heatwave event ID for each day"

encoding["event_id"] = {"zlib": True, "complevel": 4, "dtype": "float32"}

print("\n正在更新日尺度文件（加入 event_id）...")
ds_daily.to_netcdf(DAILY_OUT, encoding=encoding)
print("已更新：", DAILY_OUT)


# =========================================================
# 6. 补充 season / year 信息
# =========================================================
def month_to_season(month):
    if month in (12, 1, 2):
        return "DJF"
    elif month in (3, 4, 5):
        return "MAM"
    elif month in (6, 7, 8):
        return "JJA"
    else:
        return "SON"

if len(df_evt) > 0:
    df_evt["start_date"] = pd.to_datetime(df_evt["start_date"])
    df_evt["end_date"] = pd.to_datetime(df_evt["end_date"])
    df_evt["peak_date"] = pd.to_datetime(df_evt["peak_date"])

    df_evt["year"] = df_evt["start_date"].dt.year
    df_evt["month"] = df_evt["start_date"].dt.month
    df_evt["season"] = df_evt["month"].apply(month_to_season)

    # 调整列顺序
    cols = [
        "event_id",
        "start_date", "end_date", "peak_date",
        "year", "month", "season",
        "duration_days",
        "mean_intensity", "max_intensity", "cum_intensity",
        "mean_sst", "mean_threshold_exceedance",
        "start_index", "end_index", "peak_index",
    ]
    df_evt = df_evt[cols]

# 保存事件表
df_evt.to_csv(EVENT_OUT, index=False, encoding="utf-8-sig")
print("\n事件表已保存：", EVENT_OUT)

print("\n全部完成。")
print("日尺度文件：", DAILY_OUT)
print("事件表：", EVENT_OUT)