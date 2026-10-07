import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. settings
# ============================================================
IN_FILE = (
    "/path/to/data/ABH_170E240E_1981_2024/"
    "ERA5_Z500_daily_170E240E_35N90N_1981_2024_merged_gpm_no_leap.nc"
)

OUT_DIR = "/path/to/data/ABH_170E240E_1981_2024/events_ONDJF"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_INSTANT_MASK = os.path.join(
    OUT_DIR,
    "ERA5_ABH_instant_mask_ONDJF_170E-240E_50N-75N_1981_2024.nc"
)

OUT_SPATIAL_MASK = os.path.join(
    OUT_DIR,
    "ERA5_ABH_spatial_mask_ONDJF_170E-240E_50N-75N_1981_2024.nc"
)

OUT_EVENT_ID = os.path.join(
    OUT_DIR,
    "ERA5_ABH_event_id_daily_ONDJF_170E-240E_50N-75N_1981_2024.nc"
)

OUT_EVENTS_CSV = os.path.join(
    OUT_DIR,
    "ERA5_ABH_events_ONDJF_170E-240E_50N-75N_1981_2024.csv"
)

OUT_QUICK_CHECK = os.path.join(
    OUT_DIR,
    "ERA5_ABH_events_ONDJF_170E-240E_50N-75N_1981_2024_quick_check.txt"
)

# 识别核心区域
CORE_LAT_MIN = 50
CORE_LAT_MAX = 75
CORE_LON_MIN = 170
CORE_LON_MAX = 240

# 梯度判据
GHGN_THRESHOLD = -10.0
GHGS_THRESHOLD = 0.0

# 事件持续时间阈值
MIN_DURATION = 5

# 空间筛选阈值，沿用上一版
MIN_LON_SPAN_DEG = 15.0
MIN_LAT_SPAN_DEG = 5.0
MIN_LON_SPAN_GRID = 10

# ONDJF: Oct, Nov, Dec, Jan, Feb
SEASON_NAME = "ONDJF"
TARGET_MONTHS = [10, 11, 12, 1, 2]

# 8-neighbor connected component
STRUCTURE_8 = np.ones((3, 3), dtype=int)


# ============================================================
# 2. helper functions
# ============================================================
def calc_winter_year(times):
    """
    ONDJF winter year:
      Oct/Nov/Dec -> current year
      Jan/Feb -> previous year
    """
    times = pd.to_datetime(times)
    years = times.year
    months = times.month
    winter_year = np.where(np.isin(months, [10, 11, 12]), years, years - 1)
    return winter_year


def connected_spatial_filter(mask_2d, lats, lons):
    """
    对单日 instant mask 做空间连通区域筛选。

    输入:
      mask_2d: bool array, shape=(lat, lon)

    保留满足条件的连通区域:
      lon_span >= MIN_LON_SPAN_DEG
      lat_span >= MIN_LAT_SPAN_DEG
      lon_grid_span >= MIN_LON_SPAN_GRID
    """
    if not np.any(mask_2d):
        return np.zeros_like(mask_2d, dtype=bool)

    labeled, nlab = label(mask_2d, structure=STRUCTURE_8)
    out = np.zeros_like(mask_2d, dtype=bool)

    for lab_id in range(1, nlab + 1):
        yy, xx = np.where(labeled == lab_id)

        if len(yy) == 0:
            continue

        comp_lats = lats[yy]
        comp_lons = lons[xx]

        lat_span = float(comp_lats.max() - comp_lats.min())
        lon_span = float(comp_lons.max() - comp_lons.min())
        lon_grid_span = int(xx.max() - xx.min() + 1)

        if (
            lon_span >= MIN_LON_SPAN_DEG
            and lat_span >= MIN_LAT_SPAN_DEG
            and lon_grid_span >= MIN_LON_SPAN_GRID
        ):
            out[labeled == lab_id] = True

    return out


def build_events_from_daily_mask(spatial_mask, times):
    """
    把 daily spatial mask 转成事件。

    修正版事件连接规则：
      - 某天只要识别区域内存在 blocking grid，就算 blocking day；
      - 连续 blocking days 构成候选事件；
      - “连续”必须满足真实日期相差 1 天；
      - 防止 Feb 末和 Oct 初在 ONDJF 子序列中被误连；
      - duration >= MIN_DURATION 保留。
    """
    times = pd.to_datetime(times)
    blocked_day = spatial_mask.reshape(spatial_mask.shape[0], -1).any(axis=1)

    events = []
    event_id_daily = np.zeros_like(spatial_mask, dtype=np.int32)

    current_indices = []
    event_id = 0

    def close_current_event(indices, event_id):
        """
        关闭当前候选事件。若 duration >= MIN_DURATION，则保存。
        """
        if len(indices) == 0:
            return event_id

        duration = len(indices)

        if duration >= MIN_DURATION:
            event_id += 1
            idx = np.array(indices, dtype=int)

            event_id_daily[idx, :, :] = np.where(
                spatial_mask[idx, :, :],
                event_id,
                event_id_daily[idx, :, :]
            )

            events.append({
                "event_id": event_id,
                "start_index": int(indices[0]),
                "end_index": int(indices[-1]),
                "start_date": pd.to_datetime(times[indices[0]]),
                "end_date": pd.to_datetime(times[indices[-1]]),
                "duration_days": int(duration),
            })

        return event_id

    for i, is_blocked in enumerate(blocked_day):
        if not is_blocked:
            event_id = close_current_event(current_indices, event_id)
            current_indices = []
            continue

        if len(current_indices) == 0:
            current_indices = [i]
        else:
            prev_i = current_indices[-1]
            is_next_calendar_day = (times[i] - times[prev_i]).days == 1

            if is_next_calendar_day:
                current_indices.append(i)
            else:
                # ONDJF 子序列中相邻，但真实日期不连续，必须断开
                event_id = close_current_event(current_indices, event_id)
                current_indices = [i]

    # 收尾
    event_id = close_current_event(current_indices, event_id)

    return events, event_id_daily, blocked_day


def event_spatial_summary(event_id_daily, spatial_mask, events, lats, lons):
    """
    给每个事件补充空间统计信息：
      - blocked_days_check
      - max_area_grids
      - mean_area_grids
      - lat_min/max
      - lon_min/max
    """
    rows = []

    for ev in events:
        eid = ev["event_id"]
        idx0 = ev["start_index"]
        idx1 = ev["end_index"]

        sub_event_id = event_id_daily[idx0:idx1 + 1, :, :] == eid
        daily_area = sub_event_id.reshape(sub_event_id.shape[0], -1).sum(axis=1)

        yy, xx = np.where(sub_event_id.any(axis=0))

        if len(yy) > 0:
            lat_min = float(lats[yy].min())
            lat_max = float(lats[yy].max())
            lon_min = float(lons[xx].min())
            lon_max = float(lons[xx].max())
            lat_span = lat_max - lat_min
            lon_span = lon_max - lon_min
        else:
            lat_min = lat_max = lon_min = lon_max = lat_span = lon_span = np.nan

        row = ev.copy()
        row.update({
            "blocked_days_check": int((daily_area > 0).sum()),
            "max_area_grids": int(daily_area.max()) if len(daily_area) > 0 else 0,
            "mean_area_grids": float(daily_area.mean()) if len(daily_area) > 0 else np.nan,
            "event_lat_min": lat_min,
            "event_lat_max": lat_max,
            "event_lon_min": lon_min,
            "event_lon_max": lon_max,
            "event_lat_span": lat_span,
            "event_lon_span": lon_span,
        })

        rows.append(row)

    return rows


def check_event_duration_consistency(events_df):
    """
    检查事件表中的 duration_days 是否等于自然日期差。
    """
    if len(events_df) == 0:
        return pd.DataFrame()

    check = events_df.copy()
    check["calendar_duration"] = (
        pd.to_datetime(check["end_date"]) - pd.to_datetime(check["start_date"])
    ).dt.days + 1
    check["duration_diff"] = check["calendar_duration"] - check["duration_days"]

    bad = check[check["duration_diff"] != 0].copy()
    return bad


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


# ============================================================
# 3. read input
# ============================================================
print("=" * 100)
print("READ INPUT")
print("=" * 100)

ds = xr.open_dataset(IN_FILE)

if "z" not in ds.variables:
    raise ValueError(f"Variable z not found. Variables: {list(ds.variables)}")

z = ds["z"]

print(ds)
print("\nInput z shape:", z.shape)
print("Input time:", str(ds.time.values[0]), "to", str(ds.time.values[-1]))
print("Input lat :", float(ds.lat.min()), "to", float(ds.lat.max()), "n =", ds.sizes["lat"])
print("Input lon :", float(ds.lon.min()), "to", float(ds.lon.max()), "n =", ds.sizes["lon"])
print("Input NaN:", int(z.isnull().sum().values))
print("Input mean:", float(z.mean(skipna=True).values))


# ============================================================
# 4. select ONDJF
# ============================================================
print("\n" + "=" * 100)
print(f"SELECT {SEASON_NAME}")
print("=" * 100)

time_index = pd.to_datetime(ds["time"].values)
month = time_index.month

djf_mask = np.isin(month, TARGET_MONTHS)

z_djf = z.sel(time=djf_mask)
time_djf = pd.to_datetime(z_djf["time"].values)

print("Total days:", ds.sizes["time"])
print(f"{SEASON_NAME} days  :", z_djf.sizes["time"])
print(f"{SEASON_NAME} start :", time_djf[0])
print(f"{SEASON_NAME} end   :", time_djf[-1])
print(f"{SEASON_NAME} months:", sorted(set(time_djf.month)))

winter_year_djf = calc_winter_year(time_djf)

print(f"{SEASON_NAME} winter_year min:", winter_year_djf.min())
print(f"{SEASON_NAME} winter_year max:", winter_year_djf.max())

# 检查 ONDJF 时间轴真实日期跳跃位置
time_diffs = np.diff(time_djf).astype("timedelta64[D]").astype(int)
jump_idx = np.where(time_diffs != 1)[0]

print(f"Number of non-consecutive jumps in {SEASON_NAME} time axis:", len(jump_idx))
if len(jump_idx) > 0:
    print("First 10 jumps:")
    for k in jump_idx[:10]:
        print(f"  {time_djf[k]} -> {time_djf[k+1]}, diff={time_diffs[k]} days")


# ============================================================
# 5. compute GHGS / GHGN
# ============================================================
print("\n" + "=" * 100)
print("COMPUTE GRADIENT MASK")
print("=" * 100)

core_lats = np.arange(CORE_LAT_MIN, CORE_LAT_MAX + 1, 1.0)
core_lons = np.arange(CORE_LON_MIN, CORE_LON_MAX + 1, 1.0)

z_core = z_djf.sel(lat=core_lats, lon=core_lons)
z_south = z_djf.sel(lat=core_lats - 15.0, lon=core_lons)
z_north = z_djf.sel(lat=core_lats + 15.0, lon=core_lons)

# 让 south/north 的 lat 坐标与 core_lats 对齐
z_south = z_south.assign_coords(lat=core_lats)
z_north = z_north.assign_coords(lat=core_lats)

ghgs = (z_core - z_south) / 15.0
ghgn = (z_north - z_core) / 15.0

instant_mask_da = ((ghgs > GHGS_THRESHOLD) & (ghgn < GHGN_THRESHOLD))
instant_mask_da.name = "instant_blocking_mask"

print("Core shape:", z_core.shape)
print("GHGS mean:", float(ghgs.mean(skipna=True).values))
print("GHGN mean:", float(ghgn.mean(skipna=True).values))
print("Instant mask true count:", int(instant_mask_da.sum().values))

instant_mask = instant_mask_da.values.astype(bool)
times = time_djf
lats_core = instant_mask_da["lat"].values
lons_core = instant_mask_da["lon"].values


# ============================================================
# 6. spatial connected-component filtering
# ============================================================
print("\n" + "=" * 100)
print("SPATIAL FILTER")
print("=" * 100)

nt, nlat, nlon = instant_mask.shape
spatial_mask = np.zeros_like(instant_mask, dtype=bool)

for it in tqdm(range(nt), desc="Spatial filter daily masks"):
    spatial_mask[it, :, :] = connected_spatial_filter(
        instant_mask[it, :, :],
        lats_core,
        lons_core
    )

instant_blocked_days = int(instant_mask.reshape(nt, -1).any(axis=1).sum())
spatial_blocked_days = int(spatial_mask.reshape(nt, -1).any(axis=1).sum())

print(f"Total {SEASON_NAME} days:", nt)
print("Instant blocking days:", instant_blocked_days)
print("Spatial blocking days:", spatial_blocked_days)
print("Instant true grids:", int(instant_mask.sum()))
print("Spatial true grids:", int(spatial_mask.sum()))


# ============================================================
# 7. build events
# ============================================================
print("\n" + "=" * 100)
print("BUILD EVENTS")
print("=" * 100)

events, event_id_daily, blocked_day = build_events_from_daily_mask(
    spatial_mask,
    times
)

event_rows = event_spatial_summary(
    event_id_daily,
    spatial_mask,
    events,
    lats_core,
    lons_core
)

events_df = pd.DataFrame(event_rows)

if len(events_df) > 0:
    events_df["start_date"] = pd.to_datetime(events_df["start_date"])
    events_df["end_date"] = pd.to_datetime(events_df["end_date"])

    events_df["start_year"] = events_df["start_date"].dt.year
    events_df["start_month"] = events_df["start_date"].dt.month

    events_df["winter_year"] = np.where(
        events_df["start_month"].isin([10, 11, 12]),
        events_df["start_year"],
        events_df["start_year"] - 1
    )

    events_df = events_df.sort_values("start_date").reset_index(drop=True)

    # 重新排序后，为了 event_id 连续并与 event_id_daily 对应稳定，不修改 event_id。
    # event_id_daily 中的 event_id 仍然与原 event_id 一致。

print("Events retained:", len(events_df))

if len(events_df) > 0:
    print(events_df.head())
    print(events_df.tail())

    print("\nDuration stats:")
    print(events_df["duration_days"].describe())

    print("\nEvents by start month:")
    print(events_df["start_month"].value_counts().sort_index())

    print("\nEvents by winter_year:")
    print(events_df["winter_year"].value_counts().sort_index())

    print("\nBlocked days consistency:")
    diff_blocked = events_df["blocked_days_check"] - events_df["duration_days"]
    print(diff_blocked.value_counts().sort_index())

    bad_duration = check_event_duration_consistency(events_df)
    print("\nCalendar duration bad events:", len(bad_duration))

    if len(bad_duration) > 0:
        print(
            bad_duration[
                [
                    "event_id",
                    "start_date",
                    "end_date",
                    "duration_days",
                    "calendar_duration",
                    "duration_diff",
                    "winter_year",
                    "start_month",
                ]
            ].to_string(index=False)
        )
        raise ValueError("Some events still have calendar duration inconsistent with duration_days.")
    else:
        print("[OK] all events have calendar-consistent duration.")


# ============================================================
# 8. save masks and events
# ============================================================
print("\n" + "=" * 100)
print("SAVE OUTPUTS")
print("=" * 100)

coords = {
    "time": z_djf["time"].values,
    "lat": lats_core,
    "lon": lons_core,
}

instant_ds = xr.Dataset(
    {
        "instant_blocking_mask": (
            ("time", "lat", "lon"),
            instant_mask.astype(np.int8)
        )
    },
    coords=coords,
    attrs={
        "description": "Instantaneous ABH mask based on GHGS > 0 and GHGN < -10.",
        "domain": "170E-240E, 50N-75N",
        "season": SEASON_NAME,
        "period": "1981-2024",
        "GHGS": "(Z(phi0)-Z(phi0-15))/15",
        "GHGN": "(Z(phi0+15)-Z(phi0))/15",
        "note": f"{SEASON_NAME} time axis contains seasonal gaps; event connection uses real calendar continuity.",
    }
)

spatial_ds = xr.Dataset(
    {
        "spatial_blocking_mask": (
            ("time", "lat", "lon"),
            spatial_mask.astype(np.int8)
        )
    },
    coords=coords,
    attrs={
        "description": "Spatially filtered ABH mask after connected-component filtering.",
        "domain": "170E-240E, 50N-75N",
        "season": SEASON_NAME,
        "period": "1981-2024",
        "spatial_filter": (
            f"lon_span >= {MIN_LON_SPAN_DEG}, "
            f"lat_span >= {MIN_LAT_SPAN_DEG}, "
            f"lon_grid_span >= {MIN_LON_SPAN_GRID}"
        ),
        "note": "Event connection requires real calendar continuity to avoid Feb-Oct false connection.",
    }
)

event_id_ds = xr.Dataset(
    {
        "event_id": (
            ("time", "lat", "lon"),
            event_id_daily.astype(np.int32)
        )
    },
    coords=coords,
    attrs={
        "description": "Daily ABH event id on spatial blocking mask grids. 0 means no event.",
        "domain": "170E-240E, 50N-75N",
        "season": SEASON_NAME,
        "period": "1981-2024",
        "min_duration_days": str(MIN_DURATION),
        "event_connection": "Consecutive blocking days with real calendar day difference equal to 1.",
    }
)

encoding_mask = {
    "instant_blocking_mask": {
        "zlib": True,
        "complevel": 4,
        "dtype": "int8",
    }
}

encoding_spatial = {
    "spatial_blocking_mask": {
        "zlib": True,
        "complevel": 4,
        "dtype": "int8",
    }
}

encoding_event = {
    "event_id": {
        "zlib": True,
        "complevel": 4,
        "dtype": "int32",
    }
}

# 先删旧文件，避免权限/覆盖问题
for p in [OUT_INSTANT_MASK, OUT_SPATIAL_MASK, OUT_EVENT_ID, OUT_EVENTS_CSV, OUT_QUICK_CHECK]:
    safe_remove(p)

instant_ds.to_netcdf(OUT_INSTANT_MASK, encoding=encoding_mask)
spatial_ds.to_netcdf(OUT_SPATIAL_MASK, encoding=encoding_spatial)
event_id_ds.to_netcdf(OUT_EVENT_ID, encoding=encoding_event)

events_df.to_csv(OUT_EVENTS_CSV, index=False, encoding="utf-8-sig")

print("Saved instant mask:", OUT_INSTANT_MASK)
print("Saved spatial mask:", OUT_SPATIAL_MASK)
print("Saved event id    :", OUT_EVENT_ID)
print("Saved events csv  :", OUT_EVENTS_CSV)


# ============================================================
# 9. quick check text
# ============================================================
print("\n" + "=" * 100)
print("QUICK CHECK")
print("=" * 100)

lines = []
lines.append("ABH EVENT DETECTION QUICK CHECK")
lines.append("=" * 80)
lines.append(f"Input file: {IN_FILE}")
lines.append("Period: 1981-2024")
lines.append(f"Season: {SEASON_NAME}")
lines.append("Mother domain: 170E-240E, 35N-90N")
lines.append("Detection domain: 170E-240E, 50N-75N")
lines.append(f"GHGS threshold: > {GHGS_THRESHOLD}")
lines.append(f"GHGN threshold: < {GHGN_THRESHOLD}")
lines.append(f"Minimum duration: {MIN_DURATION} days")
lines.append("Event connection: blocking days must be consecutive in real calendar dates")
lines.append("")
lines.append(f"Total {SEASON_NAME} days: {nt}")
lines.append(f"Number of non-consecutive jumps in {SEASON_NAME} time axis: {len(jump_idx)}")
lines.append(f"Instant blocking days: {instant_blocked_days}")
lines.append(f"Spatial blocking days: {spatial_blocked_days}")
lines.append(f"Instant true grids: {int(instant_mask.sum())}")
lines.append(f"Spatial true grids: {int(spatial_mask.sum())}")
lines.append(f"Total events retained duration >= {MIN_DURATION}: {len(events_df)}")

if len(events_df) > 0:
    bad_duration = check_event_duration_consistency(events_df)

    lines.append("")
    lines.append("Duration stats:")
    lines.append(str(events_df["duration_days"].describe()))
    lines.append("")
    lines.append("Events by start month:")
    lines.append(str(events_df["start_month"].value_counts().sort_index()))
    lines.append("")
    lines.append("Events by winter year:")
    lines.append(str(events_df["winter_year"].value_counts().sort_index()))
    lines.append("")
    lines.append("Blocked days check minus duration_days:")
    lines.append(str((events_df["blocked_days_check"] - events_df["duration_days"]).value_counts().sort_index()))
    lines.append("")
    lines.append(f"Calendar duration bad events: {len(bad_duration)}")
    lines.append("")
    lines.append("First 10 events:")
    lines.append(str(events_df.head(10)))
    lines.append("")
    lines.append("Last 10 events:")
    lines.append(str(events_df.tail(10)))

with open(OUT_QUICK_CHECK, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines[:35]))
print("\nSaved quick check:", OUT_QUICK_CHECK)

ds.close()

print("\nDone.")
