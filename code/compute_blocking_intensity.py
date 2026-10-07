import os
import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 1. paths and settings
# ============================================================
EVENT_FILE = (
    "/path/to/data/ABH_170E240E_1981_2024/events_ONDJF/"
    "ERA5_ABH_events_ONDJF_170E-240E_50N-75N_1981_2024.csv"
)

ANOM_FILE = (
    "/path/to/data/ABH_170E240E_1981_2024/anomaly/"
    "ERA5_Z500_anom_NOLEAP_1981_2024_vs1981_2020_gpm_170E240E_35N90N_detrended.nc"
)

OUT_DIR = "/path/to/data/ABH_170E240E_1981_2024/compute_blocking_intensity"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_EVENT_INTENSITY = os.path.join(
    OUT_DIR,
    "ABH_blocking_intensity_dailymax_cumulative_ONDJF_170E-240E_50N-75N_1981_2024.csv"
)

OUT_DAILY_INTENSITY = os.path.join(
    OUT_DIR,
    "ABH_daily_intensity_by_event_ONDJF_170E-240E_50N-75N_1981_2024.csv"
)

OUT_CHECK = os.path.join(
    OUT_DIR,
    "check_ABH_intensity_ONDJF_170E-240E_50N-75N_1981_2024.txt"
)

SEASON_NAME = "ONDJF"

# 强度搜索区域，与识别区一致
LAT_MIN = 50
LAT_MAX = 75
LON_MIN = 170
LON_MAX = 240

# 10°×10° box
BOX_LAT_DEG = 10
BOX_LON_DEG = 10

# 1°网格下，50–60N 包含 11 个格点，跨度正好 10°
# 之前复现也是这个逻辑
BOX_LAT_POINTS = BOX_LAT_DEG + 1
BOX_LON_POINTS = BOX_LON_DEG + 1


# ============================================================
# 2. helper functions
# ============================================================
def calc_winter_year(dt):
    """
    10/11/12月归当年，1/2月归上一年。
    """
    dt = pd.to_datetime(dt)
    if dt.month in [10, 11, 12]:
        return dt.year
    return dt.year - 1


def get_box_mean_max(field_2d, lats, lons):
    """
    对单日 anomaly field 搜索所有 10°×10° box，
    返回最大 box mean 及其位置。

    输入:
      field_2d: shape = (lat, lon)
      lats/lons: 1D coordinate arrays

    输出:
      best_mean
      best_lat_min, best_lat_max, best_lon_min, best_lon_max
    """
    nlat, nlon = field_2d.shape

    best_mean = np.nan
    best_lat_min = np.nan
    best_lat_max = np.nan
    best_lon_min = np.nan
    best_lon_max = np.nan

    for i in range(0, nlat - BOX_LAT_POINTS + 1):
        for j in range(0, nlon - BOX_LON_POINTS + 1):
            box = field_2d[i:i + BOX_LAT_POINTS, j:j + BOX_LON_POINTS]

            if np.all(np.isnan(box)):
                continue

            m = float(np.nanmean(box))

            if np.isnan(best_mean) or m > best_mean:
                best_mean = m
                best_lat_min = float(lats[i])
                best_lat_max = float(lats[i + BOX_LAT_POINTS - 1])
                best_lon_min = float(lons[j])
                best_lon_max = float(lons[j + BOX_LON_POINTS - 1])

    return best_mean, best_lat_min, best_lat_max, best_lon_min, best_lon_max


def check_duration_from_dates(start_date, end_date):
    return (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1


# ============================================================
# 3. read inputs
# ============================================================
print("=" * 100)
print("READ INPUTS")
print("=" * 100)

events = pd.read_csv(EVENT_FILE)

events["start_date"] = pd.to_datetime(events["start_date"])
events["end_date"] = pd.to_datetime(events["end_date"])

print("Events file:", EVENT_FILE)
print("Number of events:", len(events))
print(events.head())

ds = xr.open_dataset(ANOM_FILE)

if "z" not in ds.variables:
    raise ValueError(f"Variable z not found in anomaly file. Variables: {list(ds.variables)}")

z = ds["z"]

print("\nAnomaly file:", ANOM_FILE)
print(ds)
print("Anomaly shape:", z.shape)
print("Anomaly time:", str(ds.time.values[0]), "to", str(ds.time.values[-1]))
print("Anomaly lat :", float(ds.lat.min()), "to", float(ds.lat.max()), "n =", ds.sizes["lat"])
print("Anomaly lon :", float(ds.lon.min()), "to", float(ds.lon.max()), "n =", ds.sizes["lon"])
print("Anomaly NaN:", int(z.isnull().sum().values))
print("Anomaly mean:", float(z.mean(skipna=True).values))

# 取强度搜索区
z_box_domain = z.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)

lats = z_box_domain["lat"].values
lons = z_box_domain["lon"].values

print("\nIntensity domain:")
print("lat:", float(lats.min()), "to", float(lats.max()), "n =", len(lats))
print("lon:", float(lons.min()), "to", float(lons.max()), "n =", len(lons))
print("box points:", BOX_LAT_POINTS, "x", BOX_LON_POINTS)

if len(lats) != 26:
    print("[WARNING] lat points should be 26 for 50–75N on 1° grid.")

if len(lons) != 71:
    print("[WARNING] lon points should be 71 for 170–240E on 1° grid.")

if len(lats) < BOX_LAT_POINTS or len(lons) < BOX_LON_POINTS:
    raise ValueError("Intensity domain is smaller than the 10°x10° box.")


# ============================================================
# 4. compute daily and event intensity
# ============================================================
print("\n" + "=" * 100)
print("COMPUTE INTENSITY")
print("=" * 100)

daily_rows = []
event_rows = []

# 为了加快日期索引
all_times = pd.to_datetime(ds["time"].values)
time_to_index = {t: i for i, t in enumerate(all_times)}

for _, ev in tqdm(events.iterrows(), total=len(events), desc="Events"):
    event_id = int(ev["event_id"])
    start_date = pd.to_datetime(ev["start_date"])
    end_date = pd.to_datetime(ev["end_date"])
    duration_days = int(ev["duration_days"])

    expected_duration = check_duration_from_dates(start_date, end_date)

    if expected_duration != duration_days:
        print(
            f"[WARNING] event {event_id}: duration_days={duration_days}, "
            f"date-derived duration={expected_duration}"
        )

    event_dates = pd.date_range(start_date, end_date, freq="D")

    daily_intensities = []

    for dt in event_dates:
        if dt not in time_to_index:
            raise ValueError(f"Date {dt} of event {event_id} not found in anomaly time axis.")

        # 取当天强度搜索区 anomaly
        field = z_box_domain.sel(time=dt).values.astype(np.float64)

        best_mean, best_lat_min, best_lat_max, best_lon_min, best_lon_max = get_box_mean_max(
            field,
            lats,
            lons
        )

        daily_intensities.append(best_mean)

        daily_rows.append({
            "event_id": event_id,
            "date": dt,
            "daily_intensity_gpm": best_mean,
            "best_box_lat_min": best_lat_min,
            "best_box_lat_max": best_lat_max,
            "best_box_lon_min": best_lon_min,
            "best_box_lon_max": best_lon_max,
        })

    daily_intensities = np.array(daily_intensities, dtype=np.float64)

    max_intensity = float(np.nanmax(daily_intensities))
    cumulative_intensity = float(np.nansum(daily_intensities))
    mean_daily_intensity = float(np.nanmean(daily_intensities))
    min_daily_intensity = float(np.nanmin(daily_intensities))

    # 峰值日期与 box
    imax = int(np.nanargmax(daily_intensities))
    peak_date = event_dates[imax]

    peak_row = daily_rows[-len(event_dates) + imax]

    event_rows.append({
        "event_id": event_id,
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": duration_days,
        "winter_year": int(ev["winter_year"]) if "winter_year" in ev.index else calc_winter_year(start_date),
        "start_year": int(ev["start_year"]) if "start_year" in ev.index else start_date.year,
        "start_month": int(ev["start_month"]) if "start_month" in ev.index else start_date.month,

        "max_intensity_gpm": max_intensity,
        "cumulative_intensity_gpm": cumulative_intensity,
        "mean_daily_intensity_gpm": mean_daily_intensity,
        "min_daily_intensity_gpm": min_daily_intensity,

        "peak_date": peak_date,
        "peak_box_lat_min": peak_row["best_box_lat_min"],
        "peak_box_lat_max": peak_row["best_box_lat_max"],
        "peak_box_lon_min": peak_row["best_box_lon_min"],
        "peak_box_lon_max": peak_row["best_box_lon_max"],
    })

daily_df = pd.DataFrame(daily_rows)
event_intensity_df = pd.DataFrame(event_rows)


# ============================================================
# 5. consistency checks
# ============================================================
print("\n" + "=" * 100)
print("CONSISTENCY CHECKS")
print("=" * 100)

print("Daily rows:", len(daily_df))
print("Event rows:", len(event_intensity_df))
print("Events:", len(events))

expected_daily_rows = int(events["duration_days"].sum())
print("Expected daily rows from event duration sum:", expected_daily_rows)

if len(daily_df) != expected_daily_rows:
    print("[WARNING] daily rows != sum(duration_days)")
else:
    print("[OK] daily rows match sum(duration_days)")

if len(event_intensity_df) != len(events):
    raise ValueError("Event intensity rows do not match events.")

nan_daily = int(daily_df["daily_intensity_gpm"].isna().sum())
nan_max = int(event_intensity_df["max_intensity_gpm"].isna().sum())
nan_cum = int(event_intensity_df["cumulative_intensity_gpm"].isna().sum())

print("NaN daily intensity:", nan_daily)
print("NaN max intensity:", nan_max)
print("NaN cumulative intensity:", nan_cum)

if nan_daily != 0 or nan_max != 0 or nan_cum != 0:
    raise ValueError("NaN exists in intensity results.")

# duration consistency from daily_df
daily_count_by_event = daily_df.groupby("event_id").size()
merged_check = event_intensity_df.set_index("event_id").join(
    daily_count_by_event.rename("daily_count")
)

merged_check["duration_diff"] = merged_check["daily_count"] - merged_check["duration_days"]

print("\nDuration difference counts:")
print(merged_check["duration_diff"].value_counts().sort_index())

if not (merged_check["duration_diff"] == 0).all():
    raise ValueError("Some events have daily intensity count != duration_days.")

print("\nMax intensity stats:")
print(event_intensity_df["max_intensity_gpm"].describe())

print("\nCumulative intensity stats:")
print(event_intensity_df["cumulative_intensity_gpm"].describe())

print("\nTop 10 max intensity events:")
print(
    event_intensity_df.sort_values("max_intensity_gpm", ascending=False)
    .head(10)
    [
        [
            "event_id",
            "start_date",
            "end_date",
            "duration_days",
            "winter_year",
            "max_intensity_gpm",
            "cumulative_intensity_gpm",
            "peak_date",
            "peak_box_lat_min",
            "peak_box_lat_max",
            "peak_box_lon_min",
            "peak_box_lon_max",
        ]
    ]
)

print("\nTop 10 cumulative intensity events:")
print(
    event_intensity_df.sort_values("cumulative_intensity_gpm", ascending=False)
    .head(10)
    [
        [
            "event_id",
            "start_date",
            "end_date",
            "duration_days",
            "winter_year",
            "max_intensity_gpm",
            "cumulative_intensity_gpm",
        ]
    ]
)


# ============================================================
# 6. save outputs
# ============================================================
print("\n" + "=" * 100)
print("SAVE OUTPUTS")
print("=" * 100)

event_intensity_df.to_csv(OUT_EVENT_INTENSITY, index=False, encoding="utf-8-sig")
daily_df.to_csv(OUT_DAILY_INTENSITY, index=False, encoding="utf-8-sig")

print("Saved event intensity:")
print(OUT_EVENT_INTENSITY)

print("Saved daily intensity:")
print(OUT_DAILY_INTENSITY)


# ============================================================
# 7. write check txt
# ============================================================
lines = []

lines.append("ABH INTENSITY CHECK")
lines.append("=" * 80)
lines.append(f"Event file: {EVENT_FILE}")
lines.append(f"Anomaly file: {ANOM_FILE}")
lines.append(f"Output event intensity: {OUT_EVENT_INTENSITY}")
lines.append(f"Output daily intensity: {OUT_DAILY_INTENSITY}")
lines.append("")
lines.append("Settings:")
lines.append("Period: 1981-2024")
lines.append(f"Event season: {SEASON_NAME}")
lines.append("Intensity domain: 170E-240E, 50N-75N")
lines.append("Box size: 10deg x 10deg")
lines.append("Grid: 1 degree")
lines.append(f"Box points: {BOX_LAT_POINTS} x {BOX_LON_POINTS}")
lines.append("")
lines.append("Counts:")
lines.append(f"Events: {len(events)}")
lines.append(f"Daily rows: {len(daily_df)}")
lines.append(f"Expected daily rows from duration sum: {expected_daily_rows}")
lines.append(f"NaN daily intensity: {nan_daily}")
lines.append(f"NaN max intensity: {nan_max}")
lines.append(f"NaN cumulative intensity: {nan_cum}")
lines.append("")
lines.append("Max intensity stats:")
lines.append(str(event_intensity_df["max_intensity_gpm"].describe()))
lines.append("")
lines.append("Cumulative intensity stats:")
lines.append(str(event_intensity_df["cumulative_intensity_gpm"].describe()))
lines.append("")
lines.append("Duration difference counts:")
lines.append(str(merged_check["duration_diff"].value_counts().sort_index()))
lines.append("")
lines.append("Top 10 max intensity events:")
lines.append(str(
    event_intensity_df.sort_values("max_intensity_gpm", ascending=False)
    .head(10)
    [
        [
            "event_id",
            "start_date",
            "end_date",
            "duration_days",
            "winter_year",
            "max_intensity_gpm",
            "cumulative_intensity_gpm",
            "peak_date",
            "peak_box_lat_min",
            "peak_box_lat_max",
            "peak_box_lon_min",
            "peak_box_lon_max",
        ]
    ]
))
lines.append("")
lines.append("Top 10 cumulative intensity events:")
lines.append(str(
    event_intensity_df.sort_values("cumulative_intensity_gpm", ascending=False)
    .head(10)
    [
        [
            "event_id",
            "start_date",
            "end_date",
            "duration_days",
            "winter_year",
            "max_intensity_gpm",
            "cumulative_intensity_gpm",
        ]
    ]
))

with open(OUT_CHECK, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Saved check:")
print(OUT_CHECK)

ds.close()
gc.collect()

print("\nDone.")
