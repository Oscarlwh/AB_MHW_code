import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. settings
# ============================================================
START_YEAR = 1981
END_YEAR = 2024

base_dir = "/path/to/data/ABH_170E240E_1981_2024"
yearly_dir = os.path.join(base_dir, "yearly_daily_gpm_no_leap")

out_file = os.path.join(
    base_dir,
    "ERA5_Z500_daily_170E240E_35N90N_1981_2024_merged_gpm_no_leap.nc"
)

log_file = os.path.join(
    base_dir,
    "merge_check_1981_2024_170E240E_35N90N.csv"
)

target_lats = np.arange(35, 91, 1.0)
target_lons = np.arange(170, 241, 1.0)

expected_ntime = (END_YEAR - START_YEAR + 1) * 365


# ============================================================
# 2. preprocess for xarray merge_z500_daily
# ============================================================
def preprocess_for_merge(ds):
    """
    合并前清理每个逐年文件。

    目标：
    1. 只保留 z 变量；
    2. 删除 lev/level/pressure_level/number/expver 等多余变量或坐标；
    3. 如果 z 仍然带单层 pressure dimension，则 squeeze 掉；
    4. 统一维度顺序为 time, lat, lon；
    5. 统一 lat/lon/time 排序。
    """

    # ------------------------------------------------------------
    # 只保留 z
    # ------------------------------------------------------------
    if "z" not in ds.variables:
        raise ValueError(f"Variable 'z' not found. Variables: {list(ds.variables)}")

    ds = ds[["z"]]

    # ------------------------------------------------------------
    # 如果 z 还有多余单层维度，先 squeeze
    # ------------------------------------------------------------
    possible_level_dims = [
        "lev",
        "level",
        "pressure_level",
        "isobaricInhPa",
        "number",
        "expver",
    ]

    for dim in possible_level_dims:
        if dim in ds["z"].dims:
            if ds.sizes[dim] == 1:
                ds["z"] = ds["z"].squeeze(dim, drop=True)
            else:
                raise ValueError(f"Unexpected non-singleton dimension {dim}: size={ds.sizes[dim]}")

    # ------------------------------------------------------------
    # 删除多余坐标或变量
    # ------------------------------------------------------------
    drop_names = [
        "lev",
        "level",
        "pressure_level",
        "isobaricInhPa",
        "number",
        "expver",
        "step",
        "surface",
        "valid_time",
    ]

    existing_drop = [
        name for name in drop_names
        if name in ds.coords or name in ds.variables
    ]

    if existing_drop:
        ds = ds.drop_vars(existing_drop, errors="ignore")

    # ------------------------------------------------------------
    # 检查必要维度
    # ------------------------------------------------------------
    needed_dims = {"time", "lat", "lon"}
    actual_dims = set(ds["z"].dims)

    if not needed_dims.issubset(actual_dims):
        raise ValueError(f"z dims are not expected. z.dims={ds['z'].dims}")

    # ------------------------------------------------------------
    # 只保留 time, lat, lon 三个维度
    # ------------------------------------------------------------
    ds["z"] = ds["z"].transpose("time", "lat", "lon")

    # ------------------------------------------------------------
    # 排序
    # ------------------------------------------------------------
    ds = ds.sortby("time").sortby("lat").sortby("lon")

    return ds


# ============================================================
# 3. check yearly files before merge_z500_daily
# ============================================================
yearly_files = []
check_rows = []

print("=" * 100)
print("CHECK YEARLY FILES")
print("=" * 100)

for year in range(START_YEAR, END_YEAR + 1):
    f = os.path.join(
        yearly_dir,
        f"ERA5_Z500_daily_170E240E_35N90N_{year}_gpm_no_leap.nc"
    )

    row = {
        "year": year,
        "file": f,
        "exists": os.path.exists(f),
        "ntime": np.nan,
        "nlat": np.nan,
        "nlon": np.nan,
        "feb29_count": np.nan,
        "duplicate_time_count": np.nan,
        "nan_count": np.nan,
        "mean": np.nan,
        "min": np.nan,
        "max": np.nan,
        "lat_match": False,
        "lon_match": False,
        "extra_coords_or_vars": "",
        "status": "UNKNOWN",
        "message": "",
    }

    if not os.path.exists(f):
        row["status"] = "MISSING"
        row["message"] = "file not found"
        check_rows.append(row)
        print(f"[MISSING] {year}: {f}")
        continue

    try:
        ds0 = xr.open_dataset(f)

        # 记录多余变量/坐标，方便追踪
        extras = []
        for name in [
            "lev",
            "level",
            "pressure_level",
            "isobaricInhPa",
            "number",
            "expver",
            "step",
            "surface",
            "valid_time",
        ]:
            if name in ds0.coords or name in ds0.variables or name in ds0.dims:
                extras.append(name)

        row["extra_coords_or_vars"] = ",".join(extras)

        # 用同一个 preprocess 检查
        ds = preprocess_for_merge(ds0)

        z = ds["z"]
        time = pd.to_datetime(ds["time"].values)

        feb29_count = int(((time.month == 2) & (time.day == 29)).sum())
        duplicate_count = int(pd.Index(time).duplicated().sum())

        lat = ds["lat"].values
        lon = ds["lon"].values

        lat_match = len(lat) == len(target_lats) and np.allclose(lat, target_lats, atol=1e-6)
        lon_match = len(lon) == len(target_lons) and np.allclose(lon, target_lons, atol=1e-6)

        nan_count = int(z.isnull().sum().values)

        row.update({
            "ntime": ds.sizes["time"],
            "nlat": ds.sizes["lat"],
            "nlon": ds.sizes["lon"],
            "feb29_count": feb29_count,
            "duplicate_time_count": duplicate_count,
            "nan_count": nan_count,
            "mean": float(z.mean(skipna=True).values),
            "min": float(z.min(skipna=True).values),
            "max": float(z.max(skipna=True).values),
            "lat_match": lat_match,
            "lon_match": lon_match,
        })

        if (
            ds.sizes["time"] == 365
            and ds.sizes["lat"] == 56
            and ds.sizes["lon"] == 71
            and feb29_count == 0
            and duplicate_count == 0
            and nan_count == 0
            and lat_match
            and lon_match
        ):
            row["status"] = "OK"
            yearly_files.append(f)

            extra_msg = f", extras={extras}" if len(extras) > 0 else ""
            print(f"[OK] {year}: shape={z.shape}, mean={row['mean']:.2f}{extra_msg}")

        else:
            row["status"] = "CHECK"
            row["message"] = "one or more checks failed"
            print(
                f"[CHECK] {year}: "
                f"ntime={row['ntime']}, nlat={row['nlat']}, nlon={row['nlon']}, "
                f"feb29={feb29_count}, dup={duplicate_count}, nan={nan_count}, "
                f"lat_match={lat_match}, lon_match={lon_match}, extras={extras}"
            )

        ds0.close()

    except Exception as e:
        row["status"] = "ERROR"
        row["message"] = repr(e)
        print(f"[ERROR] {year}: {repr(e)}")

    check_rows.append(row)

check_df = pd.DataFrame(check_rows)
check_df.to_csv(log_file, index=False, encoding="utf-8-sig")

print("\nSaved yearly check log:")
print(log_file)

bad = check_df[check_df["status"] != "OK"]

if len(bad) > 0:
    print("\n" + "=" * 100)
    print("BAD YEARS FOUND. STOP BEFORE MERGE.")
    print("=" * 100)
    print(
        bad[
            [
                "year",
                "status",
                "ntime",
                "nlat",
                "nlon",
                "feb29_count",
                "duplicate_time_count",
                "nan_count",
                "lat_match",
                "lon_match",
                "extra_coords_or_vars",
                "message",
            ]
        ]
    )
    raise SystemExit("Please fix bad yearly files before merging.")

if len(yearly_files) != END_YEAR - START_YEAR + 1:
    raise SystemExit("Yearly file count is not 44. Stop.")

print("\nAll yearly files passed checks.")
print("Number of files:", len(yearly_files))


# ============================================================
# 4. merge_z500_daily yearly files
# ============================================================
print("\n" + "=" * 100)
print("MERGE YEARLY FILES")
print("=" * 100)

ds_all = xr.open_mfdataset(
    yearly_files,
    combine="by_coords",
    preprocess=preprocess_for_merge,
    parallel=False,
    chunks={"time": 365},
    coords="minimal",
    compat="override",
    join="exact",
)

# 统一排序
ds_all = ds_all.sortby("time").sortby("lat").sortby("lon")

print("\nMerged dataset overview:")
print(ds_all)


# ============================================================
# 5. merged checks
# ============================================================
print("\n" + "=" * 100)
print("MERGED CHECKS")
print("=" * 100)

time = pd.to_datetime(ds_all["time"].values)
lat = ds_all["lat"].values
lon = ds_all["lon"].values

feb29_count = int(((time.month == 2) & (time.day == 29)).sum())
duplicate_time_count = int(pd.Index(time).duplicated().sum())

lat_match = len(lat) == len(target_lats) and np.allclose(lat, target_lats, atol=1e-6)
lon_match = len(lon) == len(target_lons) and np.allclose(lon, target_lons, atol=1e-6)

print("Expected ntime:", expected_ntime)
print("Actual ntime  :", ds_all.sizes["time"])
print("lat n:", ds_all.sizes["lat"], "min:", float(lat.min()), "max:", float(lat.max()))
print("lon n:", ds_all.sizes["lon"], "min:", float(lon.min()), "max:", float(lon.max()))
print("time start:", time[0])
print("time end  :", time[-1])
print("Feb29 count:", feb29_count)
print("Duplicate time count:", duplicate_time_count)
print("lat match target:", lat_match)
print("lon match target:", lon_match)

nan_count = int(ds_all["z"].isnull().sum().compute().values)
mean_val = float(ds_all["z"].mean(skipna=True).compute().values)
min_val = float(ds_all["z"].min(skipna=True).compute().values)
max_val = float(ds_all["z"].max(skipna=True).compute().values)
std_val = float(ds_all["z"].std(skipna=True).compute().values)

print("NaN count:", nan_count)
print("Mean:", mean_val)
print("Std :", std_val)
print("Min :", min_val)
print("Max :", max_val)

if ds_all.sizes["time"] != expected_ntime:
    raise ValueError("Merged time length is not expected.")

if ds_all.sizes["lat"] != 56 or ds_all.sizes["lon"] != 71:
    raise ValueError("Merged spatial shape is not 56 x 71.")

if feb29_count != 0:
    raise ValueError("Feb 29 exists in merged file.")

if duplicate_time_count != 0:
    raise ValueError("Duplicate time values exist in merged file.")

if nan_count != 0:
    raise ValueError("NaN exists in merged z.")

if not lat_match or not lon_match:
    raise ValueError("Merged lat/lon does not match target grid.")

if not (4500 < mean_val < 6000):
    raise ValueError("Merged z mean does not look like gpm.")

print("\n[OK] merged dataset passed all checks.")


# ============================================================
# 6. save merged file
# ============================================================
print("\n" + "=" * 100)
print("SAVE MERGED FILE")
print("=" * 100)

ds_all["z"].attrs["long_name"] = "500 hPa geopotential height"
ds_all["z"].attrs["units"] = "gpm"
ds_all["z"].attrs["note"] = (
    "Daily mean ERA5 500-hPa geopotential height, converted from geopotential "
    "by dividing by 9.80665, no-leap calendar, standardized to 1-degree grid."
)

ds_all.attrs["period"] = "1981-2024"
ds_all.attrs["domain"] = "170E-240E, 35N-90N"
ds_all.attrs["grid"] = "1 degree"
ds_all.attrs["processing"] = (
    "Merged yearly daily gpm no-leap files. "
    "Special years 1986, 2023, and 2024 were standardized to 1-degree grid. "
    "Extra coordinates such as lev/number were dropped during merge_z500_daily."
)

encoding = {
    "z": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(np.nan),
    }
}

if os.path.exists(out_file):
    print("[WARNING] output file already exists and will be overwritten:")
    print(out_file)
    os.remove(out_file)

ds_all.to_netcdf(out_file, encoding=encoding)

ds_all.close()

print("\n[MERGE DONE]")
print("Saved:")
print(out_file)
print("\nLog:")
print(log_file)