import os
import gc
import subprocess
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ============================================================
# 1. basic settings
# ============================================================
START_YEAR = 1981
END_YEAR = 2024

# Processing domain. The 5N lower bound preserves a 15-degree
# southern buffer for later calculations centered at 20N.
LON_MIN = 170
LON_MAX = 240
LAT_MIN = 5
LAT_MAX = 90

G = 9.80665

target_lats = np.arange(LAT_MIN, LAT_MAX + 1, 1.0)
target_lons = np.arange(LON_MIN, LON_MAX + 1, 1.0)

# 原始数据路径
NORMAL_DIR = "/path/to/data/ERA5"

SPECIAL_FILES = {
    1986: "/path/to/data/draft/500hPa/era5_1986_500hpageopotential.nc",
    2023: "/path/to/data/draft/500hPa/era5_2023_500hpageopotential.nc",
    2024: "/path/to/data/draft/500hPa/era5_2024_500hpageopotential.nc",
}

# 输出目录
OUT_ROOT = "/path/to/data/ABH_170E240E_1981_2024_5N90N"
OUT_YEARLY = os.path.join(OUT_ROOT, "yearly_daily_gpm_no_leap")
os.makedirs(OUT_YEARLY, exist_ok=True)

OUT_MERGED = os.path.join(
    OUT_ROOT,
    "ERA5_Z500_daily_170E240E_5N90N_1981_2024_merged_gpm_no_leap.nc"
)

OUT_LOG = os.path.join(
    OUT_ROOT,
    "process_log_1981_2024_170E240E_5N90N.csv"
)

CDO_BIN = "/opt/homebrew/bin/cdo"
CDO_GRID_FILE = "/private/tmp/cdo_grid_170E240E_5N90N_1deg.txt"

# ============================================================
# 2. helper functions
# ============================================================
def get_input_file(year):
    if year in SPECIAL_FILES:
        return SPECIAL_FILES[year]
    return os.path.join(NORMAL_DIR, f"{year}.nc")


def find_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    return None


def open_dataset_safely(path):
    """
    尝试用不同 engine 打开，增强兼容性
    """
    engines = [None, "netcdf4", "h5netcdf", "scipy"]

    last_err = None
    for eng in engines:
        try:
            if eng is None:
                return xr.open_dataset(path)
            else:
                return xr.open_dataset(path, engine=eng)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Cannot open {path}. Last error: {last_err}")


def standardize_lon(ds, lon_name):
    """
    统一经度到 0–360，并排序
    """
    lon = ds[lon_name]

    if float(lon.min()) < 0:
        ds = ds.assign_coords({lon_name: (lon % 360)})
        ds = ds.sortby(lon_name)

    return ds


def select_500hpa(z, ds):
    """
    兼容 pressure_level / level / lev 等不同名字
    """
    lev_name = find_name(ds, ["pressure_level", "level", "lev", "isobaricInhPa"])

    if lev_name is None:
        return z

    # 如果只有一个层，直接 squeeze；否则选 500
    if lev_name in z.dims:
        lev_values = ds[lev_name].values
        if len(np.atleast_1d(lev_values)) == 1:
            z = z.squeeze(lev_name, drop=True)
        else:
            z = z.sel({lev_name: 500}, method="nearest")

    return z


def subset_domain(z, lat_name, lon_name):
    """
    裁剪 170E–240E, 5N–90N
    兼容纬度升序/降序
    """
    # longitude
    z = z.sel({lon_name: slice(LON_MIN, LON_MAX)})

    # latitude
    lat = z[lat_name]
    if lat[0] > lat[-1]:
        z = z.sel({lat_name: slice(LAT_MAX, LAT_MIN)})
    else:
        z = z.sel({lat_name: slice(LAT_MIN, LAT_MAX)})

    return z


def select_target_grid(z, lat_name, lon_name):
    """
    将普通 1° 文件和 special 0.25° 文件统一到 1° 目标网格。

    special 年份的原始文件是全球 0.25° 小时数据；必须在 daily mean
    前先抽取到目标 1° 网格，否则会产生很大的逐年文件并导致后续无法合并。
    """
    z = z.sortby(lat_name).sortby(lon_name)
    z = z.sel(
        {
            lat_name: target_lats,
            lon_name: target_lons,
        },
        method="nearest",
    )
    return z


def remove_feb29(da, time_name):
    time = pd.to_datetime(da[time_name].values)
    mask = ~((time.month == 2) & (time.day == 29))
    return da.sel({time_name: mask})


def rename_to_standard(da, time_name, lat_name, lon_name):
    rename_dict = {}
    if time_name != "time":
        rename_dict[time_name] = "time"
    if lat_name != "lat":
        rename_dict[lat_name] = "lat"
    if lon_name != "lon":
        rename_dict[lon_name] = "lon"

    da = da.rename(rename_dict)
    return da


def ensure_cdo_grid_file():
    grid_text = "\n".join([
        "gridtype = lonlat",
        f"xsize = {len(target_lons)}",
        f"ysize = {len(target_lats)}",
        f"xfirst = {LON_MIN}",
        "xinc = 1",
        f"yfirst = {LAT_MIN}",
        "yinc = 1",
        "",
    ])

    with open(CDO_GRID_FILE, "w", encoding="utf-8") as f:
        f.write(grid_text)

    return CDO_GRID_FILE


def process_special_year_with_cdo(year, in_file, out_file):
    """
    Special years are 0.25° hourly files. Use CDO to reduce them to the
    same daily 1° target grid before final xarray cleanup.
    """
    if not os.path.exists(CDO_BIN):
        raise FileNotFoundError(f"CDO not found: {CDO_BIN}")

    grid_file = ensure_cdo_grid_file()
    tmp_file = os.path.join(OUT_YEARLY, f".tmp_cdo_5N90N_{year}.nc")

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    cmd = [
        CDO_BIN,
        "-O",
        "-f", "nc4",
        "-z", "zip_4",
        "-daymean",
        "-del29feb",
        f"-remapnn,{grid_file}",
        "-sellevel,500",
        f"-sellonlatbox,{LON_MIN},{LON_MAX},{LAT_MIN},{LAT_MAX}",
        in_file,
        tmp_file,
    ]

    print("[CDO]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    ds_tmp = xr.open_dataset(tmp_file)

    if "z" not in ds_tmp.variables:
        raise ValueError(f"Variable z not found in CDO output. Variables: {list(ds_tmp.variables)}")

    z_daily = ds_tmp["z"]

    for dim in ["lev", "level", "pressure_level", "isobaricInhPa"]:
        if dim in z_daily.dims:
            if z_daily.sizes[dim] == 1:
                z_daily = z_daily.squeeze(dim, drop=True)
            else:
                z_daily = z_daily.sel({dim: 500}, method="nearest")

    time_name = find_name(ds_tmp, ["time", "valid_time"])
    lat_name = find_name(ds_tmp, ["lat", "latitude"])
    lon_name = find_name(ds_tmp, ["lon", "longitude"])

    if time_name is None or lat_name is None or lon_name is None:
        raise ValueError(f"Cannot identify CDO output coordinates. Dims: {z_daily.dims}")

    z_daily = rename_to_standard(z_daily, time_name, lat_name, lon_name)

    # CDO may write no-leap/cftime timestamps for special years. Convert to
    # pandas datetime64 so all yearly files combine cleanly with xarray.
    normalized_time = []
    for t in z_daily["time"].values:
        if hasattr(t, "year") and hasattr(t, "month") and hasattr(t, "day"):
            normalized_time.append(pd.Timestamp(year=t.year, month=t.month, day=t.day))
        else:
            normalized_time.append(pd.Timestamp(t))
    z_daily = z_daily.assign_coords(time=pd.DatetimeIndex(normalized_time))

    z_daily = z_daily.transpose("time", "lat", "lon")
    z_daily = z_daily.sortby("lat").sortby("lon")

    extra_coords = [name for name in z_daily.coords if name not in ["time", "lat", "lon"]]
    if extra_coords:
        z_daily = z_daily.drop_vars(extra_coords, errors="ignore")

    z_daily = (z_daily / G).astype("float32").load()
    z_daily.name = "z"
    z_daily.attrs["long_name"] = "500 hPa geopotential height"
    z_daily.attrs["units"] = "gpm"
    z_daily.attrs["note"] = "Converted from ERA5 geopotential by dividing by 9.80665."

    ds_out = z_daily.to_dataset(name="z")
    ds_out.attrs["source"] = in_file
    ds_out.attrs["processing"] = (
        "CDO special-year path: select 500 hPa, subset 170E-240E and 5N-90N, "
        "nearest-neighbor remap to 1 degree, daily mean, remove Feb 29, "
        "convert geopotential to gpm."
    )
    ds_out.attrs["domain"] = "170E-240E, 5N-90N"
    ds_out.attrs["year"] = str(year)

    encoding = {
        "z": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }

    if os.path.exists(out_file):
        os.remove(out_file)

    ds_out.to_netcdf(out_file, encoding=encoding)

    info_update = {
        "status": "OK",
        "ntime": ds_out.sizes["time"],
        "nlat": ds_out.sizes["lat"],
        "nlon": ds_out.sizes["lon"],
        "nan_count": int(ds_out["z"].isnull().sum().values),
        "mean_gpm": float(ds_out["z"].mean(skipna=True).values),
        "min_gpm": float(ds_out["z"].min(skipna=True).values),
        "max_gpm": float(ds_out["z"].max(skipna=True).values),
    }

    ds_tmp.close()
    ds_out.close()

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    return info_update


# ============================================================
# 3. process one year
# ============================================================
def process_one_year(year):
    in_file = get_input_file(year)

    out_file = os.path.join(
        OUT_YEARLY,
        f"ERA5_Z500_daily_170E240E_5N90N_{year}_gpm_no_leap.nc"
    )

    info = {
        "year": year,
        "input_file": in_file,
        "output_file": out_file,
        "status": "UNKNOWN",
        "ntime": np.nan,
        "nlat": np.nan,
        "nlon": np.nan,
        "nan_count": np.nan,
        "mean_gpm": np.nan,
        "min_gpm": np.nan,
        "max_gpm": np.nan,
        "message": "",
    }

    if not os.path.exists(in_file):
        info["status"] = "MISSING"
        info["message"] = "input file not found"
        print(f"[MISSING] {year}: {in_file}")
        return info

    if os.path.exists(out_file):
        try:
            ds_check = xr.open_dataset(out_file)
            z_check = ds_check["z"]

            extra_existing_coords = [
                name for name in ds_check.coords
                if name not in ["time", "lat", "lon"]
            ]
            if extra_existing_coords:
                raise ValueError(f"extra coords found: {extra_existing_coords}")

            if not np.issubdtype(ds_check["time"].dtype, np.datetime64):
                raise ValueError(f"time dtype is not datetime64: {ds_check['time'].dtype}")

            info["status"] = "EXISTS"
            info["ntime"] = z_check.sizes.get("time", np.nan)
            info["nlat"] = z_check.sizes.get("lat", np.nan)
            info["nlon"] = z_check.sizes.get("lon", np.nan)
            info["nan_count"] = int(z_check.isnull().sum().values)
            info["mean_gpm"] = float(z_check.mean(skipna=True).values)
            info["min_gpm"] = float(z_check.min(skipna=True).values)
            info["max_gpm"] = float(z_check.max(skipna=True).values)
            ds_check.close()
            print(f"[EXISTS] {year}: {out_file}")
            return info
        except Exception:
            print(f"[REPROCESS] existing output broken, reprocessing {year}")

    print("\n" + "=" * 80)
    print(f"[START] {year}")
    print("Input:", in_file)

    if year in SPECIAL_FILES:
        try:
            info.update(process_special_year_with_cdo(year, in_file, out_file))

            print(f"[OK] {year}")
            print("Output:", out_file)
            print("Shape:", (info["ntime"], info["nlat"], info["nlon"]))
            print("NaN count:", info["nan_count"])
            print("Mean gpm:", info["mean_gpm"])
            print("Min gpm:", info["min_gpm"])
            print("Max gpm:", info["max_gpm"])

        except Exception as e:
            info["status"] = "ERROR"
            info["message"] = repr(e)
            print(f"[ERROR] {year}")
            print(repr(e))

        gc.collect()
        return info

    try:
        ds = open_dataset_safely(in_file)

        # variable
        if "z" in ds.variables:
            z = ds["z"]
        elif "geopotential" in ds.variables:
            z = ds["geopotential"]
        else:
            raise ValueError(f"No z/geopotential variable found. Variables: {list(ds.variables)}")

        time_name = find_name(ds, ["time", "valid_time"])
        lat_name = find_name(ds, ["latitude", "lat"])
        lon_name = find_name(ds, ["longitude", "lon"])

        if time_name is None:
            raise ValueError("No time/valid_time coordinate found")
        if lat_name is None:
            raise ValueError("No latitude/lat coordinate found")
        if lon_name is None:
            raise ValueError("No longitude/lon coordinate found")

        ds = standardize_lon(ds, lon_name)
        z = ds["z"] if "z" in ds.variables else ds["geopotential"]

        # 取 500 hPa
        z = select_500hpa(z, ds)

        # 裁剪区域
        z = subset_domain(z, lat_name, lon_name)

        # 统一到 1° 目标网格，兼容 0.25° special years
        z = select_target_grid(z, lat_name, lon_name)

        # hourly → daily mean
        z_daily = z.resample({time_name: "1D"}).mean()

        # m2/s2 → gpm
        z_daily = z_daily / G
        z_daily.name = "z"
        z_daily.attrs["long_name"] = "500 hPa geopotential height"
        z_daily.attrs["units"] = "gpm"
        z_daily.attrs["note"] = "Converted from ERA5 geopotential by dividing by 9.80665."

        # 删除 Feb 29
        z_daily = remove_feb29(z_daily, time_name)

        # 标准化维度名
        z_daily = rename_to_standard(z_daily, time_name, lat_name, lon_name)

        # 按 lat 升序、lon 升序保存
        z_daily = z_daily.sortby("lat").sortby("lon")

        extra_coords = [name for name in z_daily.coords if name not in ["time", "lat", "lon"]]
        if extra_coords:
            z_daily = z_daily.drop_vars(extra_coords, errors="ignore")

        # 加载到内存，避免写文件时原始文件句柄问题
        z_daily = z_daily.astype("float32").load()

        ds_out = z_daily.to_dataset(name="z")

        ds_out.attrs["source"] = in_file
        ds_out.attrs["processing"] = (
            "Select 500 hPa, subset 170E-240E and 5N-90N, "
            "convert geopotential to gpm, daily mean, remove Feb 29."
        )
        ds_out.attrs["domain"] = "170E-240E, 5N-90N"
        ds_out.attrs["year"] = str(year)

        encoding = {
            "z": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            }
        }

        if os.path.exists(out_file):
            os.remove(out_file)

        ds_out.to_netcdf(out_file, encoding=encoding)

        # check
        info["status"] = "OK"
        info["ntime"] = ds_out.sizes["time"]
        info["nlat"] = ds_out.sizes["lat"]
        info["nlon"] = ds_out.sizes["lon"]
        info["nan_count"] = int(ds_out["z"].isnull().sum().values)
        info["mean_gpm"] = float(ds_out["z"].mean(skipna=True).values)
        info["min_gpm"] = float(ds_out["z"].min(skipna=True).values)
        info["max_gpm"] = float(ds_out["z"].max(skipna=True).values)

        print(f"[OK] {year}")
        print("Output:", out_file)
        print("Shape:", ds_out["z"].shape)
        print("NaN count:", info["nan_count"])
        print("Mean gpm:", info["mean_gpm"])
        print("Min gpm:", info["min_gpm"])
        print("Max gpm:", info["max_gpm"])

        ds.close()
        ds_out.close()

    except Exception as e:
        info["status"] = "ERROR"
        info["message"] = repr(e)
        print(f"[ERROR] {year}")
        print(repr(e))

    gc.collect()
    return info


# ============================================================
# 4. run yearly processing
# ============================================================
logs = []

for year in range(START_YEAR, END_YEAR + 1):
    info = process_one_year(year)
    logs.append(info)

log_df = pd.DataFrame(logs)
log_df.to_csv(OUT_LOG, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("YEARLY PROCESSING SUMMARY")
print(log_df[["year", "status", "ntime", "nlat", "nlon", "nan_count", "mean_gpm", "min_gpm", "max_gpm", "message"]])
print("\nSaved log:", OUT_LOG)

# ============================================================
# 5. stop if any missing/error
# ============================================================
bad = log_df[~log_df["status"].isin(["OK", "EXISTS"])]

if len(bad) > 0:
    print("\nSome years failed or missing. Please fix them before merging:")
    print(bad[["year", "status", "input_file", "message"]])
    raise SystemExit("Stop before merge_z500_daily because some yearly files are not ready.")

# ============================================================
# 6. merge_z500_daily yearly files
# ============================================================
print("\n" + "=" * 80)
print("[MERGE] yearly files")

yearly_files = [
    os.path.join(
        OUT_YEARLY,
        f"ERA5_Z500_daily_170E240E_5N90N_{year}_gpm_no_leap.nc"
    )
    for year in range(START_YEAR, END_YEAR + 1)
]

ds_all = xr.open_mfdataset(
    yearly_files,
    combine="by_coords",
    parallel=False,
    chunks={"time": 365}
)

# 排序并检查重复时间
ds_all = ds_all.sortby("time")

time_index = pd.to_datetime(ds_all["time"].values)
duplicated = pd.Index(time_index).duplicated().sum()
if duplicated > 0:
    raise ValueError(f"Duplicated time values found: {duplicated}")

# 检查 Feb 29
feb29_count = int(((time_index.month == 2) & (time_index.day == 29)).sum())
print("Feb 29 count:", feb29_count)

# 1981–2024 共 44 年，每年 no-leap 365 天
expected_ntime = (END_YEAR - START_YEAR + 1) * 365
actual_ntime = ds_all.sizes["time"]

print("Expected ntime:", expected_ntime)
print("Actual ntime  :", actual_ntime)

if actual_ntime != expected_ntime:
    print("[WARNING] ntime is not expected. Please check missing years or dates.")

# 整体检查
print("\nMerged dataset:")
print(ds_all)

print("\nMerged stats:")
print("NaN count:", int(ds_all["z"].isnull().sum().compute().values))
print("Mean:", float(ds_all["z"].mean(skipna=True).compute().values))
print("Min :", float(ds_all["z"].min(skipna=True).compute().values))
print("Max :", float(ds_all["z"].max(skipna=True).compute().values))

encoding = {
    "z": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(np.nan),
    }
}

ds_all.to_netcdf(OUT_MERGED, encoding=encoding)

print("\n[MERGE DONE]")
print("Saved:", OUT_MERGED)
