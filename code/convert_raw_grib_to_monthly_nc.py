from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================
RAW_DIR = Path("/path/to/data/JRA55_dia_2023_request844617/raw")
OUT_ROOT = Path("/path/to/data/dia")

# 只处理 2023 年。raw 里有一个 2022-12-31 18:00 的前置时次，先跳过。
YEAR = 2023

# ============================================================
# Variable settings
# ============================================================
VAR_INFO = {
    "cnvhr": {
        "param": 242,
        "old_name": "CNVHR_GDS0_ISBL_ave6h",
        "long_name": "Convective heating rate",
    },
    "lrghr": {
        "param": 241,
        "old_name": "LRGHR_GDS0_ISBL_ave6h",
        "long_name": "Large scale condensation heating rate",
    },
    "lwhr": {
        "param": 251,
        "old_name": "LWHR_GDS0_ISBL_ave6h",
        "long_name": "Longwave radiative heating rate",
    },
    "swhr": {
        "param": 250,
        "old_name": "SWHR_GDS0_ISBL_ave6h",
        "long_name": "Solar radiative heating rate",
    },
    "vdfhr": {
        "param": 246,
        "old_name": "VDFHR_GDS0_ISBL_ave6h",
        "long_name": "Vertical diffusion heating rate",
    },
}

LEVELS = [250, 300, 400, 500, 700, 850, 925, 1000]

# ============================================================
# Helper functions
# ============================================================
def parse_time_from_name(path: Path) -> pd.Timestamp:
    """
    Example:
    fcst_phy3m125_cnvhr.2023010100.luo844617
    """
    m = re.search(r"\.(\d{10})\.luo844617$", path.name)
    if m is None:
        raise ValueError(f"Cannot parse time from filename: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d%H")


def find_main_var(ds: xr.Dataset) -> str:
    """
    Return the main GRIB data variable.
    """
    candidates = list(ds.data_vars)
    if len(candidates) == 0:
        raise ValueError("No data variables found.")
    # cfgrib usually returns only one data variable.
    return candidates[0]


def open_one_grib(path: Path) -> xr.Dataset:
    """
    Open one GRIB file with cfgrib.
    """
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",  # 不生成 .idx 文件，避免 raw 目录变乱
        },
    )
    return ds


def normalize_lat_lon_time(da: xr.DataArray, t: pd.Timestamp) -> xr.DataArray:
    """
    Rename cfgrib coordinates to match the old NCL-converted files.
    """
    rename_dict = {}

    if "latitude" in da.dims:
        rename_dict["latitude"] = "g0_lat_1"
    if "longitude" in da.dims:
        rename_dict["longitude"] = "g0_lon_2"

    da = da.rename(rename_dict)

    # 添加时间维
    da = da.expand_dims({"initial_time0_hours": [np.datetime64(t)]})

    # 确保维度顺序
    da = da.transpose("initial_time0_hours", "g0_lat_1", "g0_lon_2")

    return da


def make_initial_time_strings(times: np.ndarray) -> xr.DataArray:
    """
    Match old file style approximately:
    initial_time0: string time variable
    """
    strs = []
    for t in pd.to_datetime(times):
        strs.append(t.strftime("%m/%d/%Y (%H:%M)"))

    return xr.DataArray(
        np.array(strs, dtype="S18"),
        dims=("initial_time0_hours",),
        coords={"initial_time0_hours": times},
        attrs={
            "NCL_converted_from_type": "string",
            "units": "mm/dd/yyyy (hh:mm)",
            "long_name": "Initial time of first record",
        },
    )


def make_initial_time_encoded(times: np.ndarray) -> xr.DataArray:
    """
    Make yyyymmddhh.hh_frac encoded time similar to old files.
    """
    vals = []
    for t in pd.to_datetime(times):
        vals.append(float(t.strftime("%Y%m%d%H")))

    return xr.DataArray(
        np.array(vals, dtype="float64"),
        dims=("initial_time0_hours",),
        coords={"initial_time0_hours": times},
        attrs={
            "units": "yyyymmddhh.hh_frac",
            "long_name": "initial time encoded as double",
        },
    )


# ============================================================
# Main conversion
# ============================================================
all_raw = sorted(RAW_DIR.glob("*.luo844617"))
print(f"Total raw files: {len(all_raw)}")

for var_key, info in VAR_INFO.items():
    var_files = sorted(RAW_DIR.glob(f"fcst_phy3m125_{var_key}.*.luo844617"))
    print(f"\n==== {var_key}: {len(var_files)} files ====")

    # 按月份分组，只处理 2023 年
    by_month = {}
    for f in var_files:
        t = parse_time_from_name(f)
        if t.year != YEAR:
            continue
        ym = t.strftime("%Y%m")
        by_month.setdefault(ym, []).append((t, f))

    for ym, tf_list in sorted(by_month.items()):
        tf_list = sorted(tf_list, key=lambda x: x[0])
        start_str = tf_list[0][0].strftime("%Y%m%d%H")
        end_str = tf_list[-1][0].strftime("%Y%m%d%H")

        print(f"\n[{var_key}] {ym}: {len(tf_list)} times, {start_str} to {end_str}")

        # 先读第一个文件，识别变量名和层次坐标
        test_ds = open_one_grib(tf_list[0][1])
        main_var = find_main_var(test_ds)

        # cfgrib 常见层坐标名：isobaricInhPa / isobaricInhPa0
        level_coord = None
        for cand in ["isobaricInhPa", "isobaricInhPa0", "level"]:
            if cand in test_ds.coords or cand in test_ds.dims:
                level_coord = cand
                break

        if level_coord is None:
            print(test_ds)
            raise ValueError(f"Cannot find pressure level coordinate in {tf_list[0][1]}")

        test_ds.close()

        # 对每个层次分别输出一个月文件
        for lev in LEVELS:
            da_list = []

            for t, f in tqdm(tf_list, desc=f"{var_key} {ym} {lev}hPa"):
                ds = open_one_grib(f)
                main_var = find_main_var(ds)

                da = ds[main_var]

                # 选气压层
                if level_coord in da.dims or level_coord in da.coords:
                    da = da.sel({level_coord: lev})
                else:
                    raise ValueError(f"Level coord {level_coord} not found in {f}")

                # 去掉无用标量坐标
                drop_names = []
                for cname in da.coords:
                    if cname not in ["latitude", "longitude", "g0_lat_1", "g0_lon_2"]:
                        drop_names.append(cname)
                if drop_names:
                    da = da.drop_vars(drop_names, errors="ignore")

                da = normalize_lat_lon_time(da, t)
                da_list.append(da)

                ds.close()

            da_month = xr.concat(da_list, dim="initial_time0_hours")

            # 改变量名和属性
            old_name = info["old_name"]
            da_month.name = old_name
            da_month.attrs.update({
                "forecast_time_units": "hours",
                "forecast_time": np.int32(6),
                "level": np.int32(lev),
                "parameter_number": np.int32(info["param"]),
                "parameter_table_version": np.int32(200),
                "gds_grid_type": np.int32(0),
                "level_indicator": np.int32(100),
                "units": "k/day",
                "long_name": info["long_name"],
                "center": "Japanese Meteorological Agency - Tokyo (RSMC)",
            })

            times = da_month["initial_time0_hours"].values

            out_ds = xr.Dataset(
                {
                    old_name: da_month,
                    "initial_time0_encoded": make_initial_time_encoded(times),
                    "initial_time0": make_initial_time_strings(times),
                }
            )

            out_ds.attrs.update({
                "title": "Converted from GDEX/JRA-55 GRIB using cfgrib",
                "source_request": "LUO844617",
                "dataset": "d628008 JRA-55 Near Real-Time Data",
                "converted_by": "convert_raw_grib_to_monthly_nc.py",
            })

            # 输出目录
            out_dir = OUT_ROOT / str(lev) / var_key
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = (
                f"fcst_phy3m125.{info['param']}_{var_key}."
                f"{start_str}_{end_str}.luo844617.nc"
            )
            out_path = out_dir / out_name

            encoding = {
                old_name: {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                }
            }

            print(f"[WRITE] {out_path}")
            out_ds.to_netcdf(out_path, encoding=encoding)
            out_ds.close()

print("\nAll done.")