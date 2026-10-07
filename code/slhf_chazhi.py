import os
import numpy as np
import xarray as xr

# =========================================================
# 1. 输入输出路径
# =========================================================
infile = "/path/to/data/era5_lwh/era5_surface_latent_heat_flux_2024.nc"
outdir = os.path.dirname(infile)
outfile = os.path.join(outdir, "ERA5_mslhf_daily_1x1_200E-225E_40N-50N_2024.nc")

# =========================================================
# 2. 打开数据
# =========================================================
ds = xr.open_dataset(infile)

print("=" * 80)
print("Original dataset")
print("=" * 80)
print(ds)
print("\nVariables:", list(ds.data_vars))
print("Coordinates:", list(ds.coords))

# =========================================================
# 3. 自动识别坐标名
# =========================================================
lat_name = None
lon_name = None
time_name = None

for name in ds.coords:
    low = name.lower()
    if low in ["latitude", "lat"]:
        lat_name = name
    elif low in ["longitude", "lon"]:
        lon_name = name
    elif low in ["valid_time", "time"]:
        time_name = name

if lat_name is None or lon_name is None or time_name is None:
    raise ValueError("没有识别到 time/lat/lon 坐标，请先检查 ds。")

# =========================================================
# 4. 自动识别变量名
#    latent heat flux 常见变量名是 slhf
# =========================================================
if "slhf" in ds.data_vars:
    var_name = "slhf"
else:
    # 兜底：只有一个变量时直接取它
    if len(ds.data_vars) == 1:
        var_name = list(ds.data_vars)[0]
    else:
        raise ValueError("没有识别到 slhf，请检查变量名。")

print("\nDetected variable:", var_name)

# =========================================================
# 5. 统一经度到 0~360
# =========================================================
if float(ds[lon_name].min()) < 0:
    ds = ds.assign_coords({lon_name: (ds[lon_name] % 360)})
    ds = ds.sortby(lon_name)

# =========================================================
# 6. 先裁剪区域
#    40–50N, 200–225E
# =========================================================
ds = ds.sel({lon_name: slice(200, 225)})

lat_vals = ds[lat_name].values
if lat_vals[0] > lat_vals[-1]:
    ds = ds.sel({lat_name: slice(50, 40)})
else:
    ds = ds.sel({lat_name: slice(40, 50)})

print("\n" + "=" * 80)
print("After regional crop")
print("=" * 80)
print(ds)

# =========================================================
# 7. 坐标统一成 time / lat / lon
# =========================================================
rename_dict = {}
if time_name != "time":
    rename_dict[time_name] = "time"
if lat_name != "lat":
    rename_dict[lat_name] = "lat"
if lon_name != "lon":
    rename_dict[lon_name] = "lon"

if rename_dict:
    ds = ds.rename(rename_dict)

# 删除辅助坐标
for extra in ["expver", "number"]:
    if extra in ds.coords or extra in ds.data_vars:
        try:
            ds = ds.drop_vars(extra)
        except Exception:
            pass

print("\n" + "=" * 80)
print("After rename")
print("=" * 80)
print(ds)

# =========================================================
# 8. 单位处理：目标变成平均潜热通量 W m^-2
# =========================================================
da = ds[var_name]
print("\nVariable attrs before conversion:")
for k, v in da.attrs.items():
    print(f"{k}: {v}")

units = da.attrs.get("units", "").strip().lower()

# ERA5 slhf 常见为累计量 J m^-2
if ("j" in units and "m**-2" in units) or ("j m-2" in units) or ("j m**-2" in units):
    print("\nDetected accumulated latent heat flux in J m^-2, converting to W m^-2 by dividing by 3600.")
    da = da / 3600.0
    da.attrs["units"] = "W m**-2"
    da.attrs["long_name"] = "Mean surface latent heat flux"

elif "w" in units and "m**-2" in units:
    print("\nDetected flux already in W m^-2, no unit conversion needed.")
    da.attrs["units"] = "W m**-2"
    if "long_name" not in da.attrs:
        da.attrs["long_name"] = "Mean surface latent heat flux"

else:
    print("\nWarning: units not clearly recognized.")
    print("units string =", repr(units))
    print("代码会继续跑，但你最好核对一下变量属性。")

ds[var_name] = da

# =========================================================
# 9. 时间升序
# =========================================================
ds = ds.sortby("time")

# =========================================================
# 10. 日平均
# =========================================================
ds_daily = ds.resample(time="1D").mean()

print("\n" + "=" * 80)
print("Daily mean dataset")
print("=" * 80)
print(ds_daily)

# =========================================================
# 11. 插值到 1°×1°
# =========================================================
target_lat = np.arange(40, 51, 1.0, dtype=np.float32)   # 40~50
target_lon = np.arange(200, 226, 1.0, dtype=np.float32) # 200~225

ds_interp = ds_daily.interp(
    lat=target_lat,
    lon=target_lon,
    method="linear"
)

print("\n" + "=" * 80)
print("Interpolated dataset")
print("=" * 80)
print(ds_interp)

# =========================================================
# 12. 变量名统一：改成 mslhf
# =========================================================
if var_name != "mslhf":
    ds_interp = ds_interp.rename({var_name: "mslhf"})

# =========================================================
# 13. 属性整理
# =========================================================
ds_interp["mslhf"] = ds_interp["mslhf"].astype("float32")
ds_interp["mslhf"].attrs = {
    "long_name": "Mean surface latent heat flux",
    "units": "W m**-2"
}

ds_interp["lat"].attrs = {
    "axis": "Y",
    "standard_name": "lat",
    "long_name": "Latitude",
    "units": "degrees_north"
}
ds_interp["lon"].attrs = {
    "axis": "X",
    "standard_name": "lon",
    "long_name": "Longitude",
    "units": "degrees_east"
}
ds_interp["time"].attrs = {
    "axis": "T",
    "standard_name": "time",
    "long_name": "Time"
}

ds_interp.attrs = {
    "Title": "ERA5 daily mean surface latent heat flux",
    "Source": "ERA5 processed from hourly regional file",
    "Reference": "Processed to match target daily 1x1 format"
}

# =========================================================
# 14. 保存
# =========================================================
encoding = {
    "mslhf": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    },
    "lat": {"dtype": "float32"},
    "lon": {"dtype": "float32"}
}

ds_interp.to_netcdf(outfile, encoding=encoding)

print("\n" + "=" * 80)
print("Done")
print("=" * 80)
print("Saved to:", outfile)
print(ds_interp)

print("\nFinal dims:")
print(ds_interp.sizes)

print("\nCoordinate range:")
print("time:", str(ds_interp.time.values[0]), "to", str(ds_interp.time.values[-1]))
print("lat :", float(ds_interp.lat.min()), "to", float(ds_interp.lat.max()))
print("lon :", float(ds_interp.lon.min()), "to", float(ds_interp.lon.max()))