import os
import numpy as np
import xarray as xr

# =========================================================
# 1. 输入输出路径
# =========================================================
infile = "/path/to/data/era5_lwh/era5_surface_sensible_heat_flux_2024_200E-225E_40N-50N.nc"
outdir = os.path.dirname(infile)
outfile = os.path.join(outdir, "ERA5_msshf_daily_1x1_200E-225E_40N-50N_2024.nc")

# =========================================================
# 2. 打开文件
# =========================================================
ds = xr.open_dataset(infile)

print("=" * 80)
print("Original dataset")
print("=" * 80)
print(ds)

# =========================================================
# 3. 统一坐标名
# =========================================================
rename_dict = {}
if "valid_time" in ds.coords:
    rename_dict["valid_time"] = "time"
if "latitude" in ds.coords:
    rename_dict["latitude"] = "lat"
if "longitude" in ds.coords:
    rename_dict["longitude"] = "lon"

if rename_dict:
    ds = ds.rename(rename_dict)

# 删掉无关辅助坐标
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
# 4. 检查变量
# =========================================================
if "sshf" not in ds.data_vars:
    raise ValueError("没有找到变量 sshf，请检查文件。")

da = ds["sshf"]

print("\nsshf attrs:")
for k, v in da.attrs.items():
    print(f"{k}: {v}")

# =========================================================
# 5. 单位处理
#    目标：转成平均感热通量 W m^-2
# =========================================================
units = da.attrs.get("units", "").strip().lower()

# 常见情况：sshf 是累计量 J m^-2
if ("j" in units and "m**-2" in units) or ("j m-2" in units) or ("j m**-2" in units):
    print("\nDetected accumulated flux in J m^-2, converting to W m^-2 by dividing by 3600.")
    da = da / 3600.0
    da.attrs["units"] = "W m**-2"
    da.attrs["long_name"] = "Mean surface sensible heat flux"

# 如果原本已经是 W/m²，就直接用
elif "w" in units and "m**-2" in units:
    print("\nDetected flux already in W m^-2, no unit conversion needed.")
    da.attrs["units"] = "W m**-2"
    if "long_name" not in da.attrs:
        da.attrs["long_name"] = "Mean surface sensible heat flux"

else:
    print("\nWarning: units not clearly recognized, keeping original values. Please double-check attrs.")
    print("Recognized units string =", repr(units))

# 用处理后的 da 替换
ds["sshf"] = da

# =========================================================
# 6. 时间升序
# =========================================================
ds = ds.sortby("time")

# =========================================================
# 7. 做日平均
# =========================================================
ds_daily = ds.resample(time="1D").mean()

print("\n" + "=" * 80)
print("Daily mean dataset")
print("=" * 80)
print(ds_daily)

# =========================================================
# 8. 插值到 1°×1°
#    按师兄格式，lat 升序
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
# 9. 改变量名：sshf -> msshf
#    这样更贴近师兄文件风格
# =========================================================
ds_interp = ds_interp.rename({"sshf": "msshf"})

# =========================================================
# 10. 属性整理
# =========================================================
ds_interp["msshf"] = ds_interp["msshf"].astype("float32")
ds_interp["msshf"].attrs = {
    "long_name": "Mean surface sensible heat flux",
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
    "Title": "ERA5 daily mean surface sensible heat flux",
    "Source": "ERA5 processed from hourly regional file",
    "Reference": "Processed to match target daily 1x1 format"
}

# =========================================================
# 11. 保存
# =========================================================
encoding = {
    "msshf": {
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