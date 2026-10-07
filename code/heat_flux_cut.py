import os
import xarray as xr
import numpy as np

# =========================================================
# 1. 输入输出路径
# =========================================================
infile = "/path/to/data/era5_lwh/era5_surface_sensible_heat_flux_2024.nc"
outdir = os.path.dirname(infile)
outfile = os.path.join(
    outdir,
    "era5_surface_sensible_heat_flux_2024_200E-225E_40N-50N.nc"
)

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
# 3. 自动识别经纬度坐标名
# =========================================================
lat_name = None
lon_name = None

for name in ds.coords:
    low = name.lower()
    if low in ["latitude", "lat"]:
        lat_name = name
    elif low in ["longitude", "lon"]:
        lon_name = name

if lat_name is None or lon_name is None:
    raise ValueError("没有识别到纬度/经度坐标，请先 print(ds) 检查。")

print("\nDetected coordinates:")
print("lat =", lat_name)
print("lon =", lon_name)

# =========================================================
# 4. 统一经度到 0~360
#    135–160W 对应 200–225E
# =========================================================
lon = ds[lon_name]

if float(lon.min()) < 0:
    ds = ds.assign_coords({lon_name: (ds[lon_name] % 360)})
    ds = ds.sortby(lon_name)

# =========================================================
# 5. 裁剪经度：200E–225E
# =========================================================
ds = ds.sel({lon_name: slice(200, 225)})

# =========================================================
# 6. 裁剪纬度：40N–50N
# =========================================================
lat_vals = ds[lat_name].values

if lat_vals[0] > lat_vals[-1]:
    ds = ds.sel({lat_name: slice(50, 40)})
else:
    ds = ds.sel({lat_name: slice(40, 50)})

# =========================================================
# 7. 压缩保存
# =========================================================
encoding = {}
for var in ds.data_vars:
    encoding[var] = {
        "zlib": True,
        "complevel": 4
    }

ds.to_netcdf(outfile, encoding=encoding)

print("\n" + "=" * 80)
print("Done")
print("=" * 80)
print("Saved to:", outfile)
print(ds)

# =========================================================
# 8. 简单检查裁剪结果
# =========================================================
print("\nFinal dims:")
print(ds.sizes)

print("\nLatitude range:")
print(float(ds[lat_name].min()), "to", float(ds[lat_name].max()))

print("\nLongitude range:")
print(float(ds[lon_name].min()), "to", float(ds[lon_name].max()))