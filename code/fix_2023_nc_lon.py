from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm

ROOT = Path("/path/to/data/dia")

files = sorted(ROOT.glob("*/*/*luo844617.nc"))

print(f"Found {len(files)} files")

for path in tqdm(files, desc="Fixing longitude"):
    ds = xr.open_dataset(path)

    if "g0_lon_2" not in ds.coords:
        ds.close()
        print(f"[SKIP] no g0_lon_2: {path}")
        continue

    lon = ds["g0_lon_2"].values

    # 如果已经是 0–360，就跳过
    if np.nanmin(lon) >= 0 and np.isclose(lon[0], 0.0):
        ds.close()
        continue

    # 把 -1.25 转成 358.75，然后重新排序
    lon_new = np.mod(lon, 360.0).astype("float32")

    ds = ds.assign_coords(g0_lon_2=lon_new)
    ds = ds.sortby("g0_lon_2")

    # 经纬度转 float32，贴近旧文件
    if "g0_lat_1" in ds.coords:
        ds = ds.assign_coords(g0_lat_1=ds["g0_lat_1"].astype("float32"))
    ds = ds.assign_coords(g0_lon_2=ds["g0_lon_2"].astype("float32"))

    # 写临时文件再替换，防止中途损坏原文件
    tmp = path.with_suffix(".tmp.nc")

    encoding = {}
    for v in ds.data_vars:
        if ds[v].dtype == "float32":
            encoding[v] = {"zlib": True, "complevel": 4, "dtype": "float32"}

    ds.to_netcdf(tmp, encoding=encoding)
    ds.close()

    tmp.replace(path)

print("Done.")