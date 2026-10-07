import os
import numpy as np
import xarray as xr

# =========================================================
# 0. 路径
# =========================================================
workdir = "/path/to/data/1"

file_msshf = os.path.join(workdir, "ERA5_msshf_daily_1x1_200E-225E_40N-50N_1981_2024_NOLEAP.nc")
file_mslhf = os.path.join(workdir, "ERA5_mslhf_daily_1x1_200E-225E_40N-50N_1981_2024_NOLEAP.nc")

out_msshf = os.path.join(workdir, "ERA5_msshf_daily_1x1_200E-225E_40N-50N_1981_2024_NOLEAP_detrended.nc")
out_mslhf = os.path.join(workdir, "ERA5_mslhf_daily_1x1_200E-225E_40N-50N_1981_2024_NOLEAP_detrended.nc")

# =========================================================
# 1. 逐格点去线性趋势函数
# =========================================================
def detrend_along_time(da):
    """
    对 DataArray(time, lat, lon) 做逐格点线性去趋势
    返回：去趋势后的 DataArray
    """
    if da.dims != ("time", "lat", "lon"):
        da = da.transpose("time", "lat", "lon")

    # 时间索引：0,1,2,...,N-1
    t = xr.DataArray(
        np.arange(da.sizes["time"], dtype=np.float64),
        dims=["time"],
        coords={"time": da["time"]}
    )

    # 时间均值
    t_mean = t.mean("time")
    x_mean = da.mean("time")

    # 最小二乘斜率
    cov_tx = ((t - t_mean) * (da - x_mean)).mean("time")
    var_t = ((t - t_mean) ** 2).mean("time")
    slope = cov_tx / var_t

    # 截距
    intercept = x_mean - slope * t_mean

    # 线性趋势项
    trend = slope * t + intercept

    # 去趋势，并加回原时间均值
    da_detrended = da - trend + x_mean

    return da_detrended.astype("float32"), slope.astype("float32"), intercept.astype("float32")

# =========================================================
# 2. 单变量处理函数
# =========================================================
def process_one_file(infile, varname, outfile):
    ds = xr.open_dataset(infile)

    print("\n" + "=" * 90)
    print(f"Processing {varname}")
    print("=" * 90)
    print(ds)

    da = ds[varname]

    # 去趋势
    da_det, slope, intercept = detrend_along_time(da)

    # 输出数据集
    ds_out = xr.Dataset(
        {
            varname: da_det,
            f"{varname}_trend_slope": slope,
            f"{varname}_trend_intercept": intercept,
        },
        coords={
            "time": ds["time"],
            "lat": ds["lat"],
            "lon": ds["lon"],
        },
        attrs=ds.attrs
    )

    # 属性整理
    ds_out[varname].attrs = da.attrs.copy()
    ds_out[varname].attrs["note"] = "Linearly detrended at each grid point along time, with temporal mean added back"

    ds_out[f"{varname}_trend_slope"].attrs = {
        "long_name": f"Linear trend slope of {varname}",
        "units": f"{da.attrs.get('units', '')} per time step"
    }
    ds_out[f"{varname}_trend_intercept"].attrs = {
        "long_name": f"Linear trend intercept of {varname}",
        "units": da.attrs.get("units", "")
    }

    encoding = {
        varname: {"zlib": True, "complevel": 4, "dtype": "float32"},
        f"{varname}_trend_slope": {"zlib": True, "complevel": 4, "dtype": "float32"},
        f"{varname}_trend_intercept": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "lat": {"dtype": "float32"},
        "lon": {"dtype": "float32"},
    }

    ds_out.to_netcdf(outfile, encoding=encoding)

    print(f"\n[OK] Saved: {outfile}")
    print("\nOutput dataset:")
    print(ds_out)

    # 简单检查
    print("\nQuick check:")
    print("original mean   =", float(da.mean().values))
    print("detrended mean  =", float(ds_out[varname].mean().values))
    print("slope min/max   =", float(slope.min().values), float(slope.max().values))

# =========================================================
# 3. 处理 msshf / mslhf
# =========================================================
process_one_file(file_msshf, "msshf", out_msshf)
process_one_file(file_mslhf, "mslhf", out_mslhf)

print("\nDone.")