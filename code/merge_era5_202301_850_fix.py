#!/usr/bin/env python3
"""Merge the corrected 2023-01 ERA5 850 hPa layer into the 8-level file."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


RAW_DIR = Path("/path/to/data/ERA5_raw/2023")
BAD_FILE = RAW_DIR / "era5_pl_tuvomega_8lev_native025deg_4times_202301_BADlevels_1000-925-825-700-500-400-300-250.nc"
FIX_850_FILE = RAW_DIR / "era5_pl_tuvomega_8lev_native025deg_4times_202301_BADlevels_850.nc"
OUT_FILE = RAW_DIR / "era5_pl_tuvomega_8lev_native025deg_4times_202301.nc"
EXPECTED_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250]


def main() -> None:
    if not BAD_FILE.exists():
        raise FileNotFoundError(BAD_FILE)
    if not FIX_850_FILE.exists():
        raise FileNotFoundError(FIX_850_FILE)

    bad = xr.open_dataset(BAD_FILE, chunks="auto", engine="netcdf4")
    fix = xr.open_dataset(FIX_850_FILE, chunks="auto", engine="netcdf4")
    try:
        level_name = "pressure_level"
        base = bad.sel({level_name: [1000, 925, 700, 500, 400, 300, 250]})
        fixed = fix.sel({level_name: [850]})
        merged = xr.concat([base, fixed], dim=level_name)
        merged = merged.sel({level_name: EXPECTED_LEVELS})
        merged.attrs.update(bad.attrs)
        merged.attrs["level_fix"] = f"Replaced erroneous 825 hPa with corrected 850 hPa from {FIX_850_FILE.name}"
        for name in merged.data_vars:
            merged[name].attrs.update(bad[name].attrs)

        tmp_file = OUT_FILE.with_suffix(OUT_FILE.suffix + ".tmp")
        if tmp_file.exists():
            tmp_file.unlink()
        encoding = {name: {"zlib": True, "complevel": 1, "dtype": "float32"} for name in merged.data_vars}
        print(f"[WRITE] {tmp_file}")
        merged.to_netcdf(tmp_file, encoding=encoding)
        with xr.open_dataset(tmp_file, engine="netcdf4") as check:
            levels = [int(x) for x in check[level_name].values.tolist()]
            if levels != EXPECTED_LEVELS:
                raise ValueError(f"Unexpected merged levels: {levels}")
            missing = sorted(set(["t", "u", "v", "w"]) - set(check.data_vars))
            if missing:
                raise KeyError(f"Missing variables: {missing}")
        tmp_file.replace(OUT_FILE)
        print(f"[OK] {OUT_FILE}")
    finally:
        bad.close()
        fix.close()


if __name__ == "__main__":
    main()
