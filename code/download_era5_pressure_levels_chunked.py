#!/usr/bin/env python3
"""Download missing ERA5 pressure-level fields in smaller day chunks.

This is a conservative fallback for periods when CDS rejects large month-wide
requests. It keeps the same variables, levels, times, and grid as
download_era5_pressure_levels.py, but writes several smaller NetCDF files per
month, such as *_198602_d01-10.nc.
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

from download_era5_pressure_levels import (
    CDSAPIRC,
    LEVELS,
    OUT_ROOT,
    VARIABLES,
    request_for_month,
    resolve_variables,
    retrieve_with_retries,
    validate_netcdf,
)


YEARS = [1986, 2023, 2024]


def day_chunks(year: int, month: int, chunk_days: int) -> list[list[int]]:
    if chunk_days < 1:
        raise ValueError("--chunk-days must be >= 1")
    _, n_days = calendar.monthrange(year, month)
    return [
        list(range(start, min(start + chunk_days, n_days + 1)))
        for start in range(1, n_days + 1, chunk_days)
    ]


def grid_label(grid: float | None) -> str:
    return "native025deg" if grid is None else f"{grid:g}deg"


def variable_label(variables: list[str]) -> str:
    if variables == VARIABLES:
        return "tuvomega"
    labels = {
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "vertical_velocity": "w",
    }
    return "-".join(labels[v] for v in variables)


def full_month_file(out_root: Path, year: int, month: int, variables: list[str], grid: float | None) -> Path:
    return (
        out_root
        / str(year)
        / f"era5_pl_{variable_label(variables)}_8lev_{grid_label(grid)}_4times_{year}{month:02d}.nc"
    )


def chunk_file(
    out_root: Path,
    year: int,
    month: int,
    days: list[int],
    variables: list[str],
    grid: float | None,
) -> Path:
    label = f"d{days[0]:02d}-{days[-1]:02d}"
    return (
        out_root
        / str(year)
        / f"era5_pl_{variable_label(variables)}_8lev_{grid_label(grid)}_4times_{year}{month:02d}_{label}.nc"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=YEARS)
    parser.add_argument("--months", nargs="*", type=int, default=list(range(1, 13)))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--grid", type=float, default=1.0)
    parser.add_argument("--area", nargs=4, type=float, metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=50)
    parser.add_argument("--retry-delay", type=int, default=1800)
    args = parser.parse_args()

    if not CDSAPIRC.exists():
        raise SystemExit(f"CDS API credentials not found: {CDSAPIRC}")

    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("cdsapi is not installed for this Python environment.") from exc

    variables = resolve_variables(args.variables)
    client = cdsapi.Client()

    print(f"[CONFIG] years={args.years} months={args.months} levels={LEVELS}")
    print(f"[CONFIG] variables={variables} grid={args.grid} chunk_days={args.chunk_days}")

    for year in args.years:
        year_dir = args.out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        for month in args.months:
            month_file = full_month_file(args.out_root, year, month, variables, args.grid)
            if month_file.exists() and not args.overwrite:
                if validate_netcdf(month_file):
                    print(f"[SKIP FULL MONTH] {month_file}")
                    continue
                raise SystemExit(f"Existing full-month file is not readable: {month_file}")

            for days in day_chunks(year, month, args.chunk_days):
                out_file = chunk_file(args.out_root, year, month, days, variables, args.grid)
                tmp_file = out_file.with_suffix(out_file.suffix + ".part")
                if out_file.exists() and not args.overwrite:
                    if validate_netcdf(out_file):
                        print(f"[SKIP CHUNK] {out_file}")
                        continue
                    raise SystemExit(f"Existing chunk file is not readable: {out_file}")
                if tmp_file.exists() and not args.overwrite:
                    raise SystemExit(
                        f"Partial download file already exists:\n{tmp_file}\n"
                        "Move it aside or rerun with --overwrite."
                    )
                if tmp_file.exists():
                    tmp_file.unlink()
                if out_file.exists() and args.overwrite:
                    out_file.unlink()

                day_label = f"{days[0]:02d}-{days[-1]:02d}"
                print(f"[DOWNLOAD CHUNK] {year}-{month:02d} days {day_label} -> {out_file}")
                request = request_for_month(year, month, variables, days, args.area, args.grid)
                retrieve_with_retries(client, request, tmp_file, args.retries, args.retry_delay)
                if not validate_netcdf(tmp_file):
                    raise SystemExit(f"Downloaded file is not readable as NetCDF: {tmp_file}")
                tmp_file.replace(out_file)
                print(f"[OK CHUNK] {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
