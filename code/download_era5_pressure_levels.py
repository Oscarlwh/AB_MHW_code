#!/usr/bin/env python3
"""Download missing ERA5 pressure-level fields for Qd reconstruction.

Requires a configured CDS API account (`~/.cdsapirc`) and the `cdsapi` package.
By default the script requests official native 0.25 degree ERA5 files for
8 pressure levels and 4 synoptic times. Use `--grid 1.0` for CDS-side
1 degree output.
"""

from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path


YEARS = [1986, 2023, 2024]
LEVELS = ["1000", "925", "850", "700", "500", "400", "300", "250"]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
VARIABLE_ALIASES = {
    "t": "temperature",
    "temperature": "temperature",
    "u": "u_component_of_wind",
    "u_component_of_wind": "u_component_of_wind",
    "v": "v_component_of_wind",
    "v_component_of_wind": "v_component_of_wind",
    "w": "vertical_velocity",
    "omega": "vertical_velocity",
    "vertical_velocity": "vertical_velocity",
}
OUT_ROOT = Path("/path/to/data/ERA5_raw")
CDSAPIRC = Path.home() / ".cdsapirc"


def days_for_month(year: int, month: int, days: list[int] | None = None) -> list[str]:
    if days is not None:
        _, n_days = calendar.monthrange(year, month)
        bad = [day for day in days if day < 1 or day > n_days]
        if bad:
            raise ValueError(f"Invalid day(s) for {year}-{month:02d}: {bad}")
        return [f"{d:02d}" for d in days]
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def request_for_month(
    year: int,
    month: int,
    variables: list[str],
    days: list[int] | None = None,
    area: list[float] | None = None,
    grid: float | None = None,
) -> dict:
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "pressure_level": LEVELS,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days_for_month(year, month, days),
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if area is not None:
        request["area"] = area
    if grid is not None:
        request["grid"] = [grid, grid]
    return request


def resolve_variables(values: list[str] | None) -> list[str]:
    if not values:
        return VARIABLES
    variables = []
    for value in values:
        key = value.strip()
        try:
            variable = VARIABLE_ALIASES[key]
        except KeyError as exc:
            choices = ", ".join(sorted(VARIABLE_ALIASES))
            raise SystemExit(f"Unknown variable {value!r}. Choices/aliases: {choices}") from exc
        if variable not in variables:
            variables.append(variable)
    return variables


def validate_netcdf(path: Path) -> bool:
    try:
        import xarray as xr
    except ImportError:
        return True

    try:
        with xr.open_dataset(path, engine="netcdf4"):
            return True
    except Exception:
        return False


def retrieve_with_retries(client, request: dict, tmp_file: Path, retries: int, retry_delay: int) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[TRY {attempt}/{retries}] {tmp_file}")
            client.retrieve("reanalysis-era5-pressure-levels", request, str(tmp_file))
            return
        except Exception as exc:
            last_error = exc
            print(f"[RETRY] download failed on attempt {attempt}/{retries}: {exc!r}")
            if tmp_file.exists():
                tmp_file.unlink()
            if attempt < retries:
                print(f"[WAIT] sleeping {retry_delay} seconds before retry")
                time.sleep(retry_delay)
    raise RuntimeError(f"Download failed after {retries} attempts: {last_error!r}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=YEARS)
    parser.add_argument("--months", nargs="*", type=int, default=list(range(1, 13)))
    parser.add_argument(
        "--days",
        nargs="*",
        type=int,
        help="Optional day subset applied to every requested month, e.g. --days 1.",
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--variables",
        nargs="*",
        help="Variable subset using CDS names or aliases: t, u, v, w/omega. Default is all four.",
    )
    parser.add_argument(
        "--grid",
        type=float,
        help="Optional CDS output grid in degrees, e.g. --grid 1.0. Omit for native 0.25 degree.",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Optional CDS area subset. Omit for global native 0.25 degree data.",
    )
    parser.add_argument(
        "--confirm-global-025",
        action="store_true",
        help="Required when downloading global native-resolution 0.25 degree data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=int, default=120)
    args = parser.parse_args()
    variables = resolve_variables(args.variables)

    is_native_global = args.area is None and args.grid is None
    if is_native_global and not args.confirm_global_025:
        raise SystemExit(
            "Refusing to start a global native 0.25 degree download without explicit confirmation.\n"
            "One month is about 6-7 GB; 1986/2023/2024 together are roughly 240 GB.\n"
            "Rerun with --confirm-global-025 if you really want the full global official grid,\n"
            "use --grid 1.0 for CDS-side 1 degree output, or use --area NORTH WEST SOUTH EAST "
            "for a regional subset."
        )

    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "cdsapi is not installed for this Python environment. "
            "Install/configure cdsapi or use the CDS website download path."
        ) from exc

    if not CDSAPIRC.exists():
        raise SystemExit(
            f"CDS API credentials not found: {CDSAPIRC}\n"
            "Create this file from your Copernicus CDS account API token, then rerun.\n"
            "Expected format:\n"
            "url: https://cds.climate.copernicus.eu/api\n"
            "key: <your-personal-access-token>\n"
        )

    client = cdsapi.Client()

    for year in args.years:
        year_dir = args.out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        for month in args.months:
            grid_label = "native025deg" if args.grid is None else f"{args.grid:g}deg"
            var_label = "tuvomega" if variables == VARIABLES else "-".join(
                {"temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v", "vertical_velocity": "w"}[v]
                for v in variables
            )
            day_label = "" if args.days is None else "_" + "-".join(f"{d:02d}" for d in args.days)
            out_file = year_dir / f"era5_pl_{var_label}_8lev_{grid_label}_4times_{year}{month:02d}{day_label}.nc"
            tmp_file = out_file.with_suffix(out_file.suffix + ".part")
            if out_file.exists() and not args.overwrite:
                if validate_netcdf(out_file):
                    print(f"[SKIP] {out_file}")
                    continue
                raise SystemExit(
                    f"Existing file is not a readable NetCDF, likely an interrupted download:\n"
                    f"{out_file}\n"
                    "Move it aside or rerun with --overwrite."
                )
            if tmp_file.exists() and not args.overwrite:
                raise SystemExit(
                    f"Partial download file already exists:\n{tmp_file}\n"
                    "Move it aside or rerun with --overwrite."
                )
            if tmp_file.exists():
                tmp_file.unlink()
            if out_file.exists() and args.overwrite:
                out_file.unlink()
            print(f"[DOWNLOAD] {year}-{month:02d} -> {out_file}")
            retrieve_with_retries(
                client,
                request_for_month(year, month, variables, args.days, args.area, args.grid),
                tmp_file,
                args.retries,
                args.retry_delay,
            )
            if not validate_netcdf(tmp_file):
                raise SystemExit(f"Downloaded file is not readable as NetCDF: {tmp_file}")
            tmp_file.replace(out_file)
            print(f"[OK] {out_file}")


if __name__ == "__main__":
    main()
