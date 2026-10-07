# Alaska blocking and Northeast Pacific marine heatwaves

Python analysis and plotting scripts for the study of coupled feedback between Alaska blocking (AB) and early-winter Northeast Pacific marine heatwaves (MHWs).

## Contents

All 60 Python source files are in the single `code/` directory.

The source tree contains MHW identification, atmospheric data preprocessing, blocking detection, thermodynamic and transient-eddy diagnostics, and figure generation. `figure_entrypoints.json` identifies the entry scripts and preprocessing steps for main-text Figures 1–5, supplementary Figures S1–S5.

| Figure | Entry script |
|---|---|
| 1 | `code/figure_01_ab_evolution.py` |
| 2 | `code/figure_02_z500_evolution.py` |
| 3 | `code/figure_03_heat_flux_heating.py` |
| 4 | `code/figure_04_geopotential_tendencies_eady.py` |
| 5 | `code/figure_05_cam5_response.py` |
| S1 | `code/figure_s01_mhw_intensity.py` |
| S2 | `code/figure_s02_vertical_evolution.py` |
| S3 | `code/figure_s03_total_precipitation.py` |
| S4 | `code/figure_s04_convective_precipitation.py` |
| S5 | `code/figure_s05_eddy_geopotential_tendency.py` |


All scripts are in `code/`. Figure entrypoints are named `figure_01`–`figure_05` and `figure_s01`–`figure_s05`; the remaining scripts prepare data or provide diagnostic and plotting helpers. Unused alternate plot versions are excluded.

## Data and environment

Inputs comprise NOAA OISST, ERA5 atmospheric fields, JRA-55 heating components, and CAM5 simulation outputs. Raw data, manuscripts, account-specific download requests, and simulation output files are not distributed here. CAM5 experiment setup files are not included; the CAM5 script performs postprocessing of existing ensemble output.

Python packages are listed in `requirements.txt`. GRIB processing requires ecCodes; geographic plotting uses Cartopy and its map resources. Some preprocessing scripts use the external `ncdump` utility.

## Running the scripts

1. Install the dependencies and prepare the input datasets.
2. Replace `/path/to/user`, `/path/to/data`, and other input/output paths in the selected scripts with local paths. These placeholders replace machine-specific paths; scientific calculations are unchanged.
3. Follow the preprocessing entries in `figure_entrypoints.json` before executing the corresponding figure script. Keep the directory structure intact for local imports.
4. Execute the selected script in an environment containing its dependencies. For example:

```bash
python code/figure_05_cam5_response.py \
  --input /path/to/CAM5/output \
  --output /path/to/figures \
  --only-z500-days15-29 --z500-no-boxes
```

These are source scripts, not a one-command workflow. Figure scripts retain their original processing and cache dependencies. Figure 3's existing entry script also runs legacy temperature-budget plotting steps; those outputs are not part of the manuscript figure set. Input heating and turbulent-flux anomalies use the baseline periods defined in their preprocessing scripts.

The 14 MHW peak dates are listed in the relevant composite scripts; the 13-event composite excludes the 2004 event. Figure-specific averaging windows are encoded in their individual scripts. Figure 1 uses centered 5-day averages and pointwise case-bootstrap intervals; model integration days in Figure 5 refer to the numerical experiment, not observed MHW lead/lag days.

## Source integrity

`source_checksums.json` contains SHA-256 checksums for the released Python files. The released files have passed Python syntax checks. This check does not run the numerical workflow. Existing attribution in third-party source files is retained.
