# Alaska blocking and Northeast Pacific marine heatwaves

Python analysis and plotting scripts for the study of coupled feedback between Alaska blocking (AB) and early-winter Northeast Pacific marine heatwaves (MHWs).

## Contents

All  Python source files are in the single `code/` directory.

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



## Data and environment

Inputs comprise NOAA OISST, ERA5 atmospheric fields, JRA-55 heating components, and CAM5 simulation outputs. Raw data, manuscripts, account-specific download requests, and simulation output files are not distributed here. CAM5 experiment setup files are not included.

Python packages are listed in `requirements.txt`. GRIB processing requires ecCodes; geographic plotting uses Cartopy and its map resources. 
