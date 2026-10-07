#!/usr/bin/env python
""" 
Python script to download selected files from gdex.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
import sys, os
from urllib.request import build_opener

opener = build_opener()

filelist = [
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_cnvhr.2022123118-2023031718.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_cnvhr.2023031800-2023060200.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_cnvhr.2023060206-2023081706.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_cnvhr.2023081712-2023110112.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_cnvhr.2023110118-lrghr.2023011612.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lrghr.2023011618-2023040218.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lrghr.2023040300-2023061800.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lrghr.2023061806-2023090206.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lrghr.2023090212-2023111712.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lrghr.2023111718-whr.2023020112luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lwhr.2023020118-2023041818.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lwhr.2023041900-2023070400.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lwhr.2023070406-2023091806.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lwhr.2023091812-2023120312.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_lwhr.2023120318-swhr.2023021712.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_swhr.2023021718-2023050418.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_swhr.2023050500-2023072000.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_swhr.2023072006-2023100406.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_swhr.2023100412-2023121912.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_swhr.20231219182-vdfhr.20230305122.luo84461.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_vdfhr.2023030518-2023052018.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_vdfhr.2023052100-2023080500.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_vdfhr.2023080506-2023102006.luo844617.tar',
  'https://request.gdex.ucar.edu/dsrqst/LUO844617/TarFiles/fcst_phy3m125_vdfhr.2023102012-2023123118.luo844617.tar'
]

for file in filelist:
    ofile = os.path.basename(file)
    sys.stdout.write("downloading " + ofile + " ... ")
    sys.stdout.flush()
    infile = opener.open(file)
    outfile = open(ofile, "wb")
    outfile.write(infile.read())
    outfile.close()
    sys.stdout.write("done\n")
