#!/usr/bin/env python3

import sys

sys.path.insert(0, "/scratch/elena/Geometry_WCTE")

from Geometry.Device import Device

hall = Device.open_file("/scratch/elena/Geometry_WCTE/examples/wcte_bldg157.geo")
wcd = hall.wcds[0]

print("Number of mPMTs:", len(wcd.mpmts))

print("\n=== mPMTs ===")
for idx, mpmt in enumerate(wcd.mpmts[:5]):
    print(f"enumerate = {idx}")
    print(f"number    = {getattr(mpmt, 'number', 'N/A')}")
    print(f"i         = {getattr(mpmt, 'i', 'N/A')}")
    print(f"name      = {getattr(mpmt, 'name', 'N/A')}")
    print()

pmt = wcd.mpmts[0].pmts[0]

print("\n=== PMT attributes ===")
print(dir(pmt))

print("\n=== PMT values ===")
print(f"number = {getattr(pmt, 'number', 'N/A')}")
print(f"i      = {getattr(pmt, 'i', 'N/A')}")
print(f"name   = {getattr(pmt, 'name', 'N/A')}")


print("\n=== First PMT placement ===")
print(pmt.get_placement("design"))