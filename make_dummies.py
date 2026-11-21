import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

# Ensure the directory exists
os.makedirs('data/raw_geotiff', exist_ok=True)

print("Generating valid dummy GeoTIFFs for all required regions...")

# The regions expected by lab.py
regions = ['lokoja_confluence_2022', 'borno_basin_2022', 'bayelsa_coast_2022']

# Fake geographic coordinates
transform = from_origin(6.73, 7.80, 0.0001, 0.0001)

for region in regions:
    # Create fake satellite data (4 bands: RGB + NIR, 512x512 pixels)
    dummy_img = np.random.randint(0, 65535, (4, 512, 512), dtype=np.uint16)
    
    # Create a fake binary mask (1 band, 512x512 pixels, 0 or 1 for flood risk)
    dummy_mask = np.random.randint(0, 2, (1, 512, 512), dtype=np.uint8)

    # Save the Optical Image
    with rasterio.open(
        f'data/raw_geotiff/{region}_opt.tif', 'w', driver='GTiff',
        height=512, width=512, count=4, dtype=dummy_img.dtype,
        crs='+proj=latlong', transform=transform,
    ) as dst:
        dst.write(dummy_img)

    # Save the Risk Mask
    with rasterio.open(
        f'data/raw_geotiff/{region}_mask.tif', 'w', driver='GTiff',
        height=512, width=512, count=1, dtype=dummy_mask.dtype,
        crs='+proj=latlong', transform=transform,
    ) as dst:
        dst.write(dummy_mask)

print("Success! Valid dummy GeoTIFFs generated for Lokoja, Borno, and Bayelsa.")