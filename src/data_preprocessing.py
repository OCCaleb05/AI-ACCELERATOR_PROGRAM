import os
import glob
import zipfile
import shutil
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.wkt import loads
import warnings

# Suppress minor geospatial warnings for a clean terminal
warnings.filterwarnings('ignore')

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw_geotiff')
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'temp_unzip')

os.makedirs(TEMP_DIR, exist_ok=True)

# Our tactical operational zones (WKT Polygons)
REGIONS = {
    'lokoja_confluence_2022': 'POLYGON((6.5 7.5, 7.0 7.5, 7.0 8.0, 6.5 8.0, 6.5 7.5))',
    'borno_basin_2022': 'POLYGON((12.5 11.5, 13.5 11.5, 13.5 12.5, 12.5 12.5, 12.5 11.5))',
    'bayelsa_coast_2022': 'POLYGON((5.5 4.2, 6.5 4.2, 6.5 5.2, 5.5 5.2, 5.5 4.2))'
}

def find_band_file(extracted_dir, band_suffix):
    search_pattern = os.path.join(extracted_dir, '**', f'*{band_suffix}')
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"Could not find band {band_suffix} in {extracted_dir}")
    return files[0]

def process_region(region_name, wkt_polygon):
    zip_file = os.path.join(RAW_DIR, f"{region_name}_raw.zip")
    
    if not os.path.exists(zip_file):
        print(f"Skipping {region_name}: Raw .zip not found.")
        return

    print(f"\n--- Forging NDWI Data for {region_name} ---")
    extracted_folder = os.path.join(TEMP_DIR, region_name)
    
    print("1. Unzipping 1GB payload...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    try:
        # Locate the 10m Optical Bands
        b2_path = find_band_file(extracted_folder, 'B02_10m.jp2') # Blue
        b3_path = find_band_file(extracted_folder, 'B03_10m.jp2') # Green
        b4_path = find_band_file(extracted_folder, 'B04_10m.jp2') # Red
        b8_path = find_band_file(extracted_folder, 'B08_10m.jp2') # Near Infrared (NIR)
        
        geom = loads(wkt_polygon)
        geom_geojson = {"type": "Polygon", "coordinates": [list(geom.exterior.coords)]}
        
        cropped_bands = []
        out_meta = None

        print("2. Cropping optical tensors to operational zone...")
        for band_path in [b2_path, b3_path, b4_path, b8_path]:
            with rasterio.open(band_path) as src:
                transformed_geom = transform_geom('EPSG:4326', src.crs, geom_geojson)
                out_image, out_transform = mask(src, [transformed_geom], crop=True)
                cropped_bands.append(out_image[0]) 
                
                if out_meta is None:
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "count": 4 
                    })

        # Stack into a single tensor (Blue=0, Green=1, Red=2, NIR=3)
        optical_tensor = np.stack(cropped_bands)

        # --- THE NDWI UPGRADE ---
        print("3. Generating pure NDWI ground truth mask from light physics...")
        green_band = optical_tensor[1].astype(np.float32)
        nir_band = optical_tensor[3].astype(np.float32)
        
        # NDWI Formula: (Green - NIR) / (Green + NIR)
        # We add 1e-8 to the denominator to prevent division by zero on dead pixels
        ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-8)
        
        # Any index greater than 0.0 is classified as physical water
        binary_mask = (ndwi > 0.0).astype(np.uint8)
        binary_mask = np.expand_dims(binary_mask, axis=0) # Shape to (1, H, W)

        print("4. Saving AI-ready .tif tensors...")
        opt_out_path = os.path.join(RAW_DIR, f"{region_name}_opt.tif")
        mask_out_path = os.path.join(RAW_DIR, f"{region_name}_mask.tif")

        with rasterio.open(opt_out_path, "w", **out_meta) as dest:
            dest.write(optical_tensor)

        mask_meta = out_meta.copy()
        mask_meta.update({"count": 1, "dtype": 'uint8'})
        with rasterio.open(mask_out_path, "w", **mask_meta) as dest:
            dest.write(binary_mask)
            
        print(f"Success! {region_name} forged with perfect NDWI mapping.")

    except Exception as e:
        print(f"Error processing {region_name}: {str(e)}")
        
    finally:
        print("5. Sweeping temporary files...")
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)

if __name__ == "__main__":
    for region, wkt in REGIONS.items():
        process_region(region, wkt)
    print("\n--- ALL PREPROCESSING COMPLETE ---")
    print("You may now run 'python lab.py' to begin deep learning on REAL satellite data!")