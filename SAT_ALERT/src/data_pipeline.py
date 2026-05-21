import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
import tensorflow as tf

# Define paths for real geospatial data (e.g., from DSA Data Center or ESA Copernicus)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_geotiff')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_patches')

def ensure_dirs():
    """Ensure data directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, 'val'), exist_ok=True)

def extract_patches(image_path, mask_path, patch_size=256, stride=256):
    """
    Reads a large multi-band GeoTIFF and slices it into uniform patches.
    Normalizes 16-bit satellite data to [0, 1].
    """
    patches_img = []
    patches_mask = []

    with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
        width, height = src_img.width, src_img.height
        
        # Slide a window across the large satellite image
        for top in range(0, height - patch_size + 1, stride):
            for left in range(0, width - patch_size + 1, stride):
                window = Window(left, top, patch_size, patch_size)
                
                # Read Multi-spectral bands (e.g., Red, Green, Blue, Near-Infrared)
                # Transpose to (Height, Width, Channels) for TensorFlow
                img_patch = src_img.read([1, 2, 3, 4], window=window).transpose(1, 2, 0)
                
                # Normalize 16-bit data (0-65535) to 0.0 - 1.0
                img_patch = img_patch.astype(np.float32) / 10000.0
                img_patch = np.clip(img_patch, 0.0, 1.0)
                
                # Read corresponding binary risk/inundation mask
                mask_patch = src_mask.read(1, window=window)
                mask_patch = np.expand_dims(mask_patch, axis=-1).astype(np.float32)
                
                patches_img.append(img_patch)
                patches_mask.append(mask_patch)

    return np.array(patches_img), np.array(patches_mask)

def process_and_split_dataset(region_dict, patch_size=256):
    """
    Processes GeoTIFFs and enforces SPATIAL SPLITTING.
    Example region_dict: 
    { 'train': ['north_east_img1.tif', 'benue_basin.tif'],
      'val': ['niger_delta_img1.tif'] }
    """
    ensure_dirs()
    
    for split, files in region_dict.items():
        print(f"Processing {split} split...")
        all_img_patches = []
        all_mask_patches = []
        
        for base_name in files:
            img_path = os.path.join(DATA_DIR, f"{base_name}_opt.tif")
            mask_path = os.path.join(DATA_DIR, f"{base_name}_mask.tif")
            
            # Mocking the extraction if files don't exist yet for the MVP testing phase
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found. Skipping to next.")
                continue
                
            imgs, masks = extract_patches(img_path, mask_path, patch_size)
            all_img_patches.extend(imgs)
            all_mask_patches.extend(masks)
            
        if all_img_patches:
            # Save patches as compressed numpy arrays for rapid TensorFlow loading
            save_path = os.path.join(PROCESSED_DIR, split, 'dataset.npz')
            np.savez_compressed(save_path, images=np.array(all_img_patches), masks=np.array(all_mask_patches))
            print(f"Saved {len(all_img_patches)} patches to {save_path}")

def build_tf_dataset(split='train', batch_size=16):
    """
    Loads processed patches into a highly optimized tf.data.Dataset pipeline.
    Includes data augmentation for the training set.
    """
    data_path = os.path.join(PROCESSED_DIR, split, 'dataset.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")
        
    data = np.load(data_path)
    images, masks = data['images'], data['masks']
    
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    # NEW: Bulletproof sanitization. Force all mask pixels to strict 0.0 or 1.0
    dataset = dataset.map(
        lambda x, y: (x, tf.cast(y > 0.5, tf.float32)), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if split == 'train':
        # Apply standard satellite augmentations
        dataset = dataset.map(
            lambda x, y: (tf.image.random_flip_left_right(x), tf.image.random_flip_left_right(y)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(
            lambda x, y: (tf.image.random_flip_up_down(x), tf.image.random_flip_up_down(y)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.shuffle(buffer_size=500)
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    # Example operational execution mapping specific geographic sectors
    regions = {
        'train': ['lokoja_confluence_2022', 'borno_basin_2022'],
        'val':   ['bayelsa_coast_2022'] # Spatial split to ensure generalized tactical mobility forecasting
    }
    process_and_split_dataset(regions)