import os
import numpy as np
import rasterio
import tensorflow as tf
from src.models import build_terrain_unet, build_fusion_model
from src.data_pipeline import process_and_split_dataset, build_tf_dataset

# Define directories for saving our deep learning models
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw_geotiff')
WEATHER_DIR = os.path.join(BASE_DIR, 'data', 'processed_weather')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_terrain_model():
    """
    Trains the U-Net architecture for high-resolution topographic inundation mapping.
    Uses the spatial splits from our real GeoTIFF data pipeline.
    """
    print("\n--- Initiating Terrain U-Net Training ---")
    
    # 1. Define Geographic Splits (Ensure these match the files downloaded)
    regions = {
        'train': ['lokoja_confluence_2022', 'borno_basin_2022'],
        'val':   ['bayelsa_coast_2022'] 
    }
    
    # Process the raw GeoTIFFs into 256x256 patches
    print("Processing Geospatial Data...")
    process_and_split_dataset(regions)
    
    # Load optimized tf.data pipelines
    try:
        train_dataset = build_tf_dataset(split='train', batch_size=16)
        val_dataset = build_tf_dataset(split='val', batch_size=16)
    except FileNotFoundError:
        print("CRITICAL: Processed data not found. Please ensure raw .tif files are in the data/raw_geotiff folder.")
        return

    # 2. Initialize the Model
    model = build_terrain_unet(input_shape=(256, 256, 4))
    
    # 3. Define Training Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'terrain_unet_best.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, # Stop if validation loss doesn't improve for 5 epochs
            restore_best_weights=True
        )
    ]
    
    # 4. Execute Deep Learning Training
    print("Starting Training Loop...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30, # Real deep learning requires multiple passes over the data
        callbacks=callbacks
    )
    print("--- Terrain U-Net Training Complete ---")

def extract_fusion_data(regions_list):
    """
    Extracts aligned optical image patches, weather time-series tensors,
    and continuous flood risk labels for each region.
    """
    all_imgs, all_ts, all_labels = [], [], []

    for region in regions_list:
        # Build paths for optical data, flood mask, and weather time-series
        opt_path = os.path.join(RAW_DIR, f"{region}_opt.tif")
        mask_path = os.path.join(RAW_DIR, f"{region}_mask.tif")
        ts_path = os.path.join(WEATHER_DIR, f"{region}_ts.npy")

        # Skip regions missing any required input file
        if not (os.path.exists(opt_path) and os.path.exists(ts_path) and os.path.exists(mask_path)):
            continue

        # Load optical GeoTIFF and convert from (C, H, W) to (H, W, C)
        with rasterio.open(opt_path) as src:
            opt_data = src.read()
            opt_data = np.moveaxis(opt_data, 0, -1)

        # Load single-channel flood mask
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)

        # Load preprocessed weather time-series for this region
        ts_data = np.load(ts_path)

        # Create 256x256 patches from the full scene
        h, w, _ = opt_data.shape
        patch_size = 256
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                img_patch = opt_data[i:i + patch_size, j:j + patch_size, :]
                mask_patch = mask_data[i:i + patch_size, j:j + patch_size]

                # Discard invalid patches that are all zero (satellite border blackouts)
                if np.max(img_patch) > 0:
                    # Normalize optical reflectance to [0, 1]
                    img_patch = img_patch.astype(np.float32)
                    img_patch = img_patch / np.max(img_patch)

                    # Risk score = fraction of the patch currently flooded
                    risk_score = np.mean(mask_patch > 0.0)

                    all_imgs.append(img_patch)
                    all_ts.append(ts_data)
                    all_labels.append(risk_score)

    return np.array(all_imgs), np.array(all_ts), np.array(all_labels).astype(np.float32)

def train_fusion_model():
    """
    Trains the multi-modal fusion model using aligned image patches
    and weather time-series to predict a continuous flood risk score.
    """
    print("\n--- Initiating Multi-Modal Fusion Training ---")

    train_regions = ['lokoja_confluence_2022', 'borno_basin_2022']
    val_regions = ['bayelsa_coast_2022']

    print("Forging Multi-Modal Tensors for Training...")
    train_imgs, train_ts, train_labels = extract_fusion_data(train_regions)

    print("Forging Multi-Modal Tensors for Validation...")
    val_imgs, val_ts, val_labels = extract_fusion_data(val_regions)

    print("Payload Ready:")
    print(f" -> Optical Tensors: {train_imgs.shape}")
    print(f" -> Weather Tensors: {train_ts.shape}")
    print(f" -> Target Risk Scores: {train_labels.shape}")

    # Build tf.data datasets: each element is ((image, time-series), label)
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_imgs, train_ts), train_labels))
    train_dataset = train_dataset.shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(((val_imgs, val_ts), val_labels))
    val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

    # Initialize the fusion model architecture
    model = build_fusion_model(img_shape=(256, 256, 4), ts_shape=(24, 3))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'fusion_model_best.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    print("\nStarting Fusion Training Loop...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=callbacks
    )
    print("--- Multi-Modal Fusion Training Complete ---")


if __name__ == "__main__":
    import sys
    
    # Simple command line routing
    if len(sys.argv) > 1 and sys.argv[1] == 'fusion':
        train_fusion_model()
    else:
        # Default to training the computer vision component
        train_terrain_model()