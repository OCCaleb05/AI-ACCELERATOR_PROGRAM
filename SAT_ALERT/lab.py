import os
import numpy as np
import rasterio
import tensorflow as tf
from src.models import build_terrain_unet, build_fusion_model, custom_iou, bce_dice_loss
from src.data_pipeline import process_and_split_dataset, build_tf_dataset

# Define directories for saving our deep learning models
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw_geotiff')
WEATHER_DIR = os.path.join(BASE_DIR, 'data', 'processed_weather')
TERRAIN_MODEL_PATH = os.path.join(MODEL_DIR, 'terrain_unet_best.keras')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_stage1_model():
    """Load the Stage 1 U-Net if available so Stage 2 can consume real Stage 1 predictions."""
    if not os.path.exists(TERRAIN_MODEL_PATH):
        print("WARNING: Terrain U-Net weights not found. Fusion training will fall back to raw mask patches.")
        return None

    try:
        return tf.keras.models.load_model(
            TERRAIN_MODEL_PATH,
            custom_objects={'custom_iou': custom_iou, 'bce_dice_loss': bce_dice_loss}
        )
    except Exception as exc:
        print(f"WARNING: Failed to load Stage 1 model at {TERRAIN_MODEL_PATH}: {exc}")
        return None


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
    Extracts aligned terrain mask patches, weather time-series tensors,
    and continuous flood risk labels for each region.
    """
    all_masks, all_ts, all_labels = [], [], []

    terrain_model = load_stage1_model()

    for region in regions_list:
        mask_path = os.path.join(RAW_DIR, f"{region}_mask.tif")
        opt_path = os.path.join(RAW_DIR, f"{region}_opt.tif")
        ts_path = os.path.join(WEATHER_DIR, f"{region}_ts.npy")

        # Skip regions missing any required input file
        if not os.path.exists(ts_path):
            continue

        # Load preprocessed weather time-series for this region
        ts_data = np.load(ts_path)

        # Attempt to use the trained Stage 1 U-Net for mask generation
        use_stage1_prediction = terrain_model is not None and os.path.exists(opt_path)
        if use_stage1_prediction:
            with rasterio.open(opt_path) as src:
                opt_data = src.read([1, 2, 3, 4]).transpose(1, 2, 0).astype(np.float32)
                opt_data = np.clip(opt_data / 10000.0, 0.0, 1.0)

            mask_data = None
        elif os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                mask_data = src.read(1)
        else:
            continue

        patch_size = 256
        if use_stage1_prediction:
            h, w, _ = opt_data.shape
        else:
            h, w = mask_data.shape

        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                if use_stage1_prediction:
                    opt_patch = opt_data[i:i + patch_size, j:j + patch_size, :]
                    pred_mask = terrain_model.predict(np.expand_dims(opt_patch, axis=0), verbose=0)[0]
                    mask_patch = (pred_mask > 0.5).astype(np.float32)
                else:
                    mask_patch = mask_data[i:i + patch_size, j:j + patch_size].astype(np.float32)
                    mask_patch = np.expand_dims(mask_patch, axis=-1)

                # Discard invalid patches that are all zero (satellite border blackouts)
                if np.max(mask_patch) > 0:
                    risk_score = np.mean(mask_patch > 0.0)
                    all_masks.append(mask_patch)
                    all_ts.append(ts_data)
                    all_labels.append(risk_score)

    return np.array(all_masks), np.array(all_ts), np.array(all_labels).astype(np.float32)

def train_fusion_model():
    """
    Trains the multi-modal fusion model using aligned terrain mask patches
    and weather time-series to predict a continuous flood risk score.
    """
    print("\n--- Initiating Multi-Modal Fusion Training ---")

    train_regions = ['lokoja_confluence_2022', 'borno_basin_2022']
    val_regions = ['bayelsa_coast_2022']

    print("Forging Multi-Modal Tensors for Training...")
    train_masks, train_ts, train_labels = extract_fusion_data(train_regions)

    print("Forging Multi-Modal Tensors for Validation...")
    val_masks, val_ts, val_labels = extract_fusion_data(val_regions)

    print("Payload Ready:")
    print(f" -> Terrain Mask Tensors: {train_masks.shape}")
    print(f" -> Weather Tensors: {train_ts.shape}")
    print(f" -> Target Risk Scores: {train_labels.shape}")

    # Build tf.data datasets: each element is ((mask, time-series), label)
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_masks, train_ts), train_labels))
    train_dataset = train_dataset.shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(((val_masks, val_ts), val_labels))
    val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

    # Initialize the fusion model architecture
    model = build_fusion_model(mask_shape=(256, 256, 1), ts_shape=(24, 3))

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