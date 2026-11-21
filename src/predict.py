import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'terrain_unet_best.keras')
VAL_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_patches', 'val', 'dataset.npz')

# --- CUSTOM METRIC DEPENDENCY ---
# We must provide the custom function so Keras knows how to load the saved model
@tf.keras.utils.register_keras_serializable()
def custom_iou(y_true, y_pred):
    y_true_safe = tf.cast(y_true > 0.5, tf.float32)
    y_pred_safe = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true_safe * y_pred_safe)
    union = tf.reduce_sum(y_true_safe) + tf.reduce_sum(y_pred_safe) - intersection
    return intersection / (union + 1e-7)

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    # Dummy wrapper just for loading the model structure
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def run_inference():
    print("Loading AI Weights...")
    # Load model with custom objects
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'custom_iou': custom_iou, 'bce_dice_loss': bce_dice_loss}
    )

    print("Loading Validation Sector (Bayelsa Coast)...")
    data = np.load(VAL_DATA_PATH)
    images = data['images']
    masks = data['masks']

    # Select a few random 256x256 patches to visualize
    num_samples = 3
    indices = np.random.choice(len(images), num_samples, replace=False)

    plt.figure(figsize=(15, 5 * num_samples))

    for i, idx in enumerate(indices):
        test_img = images[idx]
        true_mask = masks[idx]

        # 1. AI PREDICTION
        # Expand dims because the model expects a batch: (1, 256, 256, 4)
        input_tensor = np.expand_dims(test_img, axis=0)
        predicted_mask = model.predict(input_tensor)[0] # Grab the first result

        # 2. EXTRACT RGB FOR DISPLAY
        # The GeoTIFF has bands (Blue, Green, Red, NIR). We want RGB (Red=2, Green=1, Blue=0)
        # We normalize the pixel values to [0, 1] for matplotlib
        rgb_img = test_img[:, :, [2, 1, 0]] 
        rgb_img = np.clip(rgb_img / np.max(rgb_img), 0, 1) # Brighten the image

        # 3. PLOTTING
        # Plot Raw Satellite
        ax1 = plt.subplot(num_samples, 3, i * 3 + 1)
        ax1.imshow(rgb_img)
        ax1.set_title(f"Raw Optical Sentinel-2 (Patch {idx})")
        ax1.axis('off')

        # Plot Ground Truth
        ax2 = plt.subplot(num_samples, 3, i * 3 + 2)
        ax2.imshow(true_mask[:, :, 0], cmap='Blues')
        ax2.set_title("ESA Ground Truth Mask")
        ax2.axis('off')

        # Plot AI Prediction
        ax3 = plt.subplot(num_samples, 3, i * 3 + 3)
        # Threshold the prediction to show definitive water (confidence > 50%)
        binary_prediction = (predicted_mask[:, :, 0] > 0.5).astype(np.uint8)
        ax3.imshow(binary_prediction, cmap='Reds')
        ax3.set_title("Sat-Alert AI Prediction")
        ax3.axis('off')

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, 'ai_inference_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS! Visual intelligence exported to: {output_path}")

if __name__ == "__main__":
    run_inference()