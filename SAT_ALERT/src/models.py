import tensorflow as tf
from tensorflow.keras import layers, models

# --- CUSTOM METRICS & LOSS FUNCTIONS ---

def custom_iou(y_true, y_pred):
    """
    A bulletproof IoU metric that bypasses Keras's brittle ScatterNd matrix.
    Safely forces all anomalies into strict binary 0.0 or 1.0 tensors.
    """
    y_true_safe = tf.cast(y_true > 0.5, tf.float32)
    y_pred_safe = tf.cast(y_pred > 0.5, tf.float32)
    
    intersection = tf.reduce_sum(y_true_safe * y_pred_safe)
    union = tf.reduce_sum(y_true_safe) + tf.reduce_sum(y_pred_safe) - intersection
    
    return intersection / (union + 1e-7)

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice Loss to heavily penalize missing the minority class (floodwaters).
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_coef = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return 1.0 - dice_coef

def bce_dice_loss(y_true, y_pred):
    """
    Combined Loss Engine: 
    BCE provides pixel-level gradient stability.
    Dice forces the model to map the imbalanced flood shapes.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# --- ARCHITECTURES ---

def build_terrain_unet(input_shape=(256, 256, 4)):
    """
    U-Net Architecture for Computer Vision Terrain Analysis.
    Takes multi-spectral satellite patches (RGB + NIR) and outputs a binary risk mask.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Encoder (Downsampling) ---
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # --- Bottleneck ---
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # --- Decoder (Upsampling) ---
    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    # Output layer: 1 channel with Sigmoid for binary classification (Flood / No Flood)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="Terrain_UNet")
    
    # Compile the model using the upgraded combined loss engine
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy', custom_iou])
    return model


def build_fusion_model(img_shape=(256, 256, 4), ts_shape=(24, 3)):
    """
    Multi-Modal Fusion Architecture.
    Combines a CNN branch for imagery and an LSTM branch for time-series (weather/sensors).
    Outputs a single risk score (0 to 1).
    """
    # 1. Image Branch (CNN)
    img_input = layers.Input(shape=img_shape, name="optical_input")
    x = layers.Conv2D(32, (3, 3), activation='relu')(img_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x) # Flatten spatial dimensions
    
    # 2. Time-Series Branch (LSTM)
    ts_input = layers.Input(shape=ts_shape, name="time_series_input")
    y = layers.LSTM(64, return_sequences=False)(ts_input)
    
    # 3. Fusion Layer
    combined = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z) # Prevent overfitting
    
    # Output Layer (Risk Score)
    risk_output = layers.Dense(1, activation='sigmoid', name="risk_score")(z)
    
    model = models.Model(inputs=[img_input, ts_input], outputs=risk_output, name="MultiModal_Fusion")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model