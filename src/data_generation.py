import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def generate_optical_images(batch_size, height=100, width=100):
    """Generate synthetic optical imagery (RGB)."""
    return np.random.rand(batch_size, height, width, 3).astype(np.float32)

def generate_spectral_images(batch_size, height=100, width=100, bands=4):
    """Generate synthetic spectral imagery (multiband)."""
    return np.random.rand(batch_size, height, width, bands).astype(np.float32)

def generate_time_series(batch_size, seq_len=24, features=3):
    """Generate synthetic time series for signals, soil moisture, pressure."""
    return np.random.rand(batch_size, seq_len, features).astype(np.float32)

def generate_historical_floods(batch_size, height=100, width=100):
    """Generate synthetic historical flood masks (binary)."""
    return np.random.randint(0, 2, (batch_size, height, width, 1)).astype(np.float32)

def generate_terrain_images(batch_size, height=100, width=100):
    """Generate synthetic terrain images for CV analysis."""
    return np.random.rand(batch_size, height, width, 3).astype(np.float32)

def generate_risk_masks(batch_size, height=100, width=100):
    """Generate synthetic risk zone masks for training CV model."""
    return np.random.randint(0, 2, (batch_size, height, width, 1)).astype(np.float32)

def generate_risk_scores(batch_size):
    """Generate synthetic risk scores for fusion model."""
    return np.random.rand(batch_size, 1).astype(np.float32)

def generate_forecast_targets(batch_size, height=100, width=100, future_steps=72):
    """Generate synthetic future inundation maps (simplified to single map)."""
    return np.random.rand(batch_size, height, width, 1).astype(np.float32)

def save_data():
    """Generate and save sample datasets."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Sample data for training
    optical = generate_optical_images(100)
    spectral = generate_spectral_images(100)
    time_series = generate_time_series(100)
    historical = generate_historical_floods(100)
    terrain = generate_terrain_images(100)
    risk_masks = generate_risk_masks(100)
    forecasts = generate_forecast_targets(100)
    risk_scores = generate_risk_scores(100)

    np.savez(os.path.join(DATA_DIR, 'train_data.npz'),
             optical=optical, spectral=spectral, time_series=time_series,
             historical=historical, terrain=terrain, risk_masks=risk_masks, forecasts=forecasts, risk_scores=risk_scores)

    print("Sample training data saved to data/train_data.npz")

if __name__ == "__main__":
    save_data()