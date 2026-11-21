import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data():
    """Load synthetic training data."""
    data_path = os.path.join(DATA_DIR, 'train_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Run data_generation.py first.")
    data = np.load(data_path)
    return {
        'optical': data['optical'],
        'spectral': data['spectral'],
        'time_series': data['time_series'],
        'historical': data['historical'],
        'terrain': data['terrain'],
        'risk_masks': data['risk_masks'],
        'forecasts': data['forecasts'],
        'risk_scores': data['risk_scores']
    }

def preprocess_image(img):
    """Normalize image to [0,1]."""
    return img / 255.0 if img.max() > 1 else img

def plot_image(img, title='Image'):
    """Plot a single image."""
    plt.figure(figsize=(6, 6))
    if img.shape[-1] == 3:
        plt.imshow(img)
    else:
        plt.imshow(img[:, :, 0], cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_mask(mask, title='Mask'):
    """Plot a binary mask."""
    plt.figure(figsize=(6, 6))
    plt.imshow(mask[:, :, 0], cmap='Reds', alpha=0.7)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_model(model, name):
    """Save model to models/ directory."""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f'{name}.pkl'), 'wb') as f:
        pickle.dump(model, f)

def load_model(name):
    """Load model from models/ directory."""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    with open(os.path.join(model_dir, f'{name}.pkl'), 'rb') as f:
        return pickle.load(f)