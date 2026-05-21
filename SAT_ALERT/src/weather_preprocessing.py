import os
import pandas as pd
import numpy as np

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEATHER_DIR = os.path.join(BASE_DIR, 'data', 'raw_weather')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed_weather')

os.makedirs(OUT_DIR, exist_ok=True)

# Map our regions to the exact dates the Sentinel-2 satellite took our pictures
# The LSTM needs the 24 days leading up to these specific events
SATELLITE_PASS_DATES = {
    'borno_basin_2022': '2022-10-03',
    'lokoja_confluence_2022': '2022-11-03',
    'bayelsa_coast_2022': '2022-11-08'
}

def forge_weather_tensors():
    print("--- FORGING TIME-SERIES WEATHER TENSORS ---")
    
    for region, target_date in SATELLITE_PASS_DATES.items():
        csv_path = os.path.join(WEATHER_DIR, f"{region}_weather.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: Could not find weather data for {region}. Skipping.")
            continue
            
        # 1. Load the data
        df = pd.read_csv(csv_path)
        
        # 2. Find the target date
        if target_date not in df['date'].values:
            print(f"Warning: Target date {target_date} not found in {region} data!")
            continue
            
        target_idx = df[df['date'] == target_date].index[0]
        
        # 3. Slice the 24 days strictly BEFORE the satellite pass
        start_idx = target_idx - 24
        if start_idx < 0:
            print(f"Error: Not enough historical data before {target_date} for {region}!")
            continue
            
        window_df = df.iloc[start_idx:target_idx]
        
        # 4. Extract numerical features: [Precipitation, Max Temp, Min Temp]
        features = window_df[['precipitation_mm', 'temp_max_c', 'temp_min_c']].values
        
        # 5. Neural Network Normalization (Min-Max Scaling to [0, 1])
        # LSTMs crash if you feed them raw temperature numbers like 35.0 mixed with 0.1mm rain
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        
        # Add 1e-8 to prevent division by zero if temps are entirely flat
        normalized_features = (features - f_min) / (f_max - f_min + 1e-8)
        
        # 6. Save the AI-Ready Tensor
        out_path = os.path.join(OUT_DIR, f"{region}_ts.npy")
        np.save(out_path, normalized_features)
        
        print(f"Successfully forged {region}:")
        print(f"  -> Extracted 24 days ({window_df['date'].iloc[0]} to {window_df['date'].iloc[-1]})")
        print(f"  -> Tensor Shape: {normalized_features.shape}")

    print("\n--- WEATHER FORGE COMPLETE ---")

if __name__ == "__main__":
    forge_weather_tensors()