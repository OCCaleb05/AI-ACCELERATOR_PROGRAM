import os
import requests
import pandas as pd

# Define target directories
WEATHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_weather'))
os.makedirs(WEATHER_DIR, exist_ok=True)

# Approximate central coordinates for our tactical zones (Latitude, Longitude)
REGIONS = {
    'lokoja_confluence_2022': {'lat': 7.80, 'lon': 6.74},
    'borno_basin_2022':       {'lat': 11.83, 'lon': 13.15},
    'bayelsa_coast_2022':     {'lat': 4.75, 'lon': 6.00}
}

# Timeframe (Capturing the monsoon build-up through the dry season)
START_DATE = '2022-08-01'
END_DATE = '2022-12-31'

def fetch_weather_data(region_name, lat, lon):
    print(f"\nContacting Open-Meteo API for {region_name}...")
    
    # Open-Meteo Archive API URL
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Payload requesting daily precipitation and temperature
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"],
        "timezone": "Africa/Lagos"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract the daily arrays
        daily_data = data.get("daily", {})
        
        # Build a structured Pandas DataFrame
        df = pd.DataFrame({
            "date": daily_data.get("time", []),
            "precipitation_mm": daily_data.get("precipitation_sum", []),
            "temp_max_c": daily_data.get("temperature_2m_max", []),
            "temp_min_c": daily_data.get("temperature_2m_min", [])
        })
        
        # Fill any missing API values with 0
        df.fillna(0, inplace=True)
        
        # Save to CSV
        output_path = os.path.join(WEATHER_DIR, f"{region_name}_weather.csv")
        df.to_csv(output_path, index=False)
        print(f"Success! Historical weather data saved to {output_path}")
        
    else:
        print(f"Failed to retrieve data for {region_name}. Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("--- INITIATING WEATHER DATA INGESTION ---")
    for region, coords in REGIONS.items():
        fetch_weather_data(region, coords['lat'], coords['lon'])
    print("\n--- WEATHER INGESTION COMPLETE ---")