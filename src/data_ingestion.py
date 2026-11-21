import os
import requests
import time

# --- COPERNICUS CREDENTIALS ---
USERNAME = '...'
PASSWORD = '...'
if not USERNAME or not PASSWORD:
    raise EnvironmentError(
        'COPERNICUS_USERNAME and COPERNICUS_PASSWORD must be set in the environment.'
    )

# Define target directories
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_geotiff')
os.makedirs(RAW_DIR, exist_ok=True)

# Define our tactical operational zones
REGIONS = {
    'lokoja_confluence_2022': 'POLYGON((6.5 7.5, 7.0 7.5, 7.0 8.0, 6.5 8.0, 6.5 7.5))',
    'borno_basin_2022': 'POLYGON((12.5 11.5, 13.5 11.5, 13.5 12.5, 12.5 12.5, 12.5 11.5))',
    'bayelsa_coast_2022': 'POLYGON((5.5 4.2, 6.5 4.2, 6.5 5.2, 5.5 5.2, 5.5 4.2))'
}

START_DATE = '2022-09-01T00:00:00.000Z'
END_DATE = '2022-12-31T23:59:59.999Z'  # Pushed into the dry season

def get_auth_token():
    print("Authenticating with Copernicus Data Space Ecosystem...")
    token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    data = {
        'client_id': 'cdse-public',
        'username': USERNAME,
        'password': PASSWORD,
        'grant_type': 'password'
    }
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        print("Authentication successful.")
        return response.json()['access_token']
    else:
        raise Exception(f"Authentication failed: {response.text}")

def search_sentinel2(wkt_polygon, access_token):
    search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    query = (
        f"?$filter=Collection/Name eq 'SENTINEL-2' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'S2MSI2A') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value lt 95.0) "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}') "
        f"and ContentDate/Start ge {START_DATE} and ContentDate/Start le {END_DATE}"
        f"&$top=1"
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Give the search phase 5 chances to survive a network flicker
    for attempt in range(5):
        try:
            response = requests.get(search_url + query, headers=headers, timeout=15)
            if response.status_code == 200:
                results = response.json().get('value', [])
                if results:
                    return results[0]['Id'], results[0]['Name']
                return None, None
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            print(f"    [Search network flicker] Retrying search... ({attempt+1}/5)")
            time.sleep(2)
            
    print("Search failed after multiple attempts.")
    return None, None

def download_product(product_id, product_name, region_name, access_token):
    """A robust, resumable downloader that survives network drops."""
    download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    file_path = os.path.join(RAW_DIR, f"{region_name}_raw.zip")
    
    max_retries = 10
    for attempt in range(max_retries):
        headers = {"Authorization": f"Bearer {access_token}"}
        mode = 'wb'
        
        # Check if file exists and how big it is to RESUME download
        if os.path.exists(file_path):
            downloaded_bytes = os.path.getsize(file_path)
            if downloaded_bytes > 0:
                headers['Range'] = f'bytes={downloaded_bytes}-'
                mode = 'ab' # Append mode
                print(f"Resuming download from {downloaded_bytes / (1024*1024):.1f} MB...")
        else:
            print(f"Initiating new download for {region_name}... (~1GB)")

        try:
            # 30-second timeout on the socket so it doesn't freeze indefinitely
            response = requests.get(download_url, headers=headers, stream=True, timeout=30)
            
            # 200 = OK (New), 206 = Partial Content (Resuming)
            if response.status_code in [200, 206]:
                with open(file_path, mode) as f:
                    # Download in larger 64KB chunks
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                print(f"\nSuccessfully finished downloading {region_name}!")
                return # Break out of the retry loop
                
            elif response.status_code == 416:
                print(f"\nFile {region_name} is already completely downloaded!")
                return
            else:
                print(f"\nDownload server error: Status {response.status_code}")
                break

        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print(f"\n[Network drop detected] Attempt {attempt+1}/{max_retries}. Reconnecting in 5 seconds...")
            time.sleep(5)
            
    print(f"\nDownload failed for {region_name} after maximum retries. Please check connection.")

if __name__ == "__main__":
    try:
        token = get_auth_token()
        for region_name, wkt in REGIONS.items():
            print(f"\nScanning for high-fidelity data over {region_name}...")
            prod_id, prod_name = search_sentinel2(wkt, token)
            
            if prod_id:
                print(f"Found suitable imagery: {prod_name}")
                download_product(prod_id, prod_name, region_name, token)
            else:
                print(f"No suitable cloud-free imagery found for {region_name} in the specified timeframe.")
    except Exception as e:
        print(str(e))