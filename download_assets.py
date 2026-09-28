import os
import requests

def download_file(url, filename):
    print(f"Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

if __name__ == "__main__":
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    # ESPCN x4 super resolution model from OpenCV's official repo
    espcn_url = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb"
    espcn_path = os.path.join(assets_dir, "ESPCN_x4.pb")
    if not os.path.exists(espcn_path):
        download_file(espcn_url, espcn_path)
    
    # Download a sample side-scan sonar image (or a proxy if a direct link to SSS isn't easily available)
    # Using an image of a shipwreck from Wikipedia as a proxy for SSS for demo purposes
    sonar_url = "https://upload.wikimedia.org/wikipedia/commons/e/ea/Side_scan_sonar_image_of_the_shipwreck_of_the_RMS_Titanic.jpg"
    sonar_path = os.path.join(assets_dir, "sample_sonar.jpg")
    if not os.path.exists(sonar_path):
        download_file(sonar_url, sonar_path)
    
    print("Assets downloading complete.")
