import requests
import os
from datetime import datetime

def download_bgp_updates(year, month, start_day, end_day, start_hour, end_hour, start_minute, end_minute):
    month_str = f'{int(month):02d}'
    
    base_url = f'http://archive.routeviews.org/bgpdata/{year}.{month_str}/UPDATES/'
    
    download_dir = f'../bgp_routeviews_updates/{year}/{month_str}'
    os.makedirs(download_dir, exist_ok=True)
    
    for day in range(start_day, end_day + 1):
        day_str = f'{day:02d}'
        
        for hour in range(start_hour, end_hour + 1):
            hour_str = f'{hour:02d}'
            
            for minute in range(start_minute, end_minute + 1, 15):
                minute_str = f'{minute:02d}'
                
                filename = f'updates.{year}{month_str}{day_str}.{hour_str}{minute_str}.bz2'
                url = f'{base_url}{filename}'
                
                file_path = os.path.join(download_dir, filename)
                
                if os.path.exists(file_path):
                    print(f'Already downloaded: {file_path}')
                    continue
                
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        print(f'Downloaded: {url} to {file_path}')
                    else:
                        print(f'Failed to download: {url} (Status Code: {response.status_code})')
                except requests.exceptions.RequestException as e:
                    print(f'Error downloading {url}: {e}')

if __name__ == "__main__":
    # Example Inputs
    year = input("Enter Year (e.g., 2024): ")
    month = input("Enter Month (1-12): ")
    start_day = int(input("Enter Start Day (1-31): "))
    end_day = int(input("Enter End Day (1-31): "))
    start_hour = int(input("Enter Start Hour (0-23): "))
    end_hour = int(input("Enter End Hour (0-23): "))
    start_minute = int(input("Enter Start Minute (0-59): "))
    end_minute = int(input("Enter End Minute (0-59): "))
    
    download_bgp_updates(year, month, start_day, end_day, start_hour, end_hour, start_minute, end_minute)