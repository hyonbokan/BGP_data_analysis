import requests
import os
from datetime import datetime

def download_bgp_updates(year, month, start_day, end_day, start_hour, end_hour, start_minute, end_minute, collector, data_type):
    try:
        month_int = int(month)
        if not 1 <= month_int <= 12:
            raise ValueError
        month_str = f'{month_int:02d}'
    except ValueError:
        print("Invalid month input. Please enter a value between 1 and 12.")
        return

    if not collector.startswith('rrc') or not collector[3:].isdigit():
        print("Invalid collector name. It should start with 'rrc' followed by digits (e.g., 'rrc00').")
        return

    # Validate data type
    valid_types = ['updates', 'rib']
    if data_type not in valid_types:
        print(f"Invalid data type. Please choose from {valid_types}.")
        return

    # Define the base URL for the specified collector, year, and month
    base_url = f'https://data.ris.ripe.net/{collector}/{year}.{month_str}/'

    # Create a directory to store downloaded files, organized by collector, year, and month
    download_dir = os.path.join(f'../ris_bgp_updates', year, month_str, collector)
    os.makedirs(download_dir, exist_ok=True)

    # List to keep track of missing or failed files
    missing_files = []

    for day in range(start_day, end_day + 1):
        day_str = f'{day:02d}'

        for hour in range(start_hour, end_hour + 1):
            hour_str = f'{hour:02d}'

            for minute in range(start_minute, end_minute + 1, 15):
                minute_str = f'{minute:02d}'

                # Construct the filename and URL
                filename = f'{data_type}.{year}{month_str}{day_str}.{hour_str}{minute_str}.gz'
                url = f'{base_url}{data_type}.{year}{month_str}{day_str}.{hour_str}{minute_str}.gz'

                # Define the path to save the file
                file_path = os.path.join(download_dir, filename)

                # Skip download if file already exists
                if os.path.exists(file_path):
                    print(f'Already downloaded: {file_path}')
                    continue

                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        print(f'Downloaded: {url} to {file_path}')
                    elif response.status_code == 404:
                        print(f'File not found (404): {url}')
                        missing_files.append(url)
                    else:
                        print(f'Failed to download: {url} (Status Code: {response.status_code})')
                        missing_files.append(url)
                except requests.exceptions.RequestException as e:
                    print(f'Error downloading {url}: {e}')
                    missing_files.append(url)

    # Summary of missing or failed files
    if missing_files:
        print("\nSummary of Missing or Failed Files:")
        for mf in missing_files:
            print(mf)
    else:
        print("\nAll files downloaded successfully.")

if __name__ == "__main__":
    try:
        # Example Inputs
        year = input("Enter Year (e.g., 2024): ").strip()
        month = input("Enter Month (1-12): ").strip()
        start_day = int(input("Enter Start Day (1-31): ").strip())
        end_day = int(input("Enter End Day (1-31): ").strip())
        start_hour = int(input("Enter Start Hour (0-23): ").strip())
        end_hour = int(input("Enter End Hour (0-23): ").strip())
        start_minute = int(input("Enter Start Minute (0-59): ").strip())
        end_minute = int(input("Enter End Minute (0-59): ").strip())
        collector = input("Enter Collector Name (e.g., rrc00): ").strip()
        data_type = input("Enter Data Type ('updates' or 'rib'): ").strip().lower()

        # Validate day inputs
        if not (1 <= start_day <= 31) or not (1 <= end_day <= 31):
            raise ValueError("Day must be between 1 and 31.")

        # Validate hour inputs
        if not (0 <= start_hour <= 23) or not (0 <= end_hour <= 23):
            raise ValueError("Hour must be between 0 and 23.")

        # Validate minute inputs
        if not (0 <= start_minute <= 59) or not (0 <= end_minute <= 59):
            raise ValueError("Minute must be between 0 and 59.")

        # Check logical consistency
        if (start_day > end_day) or (start_hour > end_hour) or (start_minute > end_minute):
            raise ValueError("Start values must be less than or equal to end values.")

        download_bgp_updates(year, month, start_day, end_day, start_hour, end_hour, start_minute, end_minute, collector, data_type)
    except ValueError as ve:
        print(f"Input Error: {ve}")
