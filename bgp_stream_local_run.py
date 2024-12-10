import pybgpstream
from datetime import datetime, timezone
import os
import statistics
import re

def process_bgp_updates(directory, target_asn, from_time_str, until_time_str):
    # Parse the time window strings into datetime objects
    from_time = datetime.strptime(from_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    until_time = datetime.strptime(until_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    # Regular expression pattern to match filenames and extract timestamp
    pattern = r'^updates\.(\d{8})\.(\d{4})\.gz$'

    # List to store update counts
    update_counts = []

    # Iterate over each file in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                # Match the filename with the pattern
                match = re.match(pattern, file)
                if match:
                    date_str = match.group(1)  # YYYYMMDD
                    time_str = match.group(2)  # HHMM

                    # Combine date and time strings
                    file_timestamp_str = date_str + time_str  # 'YYYYMMDDHHMM'

                    # Convert file timestamp to datetime object
                    file_time = datetime.strptime(file_timestamp_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

                    # Check if file time is within the desired time window
                    if file_time < from_time or file_time > until_time:
                        # Skip files outside the time window
                        continue

                    # Proceed to process the file
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")

                    # Initialize a new BGPStream instance for each file
                    stream = pybgpstream.BGPStream(data_interface="singlefile")

                    # Set the file for BGPStream to process
                    stream.set_data_interface_option("singlefile", "upd-file", file_path)

                    count = 0

                    for rec in stream.records():
                        for elem in rec:
                            # Get the element timestamp and set timezone to UTC
                            elem_time = datetime.utcfromtimestamp(elem.time).replace(tzinfo=timezone.utc)

                            # Filter elements outside the time window
                            if elem_time < from_time or elem_time > until_time:
                                continue

                            elem_type = elem.type  # 'A' for announcements, 'W' for withdrawals
                            fields = elem.fields
                            as_path = fields.get("as-path", "").split()

                            # Filter for target ASN in the AS path
                            if target_asn in as_path:
                                if elem_type in {'A', 'W'}:  # Count only announcements and withdrawals
                                    count += 1

                    # Store the count after processing each file
                    update_counts.append(count)
                else:
                    # If filename doesn't match pattern, skip it
                    continue

    # Calculate min, max, and median
    min_updates = min(update_counts) if update_counts else 0
    max_updates = max(update_counts) if update_counts else 0
    median_updates = statistics.median(update_counts) if update_counts else 0

    # Print the summary results
    print(f"Summary for AS{target_asn}:")
    print(f"Minimum updates: {min_updates}")
    print(f"Maximum updates: {max_updates}")
    print(f"Median updates: {median_updates}")

if __name__ == "__main__":
    # Define the directory and target ASN
    directory = "/home/hb/ris_bgp_updates/2024/10/rrc00"
    target_asn = "3356"  # Filter by specific ASN in AS path

    # Define the time window
    from_time_str = "2024-10-28 13:00:00"
    until_time_str = "2024-10-28 13:15:00"

    process_bgp_updates(directory, target_asn=target_asn, from_time_str=from_time_str, until_time_str=until_time_str)