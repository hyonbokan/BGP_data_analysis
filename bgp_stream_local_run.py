import pybgpstream
from datetime import datetime
import os
import statistics

def process_bgp_updates(directory, target_asn):
    # Dictionary to store update counts by timestamp
    update_counts = []

    # Iterate over each file in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bz2"):  # Process only .bz2 files
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Initialize a new BGPStream instance for each file
                stream = pybgpstream.BGPStream(data_interface="singlefile")

                # Set the file for BGPStream to process
                stream.set_data_interface_option("singlefile", "upd-file", file_path)

                # Temporary counter for updates within a time window
                count = 0
                
                # Process each record and element
                for rec in stream.records():
                    for elem in rec:
                        elem_time = datetime.utcfromtimestamp(elem.time)
                        elem_type = elem.type  # 'A' for announcements, 'W' for withdrawals
                        fields = elem.fields
                        as_path = fields.get("as-path", "").split()

                        # Filter for target ASN in the AS path
                        if target_asn in as_path:
                            if elem_type in {'A', 'W'}:  # Count only announcements and withdrawals
                                count += 1
                
                # Store the count after processing each file
                update_counts.append(count)

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
    directory = "/home/hb/bgp_updates/2024/10"
    target_asn = "3356"  # Filter by specific ASN in AS path
    
    process_bgp_updates(directory, target_asn=target_asn)
