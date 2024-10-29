import pybgpstream
import time
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
from collections import defaultdict
import ipaddress
import logging
from bgp_data_generation import extract_features

def initialize_temp_counts():
    return {
        "num_announcements": 0,
        "num_withdrawals": 0,
        "num_new_routes": 0,
        "num_origin_changes": 0,
        "num_route_changes": 0,
        "prefixes_announced": {},
        "prefixes_withdrawn": {},
        "as_path_prepending": 0,
        "bogon_prefixes": 0,
        "total_communities": 0,
        "unique_communities": set()
    }

def convert_lists_to_tuples(df):
    for col in df.columns:
        # Check if any element in the column is a list
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

    
def run_real_time_bgpstream(asn, collection_period, return_dict, target_prefixes=None):
    all_features = []
    stream = pybgpstream.BGPStream(
        project="ris-live",
        record_type="updates",
    )
    
    start_time = time.time()
    current_window_start = datetime.utcnow().replace(second=0, microsecond=0)
    index = 0
    old_routes_as = {}
    routes = {}
    
    # Initialize temp_counts with all required keys
    temp_counts = initialize_temp_counts()
    temp_counts['as_path_prepending'] = 0
    prefix_lengths = []
    med_values = []
    local_prefs = []
    communities_per_prefix = {}
    peer_updates = defaultdict(int)
    anomaly_data = {
        "target_prefixes_withdrawn": 0,
        "target_prefixes_announced": 0,
        "as_path_changes": 0,
        "unexpected_asns_in_paths": set()
    }
    
    try:
        for rec in stream.records():
            current_time = datetime.utcnow()

            # Check if the collection period has ended
            if time.time() - start_time >= collection_period.total_seconds():
                print("Collection period ended. Processing data...")
                break

            try:
                for elem in rec:
                    as_path_str = elem.fields.get('as-path', '')
                    as_path = [asn_str for asn_str in as_path_str.strip().split() if '{' not in asn_str and '(' not in asn_str]
                    prefix = elem.fields.get('prefix')
                    if not prefix:
                        continue

                    # Initialize process_update flag
                    process_update = False

                    # Check if target ASN is in the AS path
                    if asn in as_path:
                        process_update = True

                    # If target_prefixes are provided, check if the prefix is in target_prefixes
                    if target_prefixes:
                        for tgt_prefix in target_prefixes:
                            try:
                                tgt_net = ipaddress.ip_network(tgt_prefix)
                                prefix_net = ipaddress.ip_network(prefix)
                                
                                # Only compare if both prefixes are of the same IP version
                                if tgt_net.version == prefix_net.version:
                                    if prefix_net.subnet_of(tgt_net):
                                        process_update = True
                                        break
                            except ValueError:
                                logging.warning(f"Invalid prefix encountered: {tgt_prefix} or {prefix}")
                                continue  # Invalid prefix, skip

                    # If neither condition is met, skip this update
                    if not process_update:
                        continue

                    collector = rec.collector
                    peer_asn = elem.peer_asn

                    if prefix not in routes:
                        routes[prefix] = {}
                    if collector not in routes[prefix]:
                        routes[prefix][collector] = {}

                    if elem.type == 'A':  # Announcement
                        path = as_path
                        # Increment announcement counts
                        temp_counts["prefixes_announced"][prefix] = temp_counts["prefixes_announced"].get(prefix, 0) + 1
                        temp_counts["num_announcements"] += 1
                        peer_updates[peer_asn] += 1

                        # Check if it's a new route
                        if peer_asn not in routes[prefix][collector]:
                            temp_counts["num_new_routes"] += 1

                        # Compare with old routes for changes
                        if asn in old_routes_as and prefix in old_routes_as.get(asn, {}):
                            old_path = old_routes_as[asn][prefix]
                            if path != old_path:
                                temp_counts["num_route_changes"] += 1
                            if path[-1] != old_path[-1]:
                                temp_counts["num_origin_changes"] += 1

                        routes[prefix][collector][peer_asn] = path
                        
                        # Anomaly detection for announcements
                        if isinstance(target_prefixes, (list, set)) and prefix in target_prefixes:
                            anomaly_data["target_prefixes_announced"] += 1
                            # Check for unexpected ASNs in path to target prefixes
                            if asn not in path:
                                anomaly_data["unexpected_asns_in_paths"].update(set(path))
                            
                        # Collect MED and Local Preference
                        med = elem.fields.get('med')
                        if med is not None:
                            try:
                                med_values.append(int(med))
                            except ValueError:
                                pass  # Handle non-integer MED values
                        local_pref = elem.fields.get('local-pref')
                        if local_pref is not None:
                            try:
                                local_prefs.append(int(local_pref))
                            except ValueError:
                                pass  # Handle non-integer Local Preference values

                        # Collect Communities
                        communities = elem.fields.get('communities', [])
                        if communities:
                            temp_counts["total_communities"] += len(communities)
                            temp_counts["unique_communities"].update(tuple(c) for c in communities)
                            communities_per_prefix[prefix] = communities

                        # Check for AS Path Prepending
                        if len(set(path)) < len(path):
                            temp_counts["as_path_prepending"] += 1

                    elif elem.type == 'W':  # Withdrawal
                        if prefix in routes and collector in routes[prefix]:
                            if peer_asn in routes[prefix][collector]:
                                routes[prefix][collector].pop(peer_asn, None)
                                temp_counts["prefixes_withdrawn"][prefix] = temp_counts["prefixes_withdrawn"].get(prefix, 0) + 1
                                temp_counts["num_withdrawals"] += 1
                                peer_updates[peer_asn] += 1
                                
                                # Anomaly detection for withdrawals
                                if target_prefixes and prefix in target_prefixes:
                                    anomaly_data["target_prefixes_withdrawn"] += 1

            except KeyError as ke:
                print(f"KeyError processing record: {ke}. Continuing with the next record.")
                continue

            except ValueError as ve:
                print(f"ValueError processing record: {ve}. Continuing with the next record.")
                continue

            except Exception as e:
                print(f"Unexpected error processing record: {e}. Continuing with the next record.")
                continue

            # Time window check: aggregate and reset every 1 minute
            if current_time >= current_window_start + timedelta(minutes=1):
                print(f"Reached time window: {current_window_start} to {current_time}")

                # Extract features, including paths
                features, old_routes_as = extract_features(
                    index, routes, old_routes_as, asn, target_prefixes,
                    prefix_lengths, med_values, local_prefs, 
                    communities_per_prefix, peer_updates, anomaly_data, temp_counts
                )
                # Check if features is non-empty
                if features:
                    features['Timestamp'] = current_window_start.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Features at index {index}: {features}")
                    
                    all_features.append(features)

                    # Create DataFrame with an explicit index only if features is non-empty
                    try:
                        features_df = pd.DataFrame([features]).dropna(axis=1, how='all')
                        # Update return_dict with the latest features
                        return_dict['features_df'] = features_df
                    except ValueError as ve:
                        print(f"ValueError creating DataFrame from features: {ve}. Skipping this window.")
                else:
                    print(f"No features extracted for this window. Skipping DataFrame creation.")

                current_window_start = current_time.replace(second=0, microsecond=0)
                routes = {}
                index += 1
                # Reset temp_counts for the next window
                temp_counts = initialize_temp_counts()
                temp_counts['as_path_prepending'] = 0
                prefix_lengths = []
                med_values = []
                local_prefs = []
                communities_per_prefix = {}
                peer_updates = defaultdict(int)
                anomaly_data = {
                    "target_prefixes_withdrawn": 0,
                    "target_prefixes_announced": 0,
                    "as_path_changes": 0,
                    "unexpected_asns_in_paths": set()
                }
                
        if routes:
            features, old_routes_as = extract_features(
                index, routes, old_routes_as, asn, target_prefixes,
                prefix_lengths, med_values, local_prefs, 
                communities_per_prefix, peer_updates, anomaly_data, temp_counts
            )
            if features:
                features['Timestamp'] = current_window_start.strftime('%Y-%m-%d %H:%M:%S')
                all_features.append(features)
                try:
                    final_features_df = pd.DataFrame(all_features).dropna(axis=1, how='all')
                    return_dict['features_df'] = final_features_df
                except ValueError as ve:
                    print(f"ValueError creating final DataFrame from all_features: {ve}.")
            else:
                print("No features extracted in the final aggregation window.")
                
    except Exception as e:
        error_message = f"An error occurred during real-time data collection for {asn}: {e}"
        print(error_message)
        return_dict['error'] = error_message
        if all_features:
            features_df = pd.DataFrame(all_features).dropna(axis=1, how='all')
            return_dict['features_df'] = features_df


def collect_real_time_data(asn, target_prefixes=None, collection_period=timedelta(minutes=5)):
    all_collected_data = []  # List to store all collected DataFrames
    features_df = pd.DataFrame()

    print(f"\nCollecting data for ASN {asn} for {collection_period.total_seconds() // 60} minutes...")

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p = multiprocessing.Process(
        target=run_real_time_bgpstream, 
        args=(asn, collection_period, return_dict, target_prefixes)
    )
    p.start()

    start_time = time.time()
    end_time = start_time + collection_period.total_seconds()
    last_features_df = pd.DataFrame()

    no_change_counter = 0
    max_no_change_iterations = 2  # Adjust as needed

    while time.time() < end_time + 5:
        if 'error' in return_dict:
            print(f"Real-time data collection encountered an error: {return_dict['error']}")
            features_df = return_dict.get('features_df', pd.DataFrame())
            break

        features_df = return_dict.get('features_df', pd.DataFrame())

        if not features_df.empty:
            print(f"\nUpdated features_df at {datetime.utcnow()}:\n{features_df.tail(1)}\n")

            # Save the current features_df to the list
            all_collected_data.append(features_df.copy())

            if not last_features_df.empty and features_df.equals(last_features_df):
                no_change_counter += 1
                if no_change_counter >= max_no_change_iterations:
                    print(f"No changes in data for the last {no_change_counter} intervals. Restarting data collection...")
                    # Calculate remaining time for the collection
                    elapsed_time = timedelta(seconds=time.time() - start_time)
                    remaining_time = collection_period - elapsed_time
                    if remaining_time.total_seconds() <= 0:
                        print("No remaining time left for data collection. Exiting.")
                        break

                    # Restart the process with the remaining collection period
                    p.terminate()
                    p.join()
                    
                    print(f"Restarting data collection for the remaining {int(remaining_time.total_seconds())} seconds...")
                    p = multiprocessing.Process(
                        target=run_real_time_bgpstream, 
                        args=(asn, remaining_time, return_dict, target_prefixes)
                    )
                    p.start()
                    no_change_counter = 0  # Reset counter after restarting
            else:
                no_change_counter = 0  # Reset counter if data has changed

            last_features_df = features_df.copy()

        time.sleep(60)  # Sleep for 60 seconds to align with aggregation window
        
    if p.is_alive():
        print("BGPStream collection timed out. Terminating process...")
        p.terminate()
        p.join()

    # Concatenate all collected DataFrames into one final DataFrame
    if all_collected_data:
        final_features_df = pd.concat(all_collected_data, ignore_index=True)
    else:
        final_features_df = features_df

    # Convert list columns to tuples before removing duplicates
    final_features_df = convert_lists_to_tuples(final_features_df)

    # Remove duplicates from the final DataFrame
    final_features_df = final_features_df.drop_duplicates()
    print(final_features_df[['Top Peer 1 ASN', 'Top Peer 2 ASN', 'Top Prefix 1', 'Top Prefix 2']].head())
    final_features_df.to_csv(f"{asn}_real_time.csv")
    
    return final_features_df