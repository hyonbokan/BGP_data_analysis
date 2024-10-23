import pybgpstream
import editdistance
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import csv
from collections import defaultdict, Counter
import ipaddress
import logging

def plot_statistics(df_features, target_asn):
    numeric_cols = df_features.select_dtypes(include=['number']).columns

    # Define the color map
    num_colors = len(numeric_cols)
    color_map = plt.get_cmap('tab20', num_colors)

    # Plotting the statistics
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(numeric_cols):
        plt.plot(df_features['Timestamp'], df_features[col], label=col, color=color_map(i))

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Statistics for ASN {target_asn}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def detect_bgp_hijacks(from_time, until_time, target_asn=None, target_prefixes=None):
    """
    Detects BGP hijacks within a specified time range for a given ASN or list of prefixes.

    Parameters:
        from_time (str): Start time in the format 'YYYY-MM-DD HH:MM:SS'
        until_time (str): End time in the format 'YYYY-MM-DD HH:MM:SS'
        target_asn (int or str, optional): The ASN suspected of hijacking or the legitimate ASN.
        target_prefixes (list of str, optional): List of prefixes to monitor for hijacks.

    Returns:
        dict: A dictionary containing detected hijacks.
    """
    # Convert target_asn to string for consistency
    if target_asn is not None:
        target_asn = str(target_asn)

    # Initialize results
    results = {
        "prefix_origin_hijacks": set(),
        "prefix_path_hijacks": set(),
        "subprefix_origin_hijacks": set(),
        "subprefix_path_hijacks": set()
    }

    # Initialize BGPStream
    stream = pybgpstream.BGPStream(
        from_time=from_time,
        until_time=until_time,
        record_type="updates"
    )

    # Build a set of monitored prefixes
    monitored_prefixes = set()
    if target_prefixes:
        monitored_prefixes = set(target_prefixes)

    # Process BGP updates
    for rec in stream.records():
        for elem in rec:
            if elem.type != 'A':
                continue  # Only process announcements

            prefix = elem.fields.get("prefix")
            as_path = elem.fields.get("as-path", "")
            if not prefix or not as_path:
                continue

            # Split AS path into a list of ASNs, removing AS sets and confederations
            path = [asn for asn in as_path.strip().split() if '{' not in asn and '(' not in asn]
            if not path:
                continue

            origin_asn = path[-1]
            path_asns = set(path)

            # Check if the prefix matches monitored prefixes or is a subprefix
            is_monitored_prefix = prefix in monitored_prefixes
            is_subprefix = False
            for monitored_prefix in monitored_prefixes:
                try:
                    monitored_net = ipaddress.ip_network(monitored_prefix)
                    prefix_net = ipaddress.ip_network(prefix)
                    if prefix_net != monitored_net and prefix_net.subnet_of(monitored_net):
                        is_subprefix = True
                        break
                except ValueError:
                    continue  # Invalid prefix, skip

            # Detect origin hijacks
            if target_asn and origin_asn == target_asn:
                if is_monitored_prefix:
                    results["prefix_origin_hijacks"].add(prefix)
                elif is_subprefix:
                    results["subprefix_origin_hijacks"].add(prefix)

            # Detect path hijacks
            elif target_asn and target_asn in path_asns:
                if is_monitored_prefix:
                    results["prefix_path_hijacks"].add(prefix)
                elif is_subprefix:
                    results["subprefix_path_hijacks"].add(prefix)

            # If no target ASN is specified, consider any AS announcing the prefix as a potential hijack
            elif not target_asn:
                if is_monitored_prefix:
                    results["prefix_origin_hijacks"].add(prefix)
                elif is_subprefix:
                    results["subprefix_origin_hijacks"].add(prefix)

    # Convert sets to lists for easier handling
    for key in results:
        results[key] = list(results[key])

    return results

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
    

def build_routes_as(routes, target_asn):
    routes_as = {}
    for prefix in routes:
        for collector in routes[prefix]:
            for peer_asn in routes[prefix][collector]:
                path = routes[prefix][collector][peer_asn]
                if len(path) == 0:
                    continue
                if target_asn in path:
                    if target_asn not in routes_as:
                        routes_as[target_asn] = {}
                    routes_as[target_asn][prefix] = path
    return routes_as

def is_bogon_prefix(prefix):
    # List of bogon prefixes for IPv4
    bogon_ipv4_prefixes = [
        '0.0.0.0/8',
        '10.0.0.0/8',
        '100.64.0.0/10',
        '127.0.0.0/8',
        '169.254.0.0/16',
        '172.16.0.0/12',
        '192.0.0.0/24',
        '192.0.2.0/24',
        '192.168.0.0/16',
        '198.18.0.0/15',
        '198.51.100.0/24',
        '203.0.113.0/24',
        '224.0.0.0/4',
        '240.0.0.0/4'
    ]

    # List of bogon prefixes for IPv6
    bogon_ipv6_prefixes = [
        '::/128',           # Unspecified address
        '::1/128',          # Loopback address
        '::ffff:0:0/96',    # IPv4-mapped addresses
        '64:ff9b::/96',     # IPv4/IPv6 translation
        '100::/64',         # Discard prefix
        '2001:db8::/32',    # Documentation prefix
        'fc00::/7',         # Unique local addresses
        'fe80::/10',        # Link-local addresses
        'ff00::/8',         # Multicast addresses
        # Add more bogon IPv6 prefixes as needed
    ]

    try:
        network = ipaddress.ip_network(prefix, strict=False)
        if network.version == 4:
            for bogon in bogon_ipv4_prefixes:
                bogon_network = ipaddress.ip_network(bogon)
                if network.overlaps(bogon_network):
                    return True
        elif network.version == 6:
            for bogon in bogon_ipv6_prefixes:
                bogon_network = ipaddress.ip_network(bogon)
                if network.overlaps(bogon_network):
                    return True
        else:
            # Unknown IP version, consider as non-bogon
            return False
    except ValueError:
        # Invalid IP address format
        return False
    return False

def summarize_peer_updates(peer_updates):
    if not peer_updates:
        return {
            "Total Updates": 0,
            "Average Updates per Peer": 0,
            "Max Updates from a Single Peer": 0,
            "Min Updates from a Single Peer": 0,
            "Std Dev of Updates": 0
        }
    
    total_updates = sum(peer_updates.values())
    num_peers = len(peer_updates)
    avg_updates = total_updates / num_peers if num_peers else 0
    max_updates = max(peer_updates.values()) if peer_updates else 0
    min_updates = min(peer_updates.values()) if peer_updates else 0
    std_dev_updates = (sum((x - avg_updates) ** 2 for x in peer_updates.values()) / num_peers) ** 0.5 if num_peers else 0
    
    return {
        "Total Updates": total_updates,
        "Average Updates per Peer": avg_updates,
        "Max Updates from a Single Peer": max_updates,
        "Min Updates from a Single Peer": min_updates,
        "Std Dev of Updates": std_dev_updates
    }

def top_n_peer_updates(peer_updates, n=5):
    sorted_peers = sorted(peer_updates.items(), key=lambda item: item[1], reverse=True)
    top_peers = sorted_peers[:n]
    return {f"Top Peer {i+1} ASN": peer for i, (peer, _) in enumerate(top_peers)}

def summarize_prefix_announcements(prefix_announced):
    if not prefix_announced:
        return {
            "Total Prefixes Announced": 0,
            "Average Announcements per Prefix": 0,
            "Max Announcements for a Single Prefix": 0,
            "Min Announcements for a Single Prefix": 0,
            "Std Dev of Announcements": 0
        }
    
    total_announcements = sum(prefix_announced.values())
    num_prefixes = len(prefix_announced)
    avg_announcements = total_announcements / num_prefixes if num_prefixes else 0
    max_announcements = max(prefix_announced.values()) if prefix_announced else 0
    min_announcements = min(prefix_announced.values()) if prefix_announced else 0
    std_dev_announcements = (sum((x - avg_announcements) ** 2 for x in prefix_announced.values()) / num_prefixes) ** 0.5 if num_prefixes else 0
    
    return {
        "Total Prefixes Announced": num_prefixes,
        "Average Announcements per Prefix": avg_announcements,
        "Max Announcements for a Single Prefix": max_announcements,
        "Min Announcements for a Single Prefix": min_announcements,
        "Std Dev of Announcements": std_dev_announcements
    }

def top_n_prefix_announcements(prefix_announced, n=5):
    sorted_prefixes = sorted(prefix_announced.items(), key=lambda item: item[1], reverse=True)
    top_prefixes = sorted_prefixes[:n]
    return {f"Top Prefix {i+1}": prefix for i, (prefix, _) in enumerate(top_prefixes)}

def summarize_unexpected_asns(unexpected_asns):
    counter = Counter(unexpected_asns)
    top_unexpected = counter.most_common(3)  # Top 3 unexpected ASNs
    summary = {f"Unexpected ASN {i+1}": asn for i, (asn, _) in enumerate(top_unexpected)}
    return summary

def extract_features(index, routes, old_routes_as, target_asn, target_prefixes=None,
                    prefix_lengths=[], med_values=[], local_prefs=[], 
                    communities_per_prefix={}, peer_updates={}, anomaly_data={}, temp_counts=None,
                    ):
    
    if temp_counts is None:
        temp_counts = initialize_temp_counts()
        
    features = {
        "Timestamp": None,
        "Autonomous System Number": target_asn,
        "Total Routes": 0,
        "New Routes": temp_counts.get("num_new_routes", 0),
        "Withdrawals": temp_counts.get("num_withdrawals", 0),
        "Origin Changes": temp_counts.get("num_origin_changes", 0),
        "Route Changes": temp_counts.get("num_route_changes", 0),
        "Maximum Path Length": 0,
        "Average Path Length": 0,
        "Maximum Edit Distance": 0,
        "Average Edit Distance": 0,
        "Announcements": temp_counts.get("num_announcements", 0),
        "Unique Prefixes Announced": 0,
        # New features
        "Average MED": 0,
        "Average Local Preference": 0,
        "Total Communities": temp_counts.get("total_communities", 0),
        "Unique Communities": len(temp_counts.get("unique_communities", set())),
        "Community Values": [],
        "Total Updates": 0,
        "Average Updates per Peer": 0,
        "Max Updates from a Single Peer": 0,
        "Min Updates from a Single Peer": 0,
        "Std Dev of Updates": 0,
        "Top Peer 1 ASN": None,
        "Top Peer 2 ASN": None,
        "Top Peer 3 ASN": None,
        "Top Peer 4 ASN": None,
        "Top Peer 5 ASN": None,
        "Total Prefixes Announced": 0,
        "Average Announcements per Prefix": 0,
        "Max Announcements for a Single Prefix": 0,
        "Min Announcements for a Single Prefix": 0,
        "Std Dev of Announcements": 0,
        "Top Prefix 1": None,
        "Top Prefix 2": None,
        "Top Prefix 3": None,
        "Top Prefix 4": None,
        "Top Prefix 5": None,
        "Count of Unexpected ASNs in Paths": 0,
        "Unexpected ASN 1": None,
        "Unexpected ASN 2": None,
        "Unexpected ASN 3": None,
        # Anomaly detection features
        "Target Prefixes Withdrawn": anomaly_data.get("target_prefixes_withdrawn", 0),
        "Target Prefixes Announced": anomaly_data.get("target_prefixes_announced", 0),
        "AS Path Changes": anomaly_data.get("as_path_changes", 0),
        # Policy-related feature
        "AS Path Prepending": temp_counts.get("as_path_prepending", 0),
    }

    routes_as = build_routes_as(routes, target_asn)

    if index >= 0:
        num_routes = len(routes_as.get(target_asn, {}))
        sum_path_length = 0
        sum_edit_distance = 0

        # Initialize counts
        new_routes = 0
        route_changes = 0
        origin_changes = 0

        for prefix in routes_as.get(target_asn, {}).keys():
            path = routes_as[target_asn][prefix]
            if target_asn in old_routes_as and prefix in old_routes_as[target_asn]:
                path_old = old_routes_as[target_asn][prefix]

                if path != path_old:
                    route_changes += 1
                    # Check for AS path changes involving target ASN
                    if target_asn in path or target_asn in path_old:
                        anomaly_data["as_path_changes"] += 1

                        # Detect unexpected ASNs in path to target prefixes
                        if target_prefixes and prefix in target_prefixes:
                            unexpected_asns = set(path) - set(path_old)
                            if unexpected_asns - {target_asn}:
                                anomaly_data["unexpected_asns_in_paths"].update(unexpected_asns)

                if path[-1] != path_old[-1]:
                    origin_changes += 1

                path_length = len(path)
                sum_path_length += path_length
                edist = editdistance.eval(path, path_old)
                sum_edit_distance += edist
                features["Maximum Path Length"] = max(features["Maximum Path Length"], path_length)
                features["Maximum Edit Distance"] = max(features["Maximum Edit Distance"], edist)
            else:
                new_routes += 1  # This is a new route
                path_length = len(path)
                sum_path_length += path_length
                # No edit distance to calculate for new routes
                features["Maximum Path Length"] = max(features["Maximum Path Length"], path_length)

        features["New Routes"] = new_routes
        features["Route Changes"] = route_changes
        features["Origin Changes"] = origin_changes

        num_routes_total = num_routes if num_routes else 1  # Avoid division by zero
        features["Total Routes"] = num_routes
        features["Average Path Length"] = sum_path_length / num_routes_total
        features["Average Edit Distance"] = sum_edit_distance / route_changes if route_changes else 0

        # Calculate average MED and Local Preference
        features["Average MED"] = sum(med_values) / len(med_values) if med_values else 0
        features["Average Local Preference"] = sum(local_prefs) / len(local_prefs) if local_prefs else 0

        # Calculate community metrics
        features["Total Communities"] = temp_counts["total_communities"]
        features["Unique Communities"] = len(temp_counts["unique_communities"])
        
        all_communities = set()
        for communities in communities_per_prefix.values():
            for community in communities:
                # Convert community tuple/list to a string representation
                if isinstance(community, (tuple, list)):
                    community_str = ':'.join(map(str, community))
                else:
                    community_str = str(community)
                all_communities.add(community_str)

        features["Community Values"] = list(all_communities)

        # Calculate prefix length statistics
        if prefix_lengths:
            features["Average Prefix Length"] = sum(prefix_lengths) / len(prefix_lengths)
            features["Max Prefix Length"] = max(prefix_lengths)
            features["Min Prefix Length"] = min(prefix_lengths)

        # Summarize and integrate peer updates
        peer_update_summary = summarize_peer_updates(peer_updates)
        features["Total Updates"] = peer_update_summary["Total Updates"]
        features["Average Updates per Peer"] = peer_update_summary["Average Updates per Peer"]
        features["Max Updates from a Single Peer"] = peer_update_summary["Max Updates from a Single Peer"]
        features["Min Updates from a Single Peer"] = peer_update_summary["Min Updates from a Single Peer"]
        features["Std Dev of Updates"] = peer_update_summary["Std Dev of Updates"]

        # Add Top 5 Peers
        top_peers = top_n_peer_updates(peer_updates, n=5)
        for key in top_peers:
            features[key] = top_peers[key]

        # Summarize and integrate prefix announcements
        prefix_announcement_summary = summarize_prefix_announcements(temp_counts["prefixes_announced"])
        features["Total Prefixes Announced"] = prefix_announcement_summary["Total Prefixes Announced"]
        features["Average Announcements per Prefix"] = prefix_announcement_summary["Average Announcements per Prefix"]
        features["Max Announcements for a Single Prefix"] = prefix_announcement_summary["Max Announcements for a Single Prefix"]
        features["Min Announcements for a Single Prefix"] = prefix_announcement_summary["Min Announcements for a Single Prefix"]
        features["Std Dev of Announcements"] = prefix_announcement_summary["Std Dev of Announcements"]

        # Add Top 5 Prefixes
        top_prefixes = top_n_prefix_announcements(temp_counts["prefixes_announced"], n=5)
        for key in top_prefixes:
            features[key] = top_prefixes[key]

        # Summarize unexpected ASNs in paths
        unexpected_asns = anomaly_data.get("unexpected_asns_in_paths", [])
        unexpected_asn_summary = summarize_unexpected_asns(unexpected_asns)
        features["Count of Unexpected ASNs in Paths"] = len(unexpected_asns)
        features.update(unexpected_asn_summary)

    features["Unique Prefixes Announced"] = len(routes_as.get(target_asn, {}))

    return features, routes_as


def extract_bgp_data(from_time, until_time, target_asn, target_prefixes=None, 
                     collectors=['rrc00', 'route-views2', 'route-views.sydney', 'route-views.wide'],
                     output_file='bgp_features.csv'):

    stream = pybgpstream.BGPStream(
        from_time=from_time,
        until_time=until_time,
        record_type="updates",
        collectors=collectors
    )

    all_features = []
    old_routes_as = {}
    routes = {}
    current_window_start = datetime.strptime(from_time, "%Y-%m-%d %H:%M:%S")
    index = 0

    # Initialize temporary counts and data collections
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
    print(f"Starting BGP data extraction for ASN {target_asn} from {from_time} to {until_time}")

    record_count = 0
    element_count = 0

    for rec in stream.records():
        record_count += 1
        for elem in rec:
            element_count += 1
            update = elem.fields
            elem_time = datetime.utcfromtimestamp(elem.time)

            # If the time exceeds the 5-minute window, process the window and reset
            if elem_time >= current_window_start + timedelta(minutes=5):
                features, old_routes_as = extract_features(
                    index, routes, old_routes_as, target_asn, target_prefixes,
                    prefix_lengths, med_values, local_prefs, 
                    communities_per_prefix, peer_updates, anomaly_data, temp_counts
                )
                features['Timestamp'] = current_window_start.strftime("%Y-%m-%d %H:%M:%S")
                all_features.append(features)

                # Move to the next 5-minute window
                current_window_start += timedelta(minutes=5)
                routes = {}  # Reset the routes for the next window
                index += 1

                # Reset temporary counts and data collections
                temp_counts = initialize_temp_counts()
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

            prefix = update.get("prefix")
            if prefix is None:
                continue

            # Initialize process_update flag
            process_update = False

            # Check if target ASN is in the AS path
            as_path_str = update.get('as-path', "")
            as_path = [asn for asn in as_path_str.split() if '{' not in asn and '(' not in asn]
            if target_asn in as_path:
                process_update = True

            # If target_prefixes are provided, check if the prefix is in target_prefixes
            if target_prefixes:
                if prefix in target_prefixes:
                    process_update = True
                else:
                    # Optionally, check if the prefix is a subprefix of any target_prefix
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
            else:
                # If target_prefixes is None, we don't filter by prefixes
                pass

            # If neither condition is met, skip this update
            if not process_update:
                continue

            # Collect prefix length
            try:
                network = ipaddress.ip_network(prefix, strict=False)
                prefix_length = network.prefixlen
            except ValueError:
                continue  # Skip this prefix if invalid
            prefix_lengths.append(prefix_length)

            # Check for bogon prefixes
            if is_bogon_prefix(prefix):
                temp_counts["bogon_prefixes"] += 1

            peer_asn = elem.peer_asn
            collector = rec.collector

            # Count updates per peer
            peer_updates[peer_asn] += 1

            # Processing Announcements (A) and Withdrawals (W)
            if elem.type == 'A':  # Announcement
                if as_path:
                    temp_counts["prefixes_announced"][prefix] = temp_counts["prefixes_announced"].get(prefix, 0) + 1
                    temp_counts["num_announcements"] += 1
                    # Initialize routes
                    if prefix not in routes:
                        routes[prefix] = {}
                    if collector not in routes[prefix]:
                        routes[prefix][collector] = {}

                    routes[prefix][collector][peer_asn] = as_path

                    # Collect MED and Local Preference
                    med = update.get('med')
                    if med is not None:
                        try:
                            med_values.append(int(med))
                        except ValueError:
                            pass  # Handle non-integer MED values
                    local_pref = update.get('local-pref')
                    if local_pref is not None:
                        try:
                            local_prefs.append(int(local_pref))
                        except ValueError:
                            pass  # Handle non-integer Local Preference values

                    # Collect Communities
                    communities = update.get('communities', [])
                    if communities:
                        temp_counts["total_communities"] += len(communities)
                        temp_counts["unique_communities"].update(tuple(c) for c in communities)
                        communities_per_prefix[prefix] = communities

                    # Check for AS Path Prepending
                    if len(set(as_path)) < len(as_path):
                        temp_counts["as_path_prepending"] += 1

                    # Anomaly detection for announcements
                    if isinstance(target_prefixes, (list, set)) and prefix in target_prefixes:
                        anomaly_data["target_prefixes_announced"] += 1
                        # Check for unexpected ASNs in path to target prefixes
                        if target_asn not in as_path:
                            anomaly_data["unexpected_asns_in_paths"].update(set(as_path))
            elif elem.type == 'W':  # Withdrawal
                if prefix in routes and collector in routes[prefix]:
                    if peer_asn in routes[prefix][collector]:
                        routes[prefix][collector].pop(peer_asn, None)
                        temp_counts["prefixes_withdrawn"][prefix] = temp_counts["prefixes_withdrawn"].get(prefix, 0) + 1
                        temp_counts["num_withdrawals"] += 1
                        
                        # Anomaly detection for withdrawals
                        if target_prefixes and prefix in target_prefixes:
                            anomaly_data["target_prefixes_withdrawn"] += 1

    print(f"Total records processed: {record_count}")
    print(f"Total elements processed: {element_count}")

    # Process the final 5-minute window
    features, old_routes_as = extract_features(
        index, routes, old_routes_as, target_asn, target_prefixes,
        prefix_lengths, med_values, local_prefs, 
        communities_per_prefix, peer_updates, anomaly_data, temp_counts
    )
    features['Timestamp'] = current_window_start.strftime("%Y-%m-%d %H:%M:%S")
    all_features.append(features)

    # Convert the collected features into a DataFrame and save it
    df_features = pd.json_normalize(all_features, sep='_').fillna(0)
    df_features.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    return df_features

def detect_anomalies(df, numeric_cols, threshold_multiplier=2):
    diff = df[numeric_cols].diff().abs()
    
    mean_values = diff.mean()
    std_values = diff.std()
    thresholds = mean_values + threshold_multiplier * std_values
    anomalies = (diff > thresholds).any(axis=1)
    
    # Initialize the anomaly_status column with "no anomalies detected"
    df['anomaly_status'] = "no anomalies detected"
    
    # Label anomalies and add reasons
    for idx in df[anomalies].index:
        timestamp = df.loc[idx, 'timestamp']        
        reasons = []
        for col in numeric_cols:
            if diff.loc[idx, col] > thresholds[col] and df.loc[idx, col] != 0:
                reasons.append(f"{col}={df.loc[idx, col]}")
        if reasons:
            df.at[idx, 'anomaly_status'] = f"anomaly detected at {timestamp} due to high value of {', '.join(reasons)}"
    
    return df

def detect_anomalies_new(df, numeric_cols, threshold_multiplier=2):
    diff = df[numeric_cols].diff().abs()
    
    mean_values = diff.mean()
    std_values = diff.std()
    thresholds = mean_values + threshold_multiplier * std_values
    anomalies = (diff > thresholds).any(axis=1)

    df['anomaly_status'] = "No anomalies detected"
    
    for idx in df[anomalies].index:
        timestamp = df.loc[idx, 'timestamp']
        reasons = []
        for col in numeric_cols:
            if diff.loc[idx, col] > thresholds[col] and df.loc[idx, col] != 0:
                value = df.loc[idx, col]
                rounded_mean = round(mean_values[col], 2)
                rounded_std = round(std_values[col], 2)
                reasons.append(f"{col}: observed value={value}, expected mean={rounded_mean}, standard deviation={rounded_std}")
        if reasons:
            df.at[idx, 'anomaly_status'] = f"Anomaly detected at {timestamp} due to the following deviations: {', '.join(reasons)}"
    
    return df

def extract_bgp_data_and_build_weighted_graph(target_asn, from_time, until_time, collectors=['rrc00'], output_file=None):
    stream = pybgpstream.BGPStream(
        from_time=from_time,
        until_time=until_time,
        record_type="updates",
        collectors=collectors
    )

    routes = {}
    all_data = []
    current_window_start = datetime.strptime(from_time, "%Y-%m-%d %H:%M:%S")
    index = 0

    record_count = 0
    element_count = 0

    print(f"Starting data collection from {from_time} to {until_time} for collectors {collectors}")

    for rec in stream.records():
        record_count += 1
        for elem in rec:
            element_count += 1
            update = elem.fields
            elem_time = datetime.utcfromtimestamp(elem.time)

            # If time exceeds the 5-minute window, process the window and reset
            if elem_time >= current_window_start + timedelta(minutes=5):
                current_window_start += timedelta(minutes=5)
                # Reset routes here would lose paths, so we remove the reset!
                index += 1

            prefix = update.get("prefix")
            if prefix is None:
                continue

            peer_asn = update.get("peer_asn", "unknown")
            collector = rec.collector

            if prefix not in routes:
                routes[prefix] = {}
            if collector not in routes[prefix]:
                routes[prefix][collector] = {}

            # Process announcements and withdrawals
            if elem.type == 'A':
                as_path = update.get('as-path')
                if as_path:
                    path = as_path.split()
                    if path and path[-1] == target_asn:
                        if peer_asn not in routes[prefix][collector]:
                            routes[prefix][collector][peer_asn] = path  # Store the path correctly here
                            all_data.append({
                                'timestamp': elem_time,
                                'prefix': prefix,
                                'collector': collector,
                                'peer_asn': peer_asn,
                                'as_path': ' '.join(path)
                            })
                else:
                    print(f"No as-path found for prefix {prefix}")
            elif elem.type == 'W':  # Withdrawal
                if prefix in routes and collector in routes[prefix]:
                    if peer_asn in routes[prefix][collector]:
                        if routes[prefix][collector][peer_asn][-1] == target_asn:
                            routes[prefix][collector].pop(peer_asn, None)

    print(f"Total records processed: {record_count}")
    print(f"Total elements processed: {element_count}")

    # Optionally save data to CSV
    if output_file:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'prefix', 'collector', 'peer_asn', 'as_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in all_data:
                writer.writerow(row)
        
        print(f"Data saved to {output_file}")

    # Build the weighted graph from the extracted routes
    G = buildWeightedGraph(routes)

    return G

def buildWeightedGraph(routes):
    graph = nx.Graph()
    
    for prefix in routes.keys():        
        nbIp = 1  # You can update this to compute the actual number of IPs if required
        origins = []
        vertices = []
        edges = []
        
        for collector in routes[prefix].keys():
            for peer in routes[prefix][collector].keys():
                path = routes[prefix][collector][peer]
                path_vertices = []
                path_edges = []
                path_origin = ""

                if path is not None:
                    if "{" in path or "}" in path:
                        print("    Skipping path with {}")
                        pass
                    else:
                        path_vertices = path
                        path_origin = path_vertices[-1]
                        for i in range(len(path_vertices) - 1):
                            path_edges.append([path_vertices[i], path_vertices[i + 1]])
                        if path_origin not in origins:
                            origins.append(path_origin)

                        for vertex in path_vertices:
                            if vertex not in vertices:
                                vertices.append(vertex)

                        for edge in path_edges:
                            if edge not in edges:
                                edges.append(edge)
        
        for vertex in vertices:
            if not graph.has_node(vertex):
                graph.add_node(vertex, nbIp=0)

        for (a, b) in edges:
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, nbIp=0)
            graph[a][b]["nbIp"] += nbIp

        for origin in origins:
            graph.nodes[origin]["nbIp"] += nbIp

    print(f"Graph construction complete with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    return graph

def plot_weighted_graph(G, title="BGP Weighted Route Graph"):
    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42)
    
    edge_weights = np.array([G[u][v]['nbIp'] for u, v in G.edges()])
    max_weight = max(edge_weights) if len(edge_weights) > 0 else 1
    min_weight = min(edge_weights) if len(edge_weights) > 0 else 0
    normalized_weights = [(0.5 + (weight - min_weight) / (max_weight - min_weight)) * 5 
                          for weight in edge_weights]
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", edgecolors='black', linewidths=0.5, alpha=0.9)
    
    edges = nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), width=normalized_weights, edge_color=edge_weights, edge_cmap=plt.cm.Blues, alpha=0.7
    )
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])  # This line is important to properly map the colors
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Edge Weight (nbIp)', rotation=270, labelpad=15)
    
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()