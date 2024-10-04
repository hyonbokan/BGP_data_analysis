import pybgpstream
import editdistance
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import csv
from collections import defaultdict
import ipaddress


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
    
    
def build_routes_as(routes):
    routes_as = {}
    for prefix in routes:
        for collector in routes[prefix]:
            for peer_asn in routes[prefix][collector]:
                path = routes[prefix][collector][peer_asn]
                if len(path) == 0:
                    continue
                asn = path[-1]
                if asn not in routes_as:
                    routes_as[asn] = {}
                routes_as[asn][prefix] = path
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


def extract_features(index, routes, old_routes_as, target_asn, prefix_lengths, med_values, local_prefs, communities_per_prefix, peer_updates):
    features = {
        "Timestamp": None,
        "Autonomous System Number": target_asn,
        "Total Routes": 0,
        "New Routes": 0,
        "Withdrawals": 0,
        "Origin Changes": 0,
        "Route Changes": 0,
        "Maximum Path Length": 0,
        "Average Path Length": 0,
        "Maximum Edit Distance": 0,
        "Average Edit Distance": 0,
        "Announcements": 0,
        "Unique Prefixes Announced": 0,
        "Graph Average Degree": 0,
        "Graph Betweenness Centrality": 0,
        "Graph Closeness Centrality": 0,
        "Graph Eigenvector Centrality": 0,
        # New features
        "Average MED": 0,
        "Average Local Preference": 0,
        "Total Communities": 0,
        "Unique Communities": 0,
        "Updates per Peer": peer_updates,
        "Number of Unique Peers": len(peer_updates),
        "Prefixes Announced": [],
        "Prefixes Withdrawn": [],
        "Prefixes with AS Path Prepending": 0,
        "Bogon Prefixes Detected": 0,
        "Average Prefix Length": 0,
        "Max Prefix Length": 0,
        "Min Prefix Length": 0
    }

    # Set counts for withdrawals, announcements, prefixes announced/withdrawn, bogon prefixes
    features["Withdrawals"] = temp_counts["num_withdrawals"]
    features["Announcements"] = temp_counts["num_announcements"]
    features["Prefixes Announced"] = temp_counts["prefixes_announced"]
    features["Prefixes Withdrawn"] = temp_counts["prefixes_withdrawn"]
    features["Bogon Prefixes Detected"] = temp_counts["bogon_prefixes"]
    features["Prefixes with AS Path Prepending"] = temp_counts["as_path_prepending"]

    routes_as = build_routes_as(routes)

    if index >= 0 and target_asn in routes_as:
        num_routes = len(routes_as[target_asn])
        sum_path_length = 0
        sum_edit_distance = 0

        # Build the graph for the current ASN
        G = nx.Graph()
        for prefix, path in routes_as[target_asn].items():
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i + 1])

        # Calculate graph features if the graph has nodes
        if G.number_of_nodes() > 0:
            features["Graph Average Degree"] = sum(dict(G.degree).values()) / G.number_of_nodes()
            features["Graph Betweenness Centrality"] = sum(nx.betweenness_centrality(G).values()) / G.number_of_nodes()
            features["Graph Closeness Centrality"] = sum(nx.closeness_centrality(G).values()) / G.number_of_nodes()
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                features["Graph Eigenvector Centrality"] = sum(eigenvector_centrality.values()) / G.number_of_nodes()
            except nx.PowerIterationFailedConvergence:
                features["Graph Eigenvector Centrality"] = 0

        # Initialize counts
        new_routes = 0
        route_changes = 0
        origin_changes = 0

        for prefix in routes_as[target_asn].keys():
            path = routes_as[target_asn][prefix]
            if target_asn in old_routes_as and prefix in old_routes_as[target_asn]:
                path_old = old_routes_as[target_asn][prefix]

                if path != path_old:
                    route_changes += 1
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

        # Prefix length statistics
        if prefix_lengths:
            features["Average Prefix Length"] = sum(prefix_lengths) / len(prefix_lengths)
            features["Max Prefix Length"] = max(prefix_lengths)
            features["Min Prefix Length"] = min(prefix_lengths)

    features["Unique Prefixes Announced"] = len(routes_as.get(target_asn, {}))

    return features, routes_as

def extract_bgp_data(target_asn, from_time, until_time, collectors=['rrc00'], output_file='bgp_features.csv'):
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

    global temp_counts  # Declare temp_counts as global to access in extract_features

    # Initialize temporary counts for announcements and withdrawals
    temp_counts = {
        "num_announcements": 0,
        "num_withdrawals": 0,
        "prefixes_announced": [],
        "prefixes_withdrawn": [],
        "as_path_prepending": 0,
        "bogon_prefixes": 0,
        "total_communities": 0,
        "unique_communities": set()
    }

    # Additional data collections
    prefix_lengths = []
    med_values = []
    local_prefs = []
    communities_per_prefix = {}
    peer_updates = defaultdict(int)

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
                    index, routes, old_routes_as, target_asn,
                    prefix_lengths, med_values, local_prefs, communities_per_prefix, peer_updates
                )
                features['Timestamp'] = current_window_start
                all_features.append(features)

                # Move to the next 5-minute window
                current_window_start += timedelta(minutes=5)
                routes = {}  # Reset the routes for the next window
                index += 1

                # Reset temporary counts and data collections
                temp_counts = {
                    "num_announcements": 0,
                    "num_withdrawals": 0,
                    "prefixes_announced": [],
                    "prefixes_withdrawn": [],
                    "as_path_prepending": 0,
                    "bogon_prefixes": 0,
                    "total_communities": 0,
                    "unique_communities": set()
                }
                prefix_lengths = []
                med_values = []
                local_prefs = []
                communities_per_prefix = {}
                peer_updates = defaultdict(int)

            prefix = update.get("prefix")
            if prefix is None:
                continue

            # Collect prefix length
            try:
                network = ipaddress.ip_network(prefix, strict=False)
                prefix_length = network.prefixlen
            except ValueError:
                prefix_length = None  # Handle invalid prefix format
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
                path = update.get('as-path', "").split()
                if path and path[-1] == target_asn:
                    temp_counts["num_announcements"] += 1
                    temp_counts["prefixes_announced"].append(prefix)

                    # Initialize routes after checking for new routes
                    if prefix not in routes:
                        routes[prefix] = {}
                    if collector not in routes[prefix]:
                        routes[prefix][collector] = {}

                    routes[prefix][collector][peer_asn] = path

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
                        temp_counts["unique_communities"].update(communities)
                        communities_per_prefix[prefix] = communities

                    # Check for AS Path Prepending
                    # Clean the AS path to handle AS sets and confederations
                    clean_path = []
                    for asn in path:
                        if '{' in asn or '(' in asn:
                            continue  # Skip AS sets and confederations
                        clean_path.append(asn)
                    if len(set(clean_path)) < len(clean_path):
                        temp_counts["as_path_prepending"] += 1

            elif elem.type == 'W':  # Withdrawal
                if prefix in routes and collector in routes[prefix]:
                    if peer_asn in routes[prefix][collector]:
                        if routes[prefix][collector][peer_asn][-1] == target_asn:
                            routes[prefix][collector].pop(peer_asn, None)
                            temp_counts["num_withdrawals"] += 1
                            temp_counts["prefixes_withdrawn"].append(prefix)

    print(f"Total records processed: {record_count}")
    print(f"Total elements processed: {element_count}")

    # Process the final 5-minute window
    features, old_routes_as = extract_features(
        index, routes, old_routes_as, target_asn,
        prefix_lengths, med_values, local_prefs, communities_per_prefix, peer_updates
    )
    features['Timestamp'] = current_window_start
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

    # Initialize the anomaly_status column with "No anomalies detected"
    df['anomaly_status'] = "No anomalies detected"
    
    # Label anomalies and add detailed reasons with timestamp
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