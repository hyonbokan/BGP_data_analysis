import pybgpstream
import editdistance
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import get_named_colors_mapping
import networkx as nx
import matplotlib.cm as cm
import numpy as np

def plot_statistics(df_features, target_asn):
    numeric_cols = df_features.select_dtypes(include=['number']).columns

    # Define the color map
    num_colors = len(numeric_cols)
    color_map = plt.get_cmap('tab20', num_colors)

    # Plotting the statistics
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(numeric_cols):
        plt.plot(df_features['timestamp'], df_features[col], label=col, color=color_map(i))

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

def extract_features(index, routes, old_routes_as, target_asn, temp_counts):
    features = {
        "timestamp": None,
        "asn": target_asn,
        "num_routes": 0,
        "num_new_routes": 0,
        "num_withdrawals": 0,
        "num_origin_changes": 0,
        "num_route_changes": 0,
        "max_path_length": 0,
        "avg_path_length": 0,
        "max_edit_distance": 0,
        "avg_edit_distance": 0,
        "num_announcements": temp_counts["num_announcements"],
        "num_withdrawals": temp_counts["num_withdrawals"],
        "num_unique_prefixes_announced": 0,
        "unique_prefixes_list": []  # New field to store the list of unique prefixes
    }

    routes_as = build_routes_as(routes)

    if index > 0:
        if target_asn in routes_as:
            num_routes = len(routes_as[target_asn])
            sum_path_length = 0
            sum_edit_distance = 0
            unique_prefixes = []

            for prefix in routes_as[target_asn].keys():
                unique_prefixes.append(prefix)

                if target_asn in old_routes_as and prefix in old_routes_as[target_asn]:
                    path = routes_as[target_asn][prefix]
                    path_old = old_routes_as[target_asn][prefix]

                    if path != path_old:
                        features["num_route_changes"] += 1

                    if path[-1] != path_old[-1]:
                        features["num_origin_changes"] += 1

                    path_length = len(path)
                    path_old_length = len(path_old)

                    sum_path_length += path_length
                    if path_length > features["max_path_length"]:
                        features["max_path_length"] = path_length

                    edist = editdistance.eval(path, path_old)
                    sum_edit_distance += edist
                    if edist > features["max_edit_distance"]:
                        features["max_edit_distance"] = edist
                else:
                    features["num_new_routes"] += 1

            features["num_routes"] = num_routes
            features["avg_path_length"] = sum_path_length / num_routes
            features["avg_edit_distance"] = sum_edit_distance / num_routes
            features["unique_prefixes_list"] = unique_prefixes

        if target_asn in old_routes_as:
            for prefix in old_routes_as[target_asn].keys():
                if not (target_asn in routes_as and prefix in routes_as[target_asn]):
                    features["num_withdrawals"] += 1

    # Add the number of unique prefixes announced
    features["num_unique_prefixes_announced"] = len(routes_as.get(target_asn, {}))

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

    # Initialize temporary counts for announcements and withdrawals
    temp_counts = {
        "num_announcements": 0,
        "num_withdrawals": 0
    }

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
                features, old_routes_as = extract_features(index, routes, old_routes_as, target_asn, temp_counts)
                features['timestamp'] = current_window_start
                all_features.append(features)

                # Move to the next 5-minute window
                current_window_start += timedelta(minutes=5)
                routes = {}  # Reset the routes for the next window
                index += 1
                temp_counts = {
                    "num_announcements": 0,
                    "num_withdrawals": 0
                }

            prefix = update.get("prefix")
            if prefix is None:
                continue

            peer_asn = update.get("peer_asn", "unknown")
            collector = rec.collector

            if prefix not in routes:
                routes[prefix] = {}
            if collector not in routes[prefix]:
                routes[prefix][collector] = {}

            # Processing Announcements (A) and Withdrawals (W)
            if elem.type == 'A':  # Announcement
                path = update.get('as-path', "").split()
                if path and path[-1] == target_asn:
                    routes[prefix][collector][peer_asn] = path
                    temp_counts["num_announcements"] += 1
            elif elem.type == 'W':  # Withdrawal
                if prefix in routes and collector in routes[prefix]:
                    if peer_asn in routes[prefix][collector]:
                        if routes[prefix][collector][peer_asn][-1] == target_asn:
                            routes[prefix][collector].pop(peer_asn, None)
                            temp_counts["num_withdrawals"] += 1

    print(f"Total records processed: {record_count}")
    print(f"Total elements processed: {element_count}")

    # Process the final 5-minute window
    features, old_routes_as = extract_features(index, routes, old_routes_as, target_asn, temp_counts)
    features['timestamp'] = current_window_start
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

def buildGraph(routes):
    G = nx.Graph()
    edges = set()

    for prefix in routes.keys():
        for collector in routes[prefix].keys():
            for peer in routes[prefix][collector].keys():
                path = routes[prefix][collector][peer]
                if path is not None:
                    path_vertices = path.split(" ")
                    for i in range(len(path_vertices) - 1):
                        a, b = path_vertices[i], path_vertices[i + 1]
                        if a != b:
                            edges.add((a, b))

    G.add_edges_from(edges)
    return G

def extract_bgp_data_and_build_weighted_graph(target_asn, from_time, until_time, collectors=['rrc00'], output_file=None):
    stream = pybgpstream.BGPStream(
        from_time=from_time,
        until_time=until_time,
        record_type="updates",
        collectors=collectors
    )

    routes = {}
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
            if elem.type == 'A':  # Announcement
                # print(f"Processing announcement for prefix {prefix}")
                as_path = update.get('as-path')
                if as_path:
                    # print(f"Raw as-path for prefix {prefix}: {as_path}")
                    path = as_path.split()
                    if path and path[-1] == target_asn:
                        if peer_asn not in routes[prefix][collector]:
                            routes[prefix][collector][peer_asn] = path  # Store the path correctly here
                            # print(f"Added path to routes: {path}")
                else:
                    print(f"No as-path found for prefix {prefix}")
            elif elem.type == 'W':  # Withdrawal
                # print(f"Processing withdrawal for prefix {prefix}")
                if prefix in routes and collector in routes[prefix]:
                    if peer_asn in routes[prefix][collector]:
                        if routes[prefix][collector][peer_asn][-1] == target_asn:
                            routes[prefix][collector].pop(peer_asn, None)

    print(f"Total records processed: {record_count}")
    print(f"Total elements processed: {element_count}")

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

    # Use spring layout for better visualization
    pos = nx.spring_layout(G, seed=42)  # Adding seed for consistent layouts across runs
    
    # Normalize edge weights to make them visually clearer
    edge_weights = np.array([G[u][v]['nbIp'] for u, v in G.edges()])
    max_weight = max(edge_weights) if len(edge_weights) > 0 else 1  # Prevent division by zero
    min_weight = min(edge_weights) if len(edge_weights) > 0 else 0
    normalized_weights = [(0.5 + (weight - min_weight) / (max_weight - min_weight)) * 5 
                          for weight in edge_weights]  # Scale weights between 0.5 and 5 for visualization
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", edgecolors='black', linewidths=0.5, alpha=0.9)
    
    # Draw edges with varying thickness
    edges = nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), width=normalized_weights, edge_color=edge_weights, edge_cmap=plt.cm.Blues, alpha=0.7
    )
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    # Add a color bar to represent edge weight intensities
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array([])  # This line is important to properly map the colors
    cbar = plt.colorbar(sm, ax=plt.gca())  # Attach the colorbar to the current axes
    cbar.set_label('Edge Weight (nbIp)', rotation=270, labelpad=15)
    
    # Set the title and display the graph
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()