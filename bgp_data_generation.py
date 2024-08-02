import pybgpstream
import editdistance
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import get_named_colors_mapping

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
        "num_unique_prefixes_announced": 0
    }

    routes_as = build_routes_as(routes)

    if index > 0:
        if target_asn in routes_as:
            num_routes = len(routes_as[target_asn])
            sum_path_length = 0
            sum_edit_distance = 0

            for prefix in routes_as[target_asn].keys():
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

            if elem_time >= current_window_start + timedelta(minutes=5):
                features, old_routes_as = extract_features(index, routes, old_routes_as, target_asn, temp_counts)
                features['timestamp'] = current_window_start
                all_features.append(features)

                # Reset for the new window
                current_window_start += timedelta(minutes=5)
                routes = {}
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

            if elem.type == 'A':
                path = update['as-path'].split()
                if path[-1] == target_asn:
                    routes[prefix][collector][peer_asn] = path
                    temp_counts["num_announcements"] += 1
            elif elem.type == 'W':
                if prefix in routes and collector in routes[prefix]:
                    if peer_asn in routes[prefix][collector]:
                        if routes[prefix][collector][peer_asn][-1] == target_asn:
                            routes[prefix][collector].pop(peer_asn, None)
                            temp_counts["num_withdrawals"] += 1

    print(f"Total records processed: {record_count}")
    print(f"Total elements processed: {element_count}")

    # Final aggregation for the last window
    features, old_routes_as = extract_features(index, routes, old_routes_as, target_asn, temp_counts)
    features['timestamp'] = current_window_start
    all_features.append(features)

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
