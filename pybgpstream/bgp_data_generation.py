# python bgp_data_collector.py --start_time "2021-02-11 03:30:00" --end_time "2021-02-11 05:59:59" --message_type "updates" --collectors "route-views.sg" "route-views.eqix" --asn "28548"

import pybgpstream
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect and store BGP data using pybgpstream.')
    parser.add_argument('--start_time', type=str, required=True, help='Start time in UTC (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, required=True, help='End time in UTC (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--message_type', type=str, choices=['ribs', 'updates'], required=True, help='Type of BGP message to collect (ribs or updates)')
    parser.add_argument('--collectors', type=str, nargs='+', default=['rrc00'], help='List of BGP collectors to use (default: rrc00)')
    parser.add_argument('--asn', type=str, default=None, help='Target ASN for filtering (default: None)')
    return parser.parse_args()

def collect_bgp_data(start_time, end_time, message_type, collectors, asn):
    stream = pybgpstream.BGPStream(
        from_time=start_time,
        until_time=end_time,
        collectors=collectors,
        record_type=message_type
    )

    aggregate_bgp_data = []
    summary_stats = defaultdict(lambda: defaultdict(lambda: {'total_prefixes': 0, 'unique_as_paths': set(), 'total_announcements': 0, 'total_withdrawals': 0}))

    for rec in stream.records():
        for elem in rec:
            if elem.type in {"R", "A", "W"}:
                fields = elem.fields
                timestamp = datetime.utcfromtimestamp(rec.time).strftime('%Y-%m-%d %H:%M:%S')
                collector_project = rec.project
                collector_name = rec.collector
                prefix = fields.get("prefix", "")
                as_path = fields.get("as-path", "").split()
                next_hop = fields.get("next-hop", "")
                origin_as = as_path[-1] if as_path else ""
                community = fields.get("community", "")
                atomic_aggregate = fields.get("atomic-aggregate", "")
                aggregator = fields.get("aggregator", "")

                if asn and asn not in as_path:
                    continue

                aggregate_bgp_data.append({
                    'timestamp': timestamp,
                    'collector_project': collector_project,
                    'collector_name': collector_name,
                    'origin_as': origin_as,
                    'next_hop': next_hop,
                    'prefix': prefix,
                    'as_path': ' '.join(as_path),
                    'community': community,
                    'atomic_aggregate': atomic_aggregate,
                    'aggregator': aggregator,
                    'update_type': elem.type
                })

                summary_stats[origin_as][timestamp]['total_prefixes'] += 1
                summary_stats[origin_as][timestamp]['unique_as_paths'].add(' '.join(as_path))
                if elem.type == "A":
                    summary_stats[origin_as][timestamp]['total_announcements'] += 1
                elif elem.type == "W":
                    summary_stats[origin_as][timestamp]['total_withdrawals'] += 1

    for as_stats in summary_stats.values():
        for stats in as_stats.values():
            stats['unique_as_paths'] = len(stats['unique_as_paths'])

    df_aggregate = pd.DataFrame(aggregate_bgp_data)
    df_analysis = pd.concat({origin_as: pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'timestamp'}) for origin_as, data in summary_stats.items()})

    return df_aggregate, df_analysis

def main():
    args = parse_arguments()

    df_aggregate, df_analysis = collect_bgp_data(
        start_time=args.start_time,
        end_time=args.end_time,
        message_type=args.message_type,
        collectors=args.collectors,
        asn=args.asn
    )

    df_aggregate.to_csv('aggregate_bgp_data.csv', index=False)
    df_analysis.to_csv('bgp_analysis.csv', index=False)

    print("Aggregate DataFrame:")
    print(df_aggregate.head())
    print("\nAnalysis DataFrame:")
    print(df_analysis.head())

    plt.figure(figsize=(12, 6))
    for origin_as, group in df_analysis.groupby(level=0):
        timestamps = sorted(group['timestamp'])
        counts = group['total_prefixes']
        plt.plot(timestamps, counts, marker='o', linestyle='-', label=f'AS{origin_as}')

    plt.xlabel('Time (Minute)')
    plt.ylabel('Number of Prefix Updates')
    plt.title('Prefix Updates Over Time by Origin AS')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Origin AS')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
