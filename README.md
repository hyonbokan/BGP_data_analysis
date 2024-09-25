# BGP Routing Data Analysis and Feature Extraction

## Overview
This repository contains a collection of Python scripts utilizing PyBGPStream for analyzing BGP routing data, extracting relevant features, and performing graph-based analysis (e.g., AS path frequency). The scripts are designed to assist in understanding BGP routing behaviors, including updates, hijackings, and real-time analysis. Additionally, they provide tools for feature extraction and graphical analysis, allowing users to visualize and interpret the frequency of AS paths and other key metrics within the BGP data.

![BGP Feature Graph](images/googleLeakStat.png)

![BGP Graph](images/googleLeakGraph.png)

 **Install PyBGPStream**:
Visit [PyBGPStream Installation Guide](https://bgpstream.caida.org/docs/install/pybgpstream) for detailed installation instructions.

### Contents

1. **BGP Update Analysis Tutorial**: `bgp_updates_analysis.ipynb`
    
    A Jupyter notebook that provides a step-by-step guide on how to analyze BGP updates using PyBGPStream. This tutorial is for beginners looking to understand the basics of BGP data handling and use of tools such as BGPGo and PyBGPStream.

2. **Analysis of Real Cases of BGP Hijacking**: `PyBGPStream_real_cases.ipynb`,  `PyBGPStream_real_cases_2.ipynb`
    
    This notebook analyzes actual BGP hijacking incidents, demonstrating how to detect and examine suspicious BGP activities.

3. **Real-Time BGP Analysis Script**: `PyBGPStream_real_time.ipynb`

    A script designed for real-time analysis of BGP data streams. It can be used to monitor live BGP announcements and withdrawals for research and operational purposes.

4. **BGP Data Analysis and Feature Extraction**: `bgp_data_analysis_feature_extraction.ipynb`

4. **BGP Data Analysis and Feature Extraction**: `bgp_data_analysis_feature_extraction.ipynb`

    This notebook is designed for extracting detailed features from BGP data per Autonomous System Number (ASN), which can be utilized for machine learning or Large Language Model training. The script processes BGP data to calculate and extract various statistics and features.

    **Extracted Features Include**:
    - **Timestamp**: The time when the BGP update was recorded.
    - **Autonomous System Number (ASN)**: The target ASN for which the features are being analyzed.
    - **Total Routes**: The total number of active routes observed for the target ASN.
    - **New Routes**: The number of new routes added in the current observation window.
    - **Withdrawals**: The total number of route withdrawals observed.
    - **Origin Changes**: The count of changes in the origin ASN of routes.
    - **Route Changes**: The total changes observed in routing paths for the target ASN.
    - **Maximum Path Length**: The longest AS path observed in the BGP data.
    - **Average Path Length**: The average length of AS paths observed during the collection window.
    - **Maximum Edit Distance**: The highest edit distance between AS paths compared to previous updates.
    - **Average Edit Distance**: The average edit distance observed across all AS paths in the current window.
    - **Announcements**: The total number of BGP announcements received for the target ASN.
    - **Unique Prefixes Announced**: The number of unique IP prefixes announced by the target ASN.
    - **Graph Average Degree**: The average degree of nodes in the constructed AS path graph.
    - **Graph Betweenness Centrality**: The average betweenness centrality of nodes in the AS path graph, indicating the importance of nodes in the shortest paths between other nodes.
    - **Graph Closeness Centrality**: The average closeness centrality of nodes in the AS path graph, representing how close each node is to all other nodes in the graph.
    - **Graph Eigenvector Centrality**: The average eigenvector centrality of nodes in the AS path graph, indicating the influence or importance of nodes within the network.

 