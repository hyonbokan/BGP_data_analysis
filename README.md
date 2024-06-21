# BGP Routing Data Analysis and Feature Extraction

## Overview
This repository contains a collection of Python scripts utilizing PyBGPStream for analyzing BGP routing data and extracting relevant features. These scripts are designed to aid in the understanding and research of BGP routing behaviors, including updates, hijackings, and real-time analysis.

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
    - **Timestamp**: Time of the BGP update.
    - **ASN (Autonomous System Number)**: The target ASN for which features are being extracted.
    - **Number of Routes**: Total number of active routes observed.
    - **Number of New Routes**: Number of new routes added.
    - **Number of Withdrawals**: Total route withdrawals observed.
    - **Number of Origin Changes**: Changes in the origin ASN of routes.
    - **Number of Route Changes**: Changes in the routing paths.
    - **Maximum Path Length**: The longest AS path observed.
    - **Average Path Length**: The average length of AS paths.
    - **Maximum Edit Distance**: Maximum edit distance observed in AS paths from one update to the next.
    - **Average Edit Distance**: Average edit distance across all updates.
    - **Number of Announcements**: Total BGP announcements observed.
    - **Number of Unique Prefixes Announced**: Number of unique IP prefixes announced.

