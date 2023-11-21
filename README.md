# BGP Routing Data Analysis with PyBGPStream

The repository contains:
* Introduction to BGP: My personal attempt to answer the questions in Assignment 7 from the course CSE291E Internet Data Science for Cybersecurity, taught by Prof. KC Claffy at UC San Diego.
Course Link: [CSE291E Internet Data Science for Cybersecurity](https://cseweb.ucsd.edu/classes/wi23/cse291-e/syllabus.html)

* Analysis of real-life cases of BGP hijacking incidents utilizing PyBGPStream. By referencing publicly available documentation of these events. I have created Python scripts using PyBGPStream to detect and visualize the anomalies resulting from these hijacking incidents.

## Description
The goal of this assignment is to explore and analyze raw BGP (Border Gateway Protocol) routing data to address the research question of which collector peers forward ROA-invalid updates to the collector. The assignment covers various aspects of BGP data analysis, including parsing MRT files, using CAIDA's BGP2GO system, and working with BGPStream. Additionally, a Python script is included in the repository to demonstrate the extraction of BGP updates using BGPStream.

# Contents
- `Explore ROV deployment using.pdf`: The original file of Assignment 7 provided by the course.
- `bgp_updates_analysis.ipynb`: Jupyter notebook with Python scripts demonstrating BGP updates extraction using BGPStream.
- `download_mrt.py`: Python script for downloading raw MRT data from [University of Oregon Route Views Archive Project](http://archive.routeviews.org/)
- `PyBGPStream_real_cases`:  Jupyter notebook containing the Python scripts analyze and plot the BGP anomalies associated with real-life BGP hijacking incidents.

## BGP Data

Please note that the actual data of BGP update messages for the ROV-invalid prefixes was not included in the repository due to its size. However, the script of collecting update messages with required filters is provided in `bgp_updates_analysis.ipynb`.

The data is useful for analyzing the behavior and trends of ROV-invalid prefixes over time.

# Disclaimer
**Please note** that I am not a student at UC San Diego, and this repository represents my personal attempt at completing the assignment. As a self-learner, I found it valuable to work on fundamental CS courses independently. While I have made efforts to ensure accuracy, there may be mistakes. Nonetheless, I hope this repository can be helpful to others with a similar background who are learning CS fundamental courses on their own.


Feel free to explore the contents of the repository and refer to the provided files for more details.