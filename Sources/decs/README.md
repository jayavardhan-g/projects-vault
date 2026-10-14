# DECS Project

## Overview
This repository contains the DECS project, which includes a server, client, and load generator for testing and benchmarking purposes. The project is structured as follows:

- `server/` and `server_debug/`: Contains the server implementation.
- `client/` and `client_small/`: Contains the client implementation.
- `load_gen/`: Contains scripts for generating load and analyzing results.
- `results/`: Stores output CSV files from load generation tests.

## Build Instructions

### Prerequisites
- A C++ compiler (e.g., `g++` for Linux/Windows).
- Node.js for running JavaScript load generator scripts.
- Python for running analysis scripts.

### Build Commands
To build the server and client, use the following commands:

#### Server
```bash
g++ -o server server.cpp -lpqxx -lpq -lpthread
```

#### Client
```bash
g++ -o client client.cpp
```

#### Client (Small)
```bash
g++ -o client_small client_small.cpp -lpthread
```

## Load Generator Usage

### JavaScript Scripts
The `load_gen/` folder contains JavaScript scripts for generating load. Use Node.js to run these scripts. For example:

```bash
./script.sh
```
You can change the type of workload and duration for each workload using arguments or can just modify the script.sh.

### Python Scripts
The `load_gen/` folder also contains Python scripts for analyzing results. For example:

```bash
python csv_plot.py results_<work_load>.csv
```
Plots are saved in the results/ folder in load_gen/

### Output
The results of the load generation tests are stored as CSV files in the `load_gen/` folder and its subdirectories.

## Key Files and Folders

- `server.cpp`: Main server implementation.
- `client.cpp`: Main client implementation.
- `client_small.cpp`: Lightweight client implementation.
- `kvcache.h`: Header file for key-value cache.
- `httplib.h`: Header file for HTTP library.
- `load_gen/`: Contains load generation and analysis scripts.
- `results/`: Stores output CSV files from tests.

## Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the scripts as needed to suit your testing requirements.

# Other tools
- Use htop to monitor real time CPU usage.
- Use iostat -dx 1 to monitor real time disk utilization.