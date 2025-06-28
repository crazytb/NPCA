# Random Access Simulation - Throughput Analysis

This repository contains code for analyzing throughput in random access network simulations.

## Files

- `random_access.ipynb`: Original Jupyter notebook with CSMA and Slotted ALOHA simulations
- `throughput_analysis.py`: Standalone throughput analysis function
- `run_throughput_analysis.py`: Complete script that runs simulation and analyzes throughput
- `notebook_throughput_code.py`: Code snippet for use in Jupyter notebook

## How to Use

### Option 1: Use in Jupyter Notebook

1. Open `random_access.ipynb` in Jupyter
2. Run the existing cells to generate `dflog`
3. Copy the code from `notebook_throughput_code.py` into a new cell
4. Run the cell to see throughput analysis

### Option 2: Run Standalone Script

```bash
python run_throughput_analysis.py
```

### Option 3: Import Function

```python
from throughput_analysis import analyze_throughput

# After generating dflog from your simulation
results = analyze_throughput(dflog)
```

## Throughput Calculation

The throughput for each station is calculated as:

```
Throughput = (Number of Successful Transmissions × Frame Duration) / Total Simulation Time
```

Where:
- Frame Duration = frame_tx_slot × slot_time = 300 × 9 = 2700 microseconds
- Total Simulation Time = beacon_interval = 100000 microseconds

## Output

The analysis provides:

1. **Per-station throughput**: Individual throughput for each node
2. **System throughput**: Total system utilization
3. **Fairness index**: Jain's fairness index (1.0 = perfectly fair)
4. **Result distribution**: Count of successful, failed, and collision events
5. **Visualization**: Bar charts showing throughput and transmission counts (if matplotlib available)

## Example Output

```
Throughput Analysis for Each Station:
==================================================
 node  successful_tx  throughput  throughput_percentage
    1              4       0.108                   10.8
    2              3       0.081                    8.1
    3              3       0.081                    8.1
    ...

Total System Throughput: 0.6750 (67.50%)
Total Successful Transmissions: 25
Average Throughput per Node: 0.0675 (6.75%)

Fairness Index (Jain's): 0.8562
```

## Parameters

- `num_nodes`: Number of stations (default: 10)
- `beacon_interval`: Total simulation time in microseconds (default: 100000)
- `frame_tx_slot`: Frame transmission time in slots (default: 300)
- `slot_time`: Duration of each slot in microseconds (default: 9) 