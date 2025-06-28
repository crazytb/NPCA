# Calculate throughput for each station based on dflog
import numpy as np
import pandas as pd

def analyze_throughput(dflog, num_nodes=10, beacon_interval=100000, frame_tx_slot=300, slot_time=9):
    """
    Analyze throughput for each station based on the simulation log.
    
    Parameters:
    dflog: DataFrame with columns ['time', 'node', 'timestamp', 'result']
    num_nodes: Number of nodes in the simulation
    beacon_interval: Total simulation time in microseconds
    frame_tx_slot: Frame transmission time in slots
    slot_time: Duration of each slot in microseconds
    """
    
    # Calculate frame duration in microseconds
    frame_duration = frame_tx_slot * slot_time  # microseconds
    
    # Filter successful transmissions only
    successful_tx = dflog[dflog['result'] == 'succ']
    
    # Calculate throughput for each node
    throughput_data = []
    for node in range(1, num_nodes + 1):
        node_tx = successful_tx[successful_tx['node'] == node]
        num_successful_tx = len(node_tx)
        
        # Throughput = (Number of successful transmissions * Frame duration) / Total simulation time
        throughput = (num_successful_tx * frame_duration) / beacon_interval
        
        throughput_data.append({
            'node': node,
            'successful_tx': num_successful_tx,
            'throughput': throughput,
            'throughput_percentage': throughput * 100
        })
    
    # Create DataFrame for throughput results
    throughput_df = pd.DataFrame(throughput_data)
    
    # Display results
    print("Throughput Analysis for Each Station:")
    print("=" * 50)
    print(throughput_df.to_string(index=False))
    print("\n" + "=" * 50)
    
    # Calculate total system throughput
    total_throughput = throughput_df['throughput'].sum()
    total_successful_tx = throughput_df['successful_tx'].sum()
    
    print(f"Total System Throughput: {total_throughput:.4f} ({total_throughput*100:.2f}%)")
    print(f"Total Successful Transmissions: {total_successful_tx}")
    print(f"Average Throughput per Node: {total_throughput/num_nodes:.4f} ({total_throughput*100/num_nodes:.2f}%)")
    
    # Additional analysis: Show distribution of results
    print("\nTransmission Result Distribution:")
    result_counts = dflog['result'].value_counts()
    print(result_counts)
    
    # Calculate fairness index (Jain's fairness index)
    throughputs = throughput_df['throughput'].to_numpy()  # Convert to numpy array
    fairness_index = (np.sum(throughputs) ** 2) / (len(throughputs) * np.sum(throughputs ** 2))
    print(f"\nFairness Index (Jain's): {fairness_index:.4f}")
    print("Note: 1.0 = perfectly fair, lower values indicate unfairness")
    
    return throughput_df

# Example usage (uncomment and run after generating dflog):
# throughput_results = analyze_throughput(dflog) 