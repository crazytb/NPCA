#!/usr/bin/env python3
"""
Standalone script to run random access simulation and analyze throughput
"""

import numpy as np
import pandas as pd
import random

def randomaccess(numnodes, beaconinterval, frametxslot, per, raalgo):
    if raalgo == 'CSMA':
        slottime = 9
        currentslot = 0
        contentionwindowsize = [2**x for x in range(5, 11)]
        # Backoff counter initialize
        bo_stage = np.zeros(numnodes, dtype=int)
        bo_counter = np.array([np.random.randint(contentionwindowsize[x]) for x in bo_stage])

        num_succ = 0
        num_fail = 0
        num_coll = 0
        succ_timestamp = np.array([], dtype=int)
        succ_namestamp = np.array([], dtype=int)
        retx = np.zeros(numnodes, dtype=int)
        aoitimestamp = np.zeros(numnodes, dtype=int)
        df = pd.DataFrame(columns=['time', 'node', 'timestamp', 'result'])

        while currentslot < beaconinterval/slottime - frametxslot:
            per_rv = random.random()
            # Idle
            if np.min(bo_counter) != 0:
                currentslot += np.min(bo_counter)
                aoitimestamp[retx == 0] = currentslot
                bo_counter = bo_counter - np.min(bo_counter)
            # Tx succ
            elif (per_rv > per) and ((np.min(bo_counter) == 0) and (np.size(bo_counter) - np.count_nonzero(bo_counter) == 1)):
                currentslot += frametxslot
                ind, = np.where(bo_counter == 0)
                retx[ind] = 0
                df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind + 1, 'timestamp': aoitimestamp[ind]*slottime, 'result': 'succ'})
                df = pd.concat([df, df2], ignore_index=True, axis=0)
                aoitimestamp[retx == 0] = currentslot
                succ_timestamp = np.append(succ_timestamp, currentslot)
                succ_namestamp = np.append(succ_namestamp, ind[0])
                bo_stage[ind] = 0
                bo_counter[ind] = np.random.randint(contentionwindowsize[0])
                num_succ += 1
            # Tx failed
            elif (per_rv <= per) and ((np.min(bo_counter) == 0) and (np.size(bo_counter) - np.count_nonzero(bo_counter) == 1)):
                currentslot += frametxslot
                ind, = np.where(bo_counter == 0)
                retx[ind] = 1
                for x in ind:
                    if bo_stage[x] < 5:
                        bo_stage[x] += 1
                    bo_counter[x] = np.random.randint(contentionwindowsize[bo_stage[x]])
                df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind + 1, 'timestamp': aoitimestamp[ind]*slottime, 'result': 'fail'})
                df = pd.concat([df, df2], ignore_index=True, axis=0)
                aoitimestamp[retx == 0] = currentslot
                num_fail += 1
            # Tx coll
            elif np.min(bo_counter) == 0 and (np.size(bo_counter) - np.count_nonzero(bo_counter) > 1):
                currentslot += frametxslot
                ind, = np.where(bo_counter == 0)
                retx[ind] = 1
                for x in ind:
                    if bo_stage[x] < 5:
                        bo_stage[x] += 1
                    bo_counter[x] = np.random.randint(contentionwindowsize[bo_stage[x]])
                df2 = pd.DataFrame({'time': currentslot*slottime, 'node': ind + 1, 'timestamp': aoitimestamp[ind]*slottime, 'result': 'coll'})
                df = pd.concat([df, df2], ignore_index=True, axis=0)
                aoitimestamp[retx == 0] = currentslot
                num_coll += 1

        return df

def analyze_throughput(dflog, num_nodes=10, beacon_interval=100000, frame_tx_slot=300, slot_time=9):
    """
    Analyze throughput for each station based on the simulation log.
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

if __name__ == "__main__":
    # Run the simulation
    print("Running CSMA random access simulation...")
    dflog = randomaccess(numnodes=10, beaconinterval=100000, frametxslot=300, per=0, raalgo='CSMA')
    
    print(f"Simulation completed. Total events: {len(dflog)}")
    print("\nFirst few events:")
    print(dflog.head())
    
    # Analyze throughput
    print("\n" + "="*60)
    throughput_results = analyze_throughput(dflog) 