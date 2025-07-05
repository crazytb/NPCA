import pandas as pd
import matplotlib.pyplot as plt
from random_access.random_access_rev import *

if __name__ == "__main__":
    # Create simulation
    sim = SimplifiedCSMACASimulation(
        num_channels=1,
        stas_per_channel=[5],
        simulation_time=100000,  # slots
        frame_size=33,  # slots (300μs / 9μs)
    )
    
    # Run simulation
    df = sim.run()
    
    # Get statistics
    stats = sim.get_statistics()
    
    # Print results
    print("Simulation Statistics:")
    print(f"Total simulation time: {stats['total_time_us']} μs")
    print(f"Total slots: {stats['total_slots']}")
    print("\nStation Statistics:")
    for sta_id, sta_stats in stats['stations'].items():
        print(f"Station {sta_id}:")
        print(f"  Channel: {sta_stats['channel']}")
        print(f"  Successful transmissions: {sta_stats['successful_transmissions']}")
        print(f"  Collisions: {sta_stats['collisions']}")
        print(f"  Success rate: {sta_stats['success_rate']:.2%}")
        print(f"  Final state: {sta_stats['final_state']}")
    
    # Save detailed logs
    df.to_csv('csma_ca_fsm_log.csv', index=False)
    print(f"\nDetailed logs saved to 'csma_ca_fsm_log.csv'")