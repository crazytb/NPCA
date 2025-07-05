import pandas as pd
import matplotlib.pyplot as plt
from random_access.random_access_rev import *

if __name__ == "__main__":
    # Create simulation
    sim = SimplifiedCSMACASimulation(
        num_channels=2,
        stas_per_channel=[2, 5],
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
        print(f"  Average AoI: {sta_stats['average_aoi_slots']:.1f} slots ({sta_stats['average_aoi_time_us']:.1f} μs)")

    # Save detailed logs
    df.to_csv('csma_ca_fsm_log.csv', index=False)
    print(f"\nDetailed logs saved to 'csma_ca_fsm_log.csv'")

    # Additional AoI statistics
    print("\n=== AoI Summary ===")
    all_aoi_slots = [sta_stats['average_aoi_slots'] for sta_stats in stats['stations'].values()]
    print(f"Overall average AoI: {sum(all_aoi_slots)/len(all_aoi_slots):.1f} slots ({sum(all_aoi_slots)/len(all_aoi_slots)*SLOTTIME:.1f} μs)")
    print(f"Min AoI: {min(all_aoi_slots):.1f} slots")
    print(f"Max AoI: {max(all_aoi_slots):.1f} slots")