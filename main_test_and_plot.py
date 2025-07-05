import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random_access.random_access_rev import SimplifiedCSMACASimulation, SLOTTIME

simulation_time = 100000  # slots
frame_size = 330  # slots
    
def run_multichannel_simulation():
    """Run simulations for different channel configurations"""
    
    # Configuration
    simulation_configs = [
        {"num_channels": 2, "stas_per_channel": [2, 5], "label": "default"},
    ]
    
    results = {}
    
    for config in simulation_configs:
        print(f"Running simulation: {config['label']}")
        
        sim = SimplifiedCSMACASimulation(
            num_channels=config["num_channels"],
            stas_per_channel=config["stas_per_channel"],
            simulation_time=simulation_time,
            frame_size=frame_size
        )
        
        df = sim.run()
        df.to_csv(f'csma_ca_fsm_log_{config["label"].replace(" ", "_").lower()}.csv', index=False)
        print(f"Simulation completed: {config['label']}")
        stats = sim.get_statistics()
        
        results[config['label']] = {
            'config': config,
            'stats': stats,
            'dataframe': df
        }
    
    return results

def calculate_throughput(stats, simulation_time, frame_size):
    """Calculate throughput for each station and channel based on channel occupation time"""
    throughput_data = {}
    
    for sta_id, sta_stats in stats['stations'].items():
        channel = sta_stats['channel']
        if channel not in throughput_data:
            throughput_data[channel] = {}
        
        # Calculate throughput as channel occupation time / total simulation time
        # Throughput = (successful transmissions × frame_size) / total_simulation_time
        successful_channel_occupation_slots = sta_stats['successful_transmissions'] * frame_size
        throughput = successful_channel_occupation_slots / simulation_time
        throughput_data[channel][f'STA_{sta_id}'] = throughput
    
    return throughput_data

def plot_multichannel_performance(results):
    """Create performance comparison plots for 2-channel environment"""
    
    # Fixed configuration: 2 channels
    num_channels = 2
    
    # Create figure with subplots: 2 rows x 2 columns (one for each channel)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Color maps for STA identification
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    # Get the single configuration result
    config_name, result = list(results.items())[0]
    config = result['config']
    stats = result['stats']
    
    # Process each channel
    for ch_id in range(num_channels):
        # Get STAs for this channel
        channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                       if sta_stats['channel'] == ch_id]
        channel_stas.sort()  # Sort by STA ID
        
        # === THROUGHPUT PLOT (First Row) ===
        ax_throughput = axes[0, ch_id]
        
        if channel_stas:
            sta_ids = [sta_id for sta_id, _ in channel_stas]
            sta_labels = [f'STA {sta_id}' for sta_id in sta_ids]
            
            # Calculate individual throughput for each STA (channel occupation time)
            throughputs = []
            for sta_id, sta_stats in channel_stas:
                # Throughput = (successful transmissions × frame_size) / total_simulation_time
                successful_channel_occupation_slots = sta_stats['successful_transmissions'] * frame_size
                throughput = successful_channel_occupation_slots / stats['total_slots']
                throughputs.append(throughput)
            
            # Create stacked bar chart
            bottom = 0
            for i, (sta_id, throughput) in enumerate(zip(sta_ids, throughputs)):
                bar = ax_throughput.bar(0, throughput, bottom=bottom,
                                      label=f'STA {sta_id}',
                                      color=colors[sta_id % len(colors)],
                                      alpha=0.8, edgecolor='white', linewidth=1)
                bottom += throughput
                
                # Add text annotation in the middle of each segment
                if throughput > 0.01:  # Only annotate if segment is large enough
                    text_y = bottom - throughput/2
                    ax_throughput.text(0, text_y, f'{throughput:.3f}', 
                                     ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Total throughput annotation
            ax_throughput.text(0, bottom + 0.01, f'Total: {bottom:.3f}', 
                             ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax_throughput.set_title(f'Channel {ch_id} Throughput\n({len(channel_stas)} STAs)', 
                               fontsize=14, fontweight='bold')
        ax_throughput.set_ylabel('Throughput (fraction)', fontsize=12)
        ax_throughput.set_xlim(-0.4, 0.4)
        ax_throughput.set_ylim(0, 1)  # Fixed y-axis range from 0 to 1
        ax_throughput.set_xticks([])
        ax_throughput.grid(True, alpha=0.3, axis='y')
        ax_throughput.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        # === AoI PLOT (Second Row) ===
        ax_aoi = axes[1, ch_id]
        
        if channel_stas:
            sta_ids = [sta_id for sta_id, _ in channel_stas]
            aoi_values = [sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas]
            
            # Create individual bar chart for each STA
            bars = ax_aoi.bar(range(len(sta_ids)), aoi_values,
                             color=[colors[sta_id % len(colors)] for sta_id in sta_ids],
                             alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value annotations on bars
            for bar, aoi_val in zip(bars, aoi_values):
                height = bar.get_height()
                ax_aoi.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{aoi_val:.0f}', ha='center', va='bottom', fontweight='bold')
            
            ax_aoi.set_xticks(range(len(sta_ids)))
            ax_aoi.set_xticklabels([f'STA {sta_id}' for sta_id in sta_ids])
        
        ax_aoi.set_title(f'Channel {ch_id} Average AoI\n({len(channel_stas)} STAs)', 
                        fontsize=14, fontweight='bold')
        ax_aoi.set_ylabel('Average AoI (μs)', fontsize=12)
        ax_aoi.set_xlabel('Station ID', fontsize=12)
        ax_aoi.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('multichannel_performance_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_summary_statistics(results):
    """Create summary comparison plots for single configuration"""
    
    # Get the single configuration result
    config_name, result = list(results.items())[0]
    config = result['config']
    stats = result['stats']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Separate statistics by channel
    channel_0_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                     if sta_stats['channel'] == 0]
    channel_1_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                     if sta_stats['channel'] == 1]
    
    channels_data = [
        ('Channel 0', channel_0_stas),
        ('Channel 1', channel_1_stas)
    ]
    
    colors = ['skyblue', 'lightcoral']
    
    # Plot 1: Channel Throughput Comparison
    channel_names = []
    channel_throughputs = []
    
    for ch_name, ch_stas in channels_data:
        channel_names.append(ch_name)
        # Calculate channel throughput as total channel occupation time / total simulation time
        total_successful_transmissions = sum(sta_stats['successful_transmissions'] for _, sta_stats in ch_stas)
        total_channel_occupation_slots = total_successful_transmissions * frame_size
        channel_throughput = total_channel_occupation_slots / stats['total_slots']
        channel_throughputs.append(channel_throughput)
    
    bars1 = ax1.bar(channel_names, channel_throughputs, alpha=0.7, color=colors)
    ax1.set_title('Channel Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Throughput (fraction)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels and STA count
    for bar, value, (ch_name, ch_stas) in zip(bars1, channel_throughputs, channels_data):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}\n({len(ch_stas)} STAs)', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Average AoI per Channel
    channel_aois = []
    for ch_name, ch_stas in channels_data:
        if ch_stas:
            avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in ch_stas])
            channel_aois.append(avg_aoi)
        else:
            channel_aois.append(0)
    
    bars2 = ax2.bar(channel_names, channel_aois, alpha=0.7, color=colors)
    ax2.set_title('Average AoI per Channel', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average AoI (μs)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, channel_aois):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Collisions per Channel
    channel_collisions = []
    for ch_name, ch_stas in channels_data:
        total_collisions = sum(sta_stats['collisions'] for _, sta_stats in ch_stas)
        channel_collisions.append(total_collisions)
    
    bars3 = ax3.bar(channel_names, channel_collisions, alpha=0.7, color=colors)
    ax3.set_title('Total Collisions per Channel', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Collisions', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, channel_collisions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Success Rate per Channel
    channel_success_rates = []
    for ch_name, ch_stas in channels_data:
        if ch_stas:
            avg_success = np.mean([sta_stats['success_rate'] for _, sta_stats in ch_stas])
            channel_success_rates.append(avg_success * 100)
        else:
            channel_success_rates.append(0)
    
    bars4 = ax4.bar(channel_names, channel_success_rates, alpha=0.7, color=colors)
    ax4.set_title('Average Success Rate per Channel', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Success Rate (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Add value labels
    for bar, value in zip(bars4, channel_success_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('multichannel_summary_statistics.png', dpi=300, bbox_inches='tight')
    # plt.show()

def print_detailed_results(results):
    """Print detailed numerical results"""
    
    print("\n" + "="*80)
    print("MULTI-CHANNEL CSMA/CA PERFORMANCE COMPARISON")
    print("="*80)
    
    for config_name, result in results.items():
        stats = result['stats']
        config = result['config']
        
        print(f"\n📊 {config_name}")
        print("-" * 50)
        print(f"Simulation time: {stats['total_time_us']/1000:.1f} ms ({stats['total_slots']} slots)")
        print(f"Channels: {config['num_channels']}")
        print(f"STAs per channel: {config['stas_per_channel']}")
        
        # Calculate system metrics
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        total_attempts = sum(sta_stats['total_attempts'] for sta_stats in stats['stations'].values())
        total_collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * frame_size) / stats['total_slots']
        system_success_rate = total_successful / total_attempts if total_attempts > 0 else 0
        
        print(f"System throughput: {system_throughput:.4f} ({system_throughput*100:.2f}%)")
        print(f"System success rate: {system_success_rate:.4f} ({system_success_rate*100:.2f}%)")
        print(f"Total successful transmissions: {total_successful}")
        print(f"Total collisions: {total_collisions}")
        
        # Per-channel breakdown
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                          if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                print(f"\n  Channel {ch_id}:")
                for sta_id, sta_stats in channel_stas:
                    throughput = (sta_stats['successful_transmissions'] * frame_size) / stats['total_slots']
                    print(f"    STA {sta_id}: Throughput={throughput:.4f}, "
                          f"AoI={sta_stats['average_aoi_time_us']:.1f}μs, "
                          f"Success={sta_stats['success_rate']:.2%}, "
                          f"Collisions={sta_stats['collisions']}")

if __name__ == "__main__":
    # Run simulations
    print("Starting multi-channel CSMA/CA simulations...")
    results = run_multichannel_simulation()
    
    # Generate visualizations
    print("\nGenerating performance comparison plots...")
    plot_multichannel_performance(results)
    
    print("Generating summary statistics plots...")
    plot_summary_statistics(results)
    
    # Print detailed results
    print_detailed_results(results)
    
    print("\n✅ Analysis complete!")
    print("📁 Plots saved as:")
    print("   - multichannel_performance_comparison.png")
    print("   - multichannel_summary_statistics.png")