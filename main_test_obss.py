import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random_access.random_access_rev import SimplifiedCSMACASimulation, SLOTTIME

def run_obss_comparison():
    """Run simulations comparing OBSS enabled vs disabled"""
    
    # Configuration
    simulation_configs = [
        {"num_channels": 2, "stas_per_channel": [2, 5], "obss_enabled": False, 
         "obss_threshold": 0.0, "label": "No OBSS"},
        {"num_channels": 2, "stas_per_channel": [2, 5], "obss_enabled": True, 
         "obss_threshold": 0.3, "label": "OBSS 30%"},
        {"num_channels": 2, "stas_per_channel": [2, 5], "obss_enabled": True, 
         "obss_threshold": 0.7, "label": "OBSS 70%"},
    ]
    
    simulation_time = 500000  # slots
    frame_size = 330  # slots
    
    results = {}
    
    for config in simulation_configs:
        print(f"Running simulation: {config['label']}")
        
        sim = SimplifiedCSMACASimulation(
            num_channels=config["num_channels"],
            stas_per_channel=config["stas_per_channel"],
            simulation_time=simulation_time,
            frame_size=frame_size,
            obss_enabled=config["obss_enabled"],
            obss_threshold=config["obss_threshold"]
        )
        
        df = sim.run()
        stats = sim.get_statistics()
        
        results[config['label']] = {
            'config': config,
            'stats': stats,
            'dataframe': df
        }
    
    return results

def plot_obss_comparison(results):
    """Create OBSS comparison plots"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = ['skyblue', 'orange', 'lightcoral']
    
    # Plot 1: System Throughput Comparison
    ax1 = axes[0, 0]
    system_throughputs = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * 33) / stats['total_slots']
        system_throughputs.append(system_throughput)
    
    bars1 = ax1.bar(config_names, system_throughputs, alpha=0.7, color=colors)
    ax1.set_title('System Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Throughput (fraction)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(system_throughputs) * 1.2)
    
    # Add value labels
    for bar, value in zip(bars1, system_throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Average AoI Comparison
    ax2 = axes[0, 1]
    avg_aois = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        all_aois = [sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()]
        avg_aoi = np.mean(all_aois)
        avg_aois.append(avg_aoi)
    
    bars2 = ax2.bar(config_names, avg_aois, alpha=0.7, color=colors)
    ax2.set_title('Average AoI Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average AoI (μs)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, avg_aois):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Total Collisions Comparison
    ax3 = axes[1, 0]
    total_collisions = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        total_collisions.append(collisions)
    
    bars3 = ax3.bar(config_names, total_collisions, alpha=0.7, color=colors)
    ax3.set_title('Total Collisions Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Collisions', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, total_collisions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: OBSS Interference Events
    ax4 = axes[1, 1]
    obss_events = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        obss_count = stats.get('obss_interference_count', 0)
        obss_events.append(obss_count)
    
    bars4 = ax4.bar(config_names, obss_events, alpha=0.7, color=colors)
    ax4.set_title('OBSS Interference Events', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of OBSS Events', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, obss_events):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('obss_comparison.png', dpi=300, bbox_inches='tight')

def plot_channel_specific_obss_impact(results):
    """Plot OBSS impact on individual channels"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    config_names = list(results.keys())
    colors = ['skyblue', 'orange', 'lightcoral']
    
    # For each channel, plot throughput and AoI comparison
    for ch_id in range(2):  # 2 channels
        # Channel throughput comparison
        ax_throughput = axes[ch_id, 0]
        channel_throughputs = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            total_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
            channel_throughput = (total_successful * 33) / stats['total_slots']
            channel_throughputs.append(channel_throughput)
        
        bars = ax_throughput.bar(config_names, channel_throughputs, alpha=0.7, color=colors)
        ax_throughput.set_title(f'Channel {ch_id} Throughput', fontsize=12, fontweight='bold')
        ax_throughput.set_ylabel('Throughput', fontsize=10)
        ax_throughput.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, channel_throughputs):
            ax_throughput.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                              f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Channel AoI comparison
        ax_aoi = axes[ch_id, 1]
        channel_aois = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                channel_aois.append(avg_aoi)
            else:
                channel_aois.append(0)
        
        bars = ax_aoi.bar(config_names, channel_aois, alpha=0.7, color=colors)
        ax_aoi.set_title(f'Channel {ch_id} Avg AoI', fontsize=12, fontweight='bold')
        ax_aoi.set_ylabel('AoI (μs)', fontsize=10)
        ax_aoi.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, channel_aois):
            ax_aoi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Performance degradation plot
    ax_degradation = axes[0, 2]
    baseline_throughput = [results[config_names[0]]['stats']]
    
    # Calculate throughput degradation compared to no OBSS
    baseline_stats = results[config_names[0]]['stats']
    baseline_total = sum(sta_stats['successful_transmissions'] for sta_stats in baseline_stats['stations'].values())
    baseline_throughput = (baseline_total * 33) / baseline_stats['total_slots']
    
    degradations = []
    for config_name in config_names[1:]:  # Skip baseline
        stats = results[config_name]['stats']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        throughput = (total_successful * 33) / stats['total_slots']
        degradation = ((baseline_throughput - throughput) / baseline_throughput) * 100
        degradations.append(degradation)
    
    bars = ax_degradation.bar(config_names[1:], degradations, alpha=0.7, color=colors[1:])
    ax_degradation.set_title('Throughput Degradation\nvs No OBSS', fontsize=12, fontweight='bold')
    ax_degradation.set_ylabel('Degradation (%)', fontsize=10)
    ax_degradation.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, degradations):
        ax_degradation.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # AoI increase plot
    ax_aoi_increase = axes[1, 2]
    baseline_aois = [sta_stats['average_aoi_time_us'] for sta_stats in baseline_stats['stations'].values()]
    baseline_avg_aoi = np.mean(baseline_aois)
    
    aoi_increases = []
    for config_name in config_names[1:]:  # Skip baseline
        stats = results[config_name]['stats']
        all_aois = [sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()]
        avg_aoi = np.mean(all_aois)
        increase = ((avg_aoi - baseline_avg_aoi) / baseline_avg_aoi) * 100
        aoi_increases.append(increase)
    
    bars = ax_aoi_increase.bar(config_names[1:], aoi_increases, alpha=0.7, color=colors[1:])
    ax_aoi_increase.set_title('AoI Increase\nvs No OBSS', fontsize=12, fontweight='bold')
    ax_aoi_increase.set_ylabel('Increase (%)', fontsize=10)
    ax_aoi_increase.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, aoi_increases):
        ax_aoi_increase.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('obss_channel_impact.png', dpi=300, bbox_inches='tight')

def print_obss_results(results):
    """Print detailed OBSS comparison results"""
    
    print("\n" + "="*80)
    print("OBSS (Overlapping BSS) IMPACT ANALYSIS")
    print("="*80)
    
    for config_name, result in results.items():
        stats = result['stats']
        config = result['config']
        
        print(f"\n📊 {config_name}")
        print("-" * 50)
        print(f"OBSS Enabled: {stats['obss_enabled']}")
        if stats['obss_enabled']:
            print(f"OBSS Threshold: {stats['obss_threshold']:.1%}")
            print(f"OBSS Interference Events: {stats['obss_interference_count']}")
        
        print(f"Simulation time: {stats['total_time_us']/1000:.1f} ms ({stats['total_slots']} slots)")
        
        # Calculate system metrics
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        total_attempts = sum(sta_stats['total_attempts'] for sta_stats in stats['stations'].values())
        total_collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * 33) / stats['total_slots']
        system_success_rate = total_successful / total_attempts if total_attempts > 0 else 0
        avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        print(f"System throughput: {system_throughput:.4f} ({system_throughput*100:.2f}%)")
        print(f"System success rate: {system_success_rate:.4f} ({system_success_rate*100:.2f}%)")
        print(f"Average AoI: {avg_aoi:.1f} μs")
        print(f"Total successful transmissions: {total_successful}")
        print(f"Total collisions: {total_collisions}")
        
        # Per-channel breakdown
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                          if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                ch_throughput = sum((sta_stats['successful_transmissions'] * 33) / stats['total_slots'] 
                                  for _, sta_stats in channel_stas)
                ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                ch_collisions = sum(sta_stats['collisions'] for _, sta_stats in channel_stas)
                
                print(f"  Channel {ch_id}: Throughput={ch_throughput:.4f}, "
                      f"Avg AoI={ch_aoi:.1f}μs, Collisions={ch_collisions}")
    
    # Calculate performance impact
    print(f"\n📈 OBSS PERFORMANCE IMPACT")
    print("-" * 50)
    
    baseline_name = list(results.keys())[0]  # Assuming first is baseline (No OBSS)
    baseline_stats = results[baseline_name]['stats']
    baseline_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in baseline_stats['stations'].values()) * 33 / baseline_stats['total_slots']
    baseline_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in baseline_stats['stations'].values()])
    
    for config_name, result in list(results.items())[1:]:  # Skip baseline
        stats = result['stats']
        config = result['config']
        
        current_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values()) * 33 / stats['total_slots']
        current_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        throughput_degradation = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
        aoi_increase = ((current_aoi - baseline_aoi) / baseline_aoi) * 100
        
        print(f"{config_name}:")
        print(f"  Throughput degradation: {throughput_degradation:.1f}%")
        print(f"  AoI increase: {aoi_increase:.1f}%")
        print(f"  OBSS events: {stats['obss_interference_count']}")

def plot_obss_time_series(results):
    """Plot time series showing OBSS interference events"""
    
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (config_name, result) in enumerate(results.items()):
        df = result['dataframe']
        config = result['config']
        
        if not config['obss_enabled']:
            continue
            
        ax = axes[idx]
        
        # Plot channel busy status and OBSS interference
        slots = df['slot'].values
        
        # Sample every 100 slots for readability
        sample_rate = 100
        sample_slots = slots[::sample_rate]
        
        for ch_id in range(config['num_channels']):
            ch_busy = df[f'channel_{ch_id}_busy'].values[::sample_rate]
            obss_interference = df[f'channel_{ch_id}_obss_interference'].values[::sample_rate]
            
            # Plot channel busy periods
            ax.plot(sample_slots, ch_busy + ch_id * 2.5, 
                   label=f'Ch{ch_id} Busy', linewidth=1, alpha=0.7)
            
            # Plot OBSS interference events
            obss_slots = sample_slots[obss_interference]
            obss_levels = np.ones(len(obss_slots)) * (ch_id * 2.5 + 0.5)
            ax.scatter(obss_slots, obss_levels, 
                      color='red', marker='x', s=20, alpha=0.8,
                      label=f'Ch{ch_id} OBSS' if ch_id == 0 else "")
        
        ax.set_title(f'{config_name} - Channel Activity & OBSS Events', fontsize=12)
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Channel Activity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, config['num_channels'] * 2.5)
    
    plt.tight_layout()
    plt.savefig('obss_time_series.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Run OBSS comparison simulations
    print("Starting OBSS comparison simulations...")
    results = run_obss_comparison()
    
    # Generate visualizations
    print("\nGenerating OBSS comparison plots...")
    plot_obss_comparison(results)
    
    print("Generating channel-specific OBSS impact plots...")
    plot_channel_specific_obss_impact(results)
    
    print("Generating OBSS time series plots...")
    plot_obss_time_series(results)
    
    # Print detailed results
    print_obss_results(results)
    
    print("\n✅ OBSS Analysis complete!")
    print("📁 Plots saved as:")
    print("   - obss_comparison.png")
    print("   - obss_channel_impact.png") 
    print("   - obss_time_series.png")