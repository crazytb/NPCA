import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random_access.random_access_rev import SimplifiedCSMACASimulation, SLOTTIME

simulation_time = 50000  # slots
frame_size = 33  # slots
    
def run_obss_comparison():
    """Run simulations comparing OBSS enabled vs disabled"""
    
    # Configuration
    simulation_configs = [
        {"num_channels": 2, "stas_per_channel": [2, 5], "obss_enabled": False, 
         "obss_generation_rate": 0.0, "label": "No OBSS"},
        {"num_channels": 2, "stas_per_channel": [2, 5], "obss_enabled": True, 
         "obss_generation_rate": 0.0005, "label": "Low OBSS (0.05%)"},
        {"num_channels": 2, "stas_per_channel": [2, 5], "obss_enabled": True, 
         "obss_generation_rate": 0.002, "label": "High OBSS (0.2%)"},
    ]
    

    
    results = {}
    
    for config in simulation_configs:
        print(f"Running simulation: {config['label']}")
        
        sim = SimplifiedCSMACASimulation(
            num_channels=config["num_channels"],
            stas_per_channel=config["stas_per_channel"],
            simulation_time=simulation_time,
            frame_size=frame_size,
            obss_enabled=config["obss_enabled"],
            obss_generation_rate=config["obss_generation_rate"]
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
        system_throughput = (total_successful * frame_size) / stats['total_slots']
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
    
    # Plot 4: OBSS Traffic Generation and Duration
    ax4 = axes[1, 1]
    obss_events = []
    obss_durations = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        obss_count = stats.get('obss_events_generated', 0)
        obss_duration = stats.get('obss_total_duration_slots', 0)
        obss_events.append(obss_count)
        obss_durations.append(obss_duration)
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars4a = ax4.bar(x - width/2, obss_events, width, alpha=0.7, color='lightblue', label='OBSS Events')
    bars4b = ax4.bar(x + width/2, [d/100 for d in obss_durations], width, alpha=0.7, color='lightcoral', label='OBSS Duration (×100 slots)')
    
    ax4.set_title('OBSS Traffic Generation', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Count / Duration', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4a, obss_events):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar, value in zip(bars4b, obss_durations):
        ax4.text(bar.get_x() + bar.get_width()/2, (value/100) + 5,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
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
            channel_throughput = (total_successful * frame_size) / stats['total_slots']
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
    baseline_stats = results[config_names[0]]['stats']
    baseline_total = sum(sta_stats['successful_transmissions'] for sta_stats in baseline_stats['stations'].values())
    baseline_throughput = (baseline_total * frame_size) / baseline_stats['total_slots']
    
    degradations = []
    for config_name in config_names[1:]:  # Skip baseline
        stats = results[config_name]['stats']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        throughput = (total_successful * frame_size) / stats['total_slots']
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

def plot_fsm_states_analysis(results):
    """Plot FSM states analysis including OBSS_FROZEN"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = ['skyblue', 'orange', 'lightcoral']
    
    # State distribution for each configuration
    state_types = ['idle', 'backoff', 'backoff_frozen', 'obss_frozen', 'transmitting']
    
    for idx, (config_name, result) in enumerate(results.items()):
        df = result['dataframe']
        config = result['config']
        
        ax = axes[idx // 2, idx % 2]
        
        # Count state occurrences for all STAs
        state_counts = {state: 0 for state in state_types}
        
        for ch_id in range(config['num_channels']):
            states_col = f'states_ch_{ch_id}'
            if states_col in df.columns:
                for slot_states in df[states_col]:
                    for state in slot_states:
                        if state in state_counts:
                            state_counts[state] += 1
        
        # Create pie chart
        values = list(state_counts.values())
        labels = [f'{state}\n({count:,})' for state, count in state_counts.items() if count > 0]
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        if sum(values) > 0:
            ax.pie([v for v in values if v > 0], labels=labels, autopct='%1.1f%%', 
                   colors=colors_pie, startangle=90)
        
        ax.set_title(f'{config_name}\nFSM State Distribution', fontsize=12, fontweight='bold')
    
    # Remove empty subplot if needed
    if len(results) < 4:
        for i in range(len(results), 4):
            axes[i // 2, i % 2].remove()
    
    plt.tight_layout()
    plt.savefig('obss_fsm_states.png', dpi=300, bbox_inches='tight')

def plot_obss_deferrals_analysis(results):
    """Plot analysis of OBSS deferrals impact"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = ['skyblue', 'orange', 'lightcoral']
    
    # Plot 1: Total OBSS Deferrals per Configuration
    ax1 = axes[0, 0]
    
    total_deferrals = []
    for config_name in config_names:
        stats = results[config_name]['stats']
        deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        total_deferrals.append(deferrals)
    
    bars1 = ax1.bar(config_names, total_deferrals, alpha=0.7, color=colors)
    ax1.set_title('Total OBSS Deferrals', fontsize=12, fontweight='bold')
    ax1.set_ylabel('OBSS Deferrals Count')
    ax1.grid(True, alpha=0.3)
    
    for bar, value in zip(bars1, total_deferrals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: OBSS Deferrals vs Throughput
    ax2 = axes[0, 1]
    
    throughputs = []
    deferrals_list = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        throughput = (total_successful * frame_size) / stats['total_slots']
        deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        
        throughputs.append(throughput)
        deferrals_list.append(deferrals)
    
    ax2.scatter(deferrals_list, throughputs, s=100, alpha=0.7, c=colors[:len(config_names)])
    for i, config_name in enumerate(config_names):
        ax2.annotate(config_name, (deferrals_list[i], throughputs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_title('OBSS Deferrals vs Throughput', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total OBSS Deferrals')
    ax2.set_ylabel('System Throughput')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Channel Utilization Breakdown
    ax3 = axes[1, 0]
    
    width = 0.25
    x = np.arange(len(config_names))
    
    intra_utils = []
    obss_utils = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        # Calculate intra-BSS utilization
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        intra_util = (total_successful * frame_size) / (stats['total_slots'] * 2)  # 2 channels
        intra_utils.append(intra_util)
        
        # Calculate OBSS utilization
        obss_util = stats.get('obss_channel_utilization', 0)
        obss_utils.append(obss_util)
    
    bars1 = ax3.bar(x - width/2, intra_utils, width, label='Intra-BSS Utilization', alpha=0.7, color='lightblue')
    bars2 = ax3.bar(x + width/2, obss_utils, width, label='OBSS Utilization', alpha=0.7, color='lightcoral')
    
    ax3.set_title('Channel Utilization Breakdown', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Channel Utilization')
    ax3.set_xlabel('Configuration')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Per-STA OBSS Impact
    ax4 = axes[1, 1]
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        if not config['obss_enabled']:
            continue
        
        # Get AoI and OBSS deferrals for each STA
        for sta_id, sta_stats in stats['stations'].items():
            aoi = sta_stats['average_aoi_time_us']
            deferrals = sta_stats['obss_deferrals']
            channel = sta_stats['channel']
            
            marker = 'o' if channel == 0 else 's'
            color = colors[list(results.keys()).index(config_name)]
            label_added = False
            
            ax4.scatter(deferrals, aoi, marker=marker, alpha=0.7, color=color, s=60,
                       label=f'{config_name} Ch{channel}' if not label_added else "")
            label_added = True
    
    ax4.set_title('Per-STA: AoI vs OBSS Deferrals', fontsize=12, fontweight='bold')
    ax4.set_xlabel('OBSS Deferrals Count')
    ax4.set_ylabel('Average AoI (μs)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('obss_deferrals_analysis.png', dpi=300, bbox_inches='tight')

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
            print(f"OBSS Generation Rate: {stats['obss_generation_rate']:.1%} per slot")
            print(f"OBSS Events Generated: {stats['obss_events_generated']}")
            print(f"OBSS Total Duration: {stats['obss_total_duration_slots']} slots ({stats['obss_total_duration_us']/1000:.1f} ms)")
            print(f"OBSS Channel Utilization: {stats['obss_channel_utilization']:.1%}")
        
        print(f"Simulation time: {stats['total_time_us']/1000:.1f} ms ({stats['total_slots']} slots)")
        
        # Calculate system metrics
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        total_attempts = sum(sta_stats['total_attempts'] for sta_stats in stats['stations'].values())
        total_collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        total_obss_deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * frame_size) / stats['total_slots']
        system_success_rate = total_successful / total_attempts if total_attempts > 0 else 0
        avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        print(f"System throughput: {system_throughput:.4f} ({system_throughput*100:.2f}%)")
        print(f"System success rate: {system_success_rate:.4f} ({system_success_rate*100:.2f}%)")
        print(f"Average AoI: {avg_aoi:.1f} μs")
        print(f"Total successful transmissions: {total_successful}")
        print(f"Total collisions: {total_collisions}")
        print(f"Total OBSS deferrals: {total_obss_deferrals}")
        
        # Per-channel breakdown
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                          if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                ch_throughput = sum((sta_stats['successful_transmissions'] * frame_size) / stats['total_slots'] 
                                  for _, sta_stats in channel_stas)
                ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                ch_collisions = sum(sta_stats['collisions'] for _, sta_stats in channel_stas)
                ch_obss_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
                
                print(f"  Channel {ch_id}: Throughput={ch_throughput:.4f}, "
                      f"Avg AoI={ch_aoi:.1f}μs, Collisions={ch_collisions}, OBSS Deferrals={ch_obss_deferrals}")
    
    # Calculate performance impact
    print(f"\n📈 OBSS PERFORMANCE IMPACT")
    print("-" * 50)
    
    baseline_name = list(results.keys())[0]  # Assuming first is baseline (No OBSS)
    baseline_stats = results[baseline_name]['stats']
    baseline_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in baseline_stats['stations'].values()) * frame_size / baseline_stats['total_slots']
    baseline_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in baseline_stats['stations'].values()])
    
    for config_name, result in list(results.items())[1:]:  # Skip baseline
        stats = result['stats']
        
        current_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values()) * frame_size / stats['total_slots']
        current_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        throughput_degradation = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
        aoi_increase = ((current_aoi - baseline_aoi) / baseline_aoi) * 100
        
        print(f"{config_name}:")
        print(f"  Throughput degradation: {throughput_degradation:.1f}%")
        print(f"  AoI increase: {aoi_increase:.1f}%")
        print(f"  OBSS events generated: {stats['obss_events_generated']}")
        print(f"  Total OBSS deferrals: {sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())}")

if __name__ == "__main__":
    # Run OBSS comparison simulations
    print("Starting OBSS comparison simulations...")
    results = run_obss_comparison()
    
    # Generate visualizations
    print("\nGenerating OBSS comparison plots...")
    plot_obss_comparison(results)
    
    print("Generating channel-specific OBSS impact plots...")
    plot_channel_specific_obss_impact(results)
    
    print("Generating FSM states analysis...")
    plot_fsm_states_analysis(results)
    
    print("Generating OBSS deferrals analysis...")
    plot_obss_deferrals_analysis(results)
    
    # Print detailed results
    print_obss_results(results)
    
    print("\n✅ OBSS Analysis complete!")
    print("📁 Plots saved as:")
    print("   - obss_comparison.png")
    print("   - obss_channel_impact.png")
    print("   - obss_fsm_states.png")
    print("   - obss_deferrals_analysis.png")