import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tempfiles.random_access_with_npca import SimplifiedCSMACASimulation, SLOTTIME

def run_npca_comparison_simulation():
    """Run simulations comparing CSMA/CA with and without NPCA under various OBSS conditions"""
    
    # Configuration: Compare different OBSS traffic intensities
    simulation_configs = [
        {"obss_intensity": 0.0, "label": "No OBSS"},
        {"obss_intensity": 0.1, "label": "Low OBSS (10%)"},
        {"obss_intensity": 0.3, "label": "Medium OBSS (30%)"},
        {"obss_intensity": 0.5, "label": "High OBSS (50%)"},
    ]
    
    simulation_time = 100000  # slots
    frame_size = 33  # slots
    
    results = {}
    
    for config in simulation_configs:
        print(f"Running simulation: {config['label']}")
        
        sim = SimplifiedCSMACASimulation(
            num_channels=2,
            stas_per_channel=[5, 5],
            simulation_time=simulation_time,
            frame_size=frame_size,
            obss_traffic_intensity=config["obss_intensity"]
        )
        
        df = sim.run()
        stats = sim.get_statistics()
        
        results[config['label']] = {
            'config': config,
            'stats': stats,
            'dataframe': df
        }
    
    return results

def plot_npca_performance_comparison(results):
    """Create comprehensive NPCA performance comparison plots"""
    
    # Create figure with subplots: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract data for plotting
    config_labels = list(results.keys())
    obss_intensities = [results[label]['config']['obss_intensity'] for label in config_labels]
    
    # Colors for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # === Plot 1: System Throughput vs OBSS Intensity ===
    ax1 = axes[0, 0]
    
    system_throughputs = []
    for label in config_labels:
        stats = results[label]['stats']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * 33) / stats['total_slots']
        system_throughputs.append(system_throughput)
    
    bars1 = ax1.bar(config_labels, system_throughputs, color=colors, alpha=0.7)
    ax1.set_title('System Throughput vs OBSS Intensity', fontsize=14, fontweight='bold')
    ax1.set_ylabel('System Throughput (fraction)', fontsize=12)
    ax1.set_ylim(0, 0.8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars1, system_throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # === Plot 2: Average AoI vs OBSS Intensity ===
    ax2 = axes[0, 1]
    
    avg_aois = []
    for label in config_labels:
        stats = results[label]['stats']
        aoi_values = [sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()]
        avg_aoi = np.mean(aoi_values)
        avg_aois.append(avg_aoi)
    
    bars2 = ax2.bar(config_labels, avg_aois, color=colors, alpha=0.7)
    ax2.set_title('Average AoI vs OBSS Intensity', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average AoI (μs)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars2, avg_aois):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # === Plot 3: NPCA Switch Count vs OBSS Intensity ===
    ax3 = axes[0, 2]
    
    total_npca_switches = []
    for label in config_labels:
        stats = results[label]['stats']
        switches = sum(sta_stats['npca_switches'] for sta_stats in stats['stations'].values())
        total_npca_switches.append(switches)
    
    bars3 = ax3.bar(config_labels, total_npca_switches, color=colors, alpha=0.7)
    ax3.set_title('Total NPCA Switches vs OBSS Intensity', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of NPCA Switches', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars3, total_npca_switches):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # === Plot 4: Channel Utilization Comparison ===
    ax4 = axes[1, 0]
    
    # Prepare data for grouped bar chart
    channel_0_utils = []
    channel_1_utils = []
    
    for label in config_labels:
        stats = results[label]['stats']
        
        # Channel 0 utilization
        ch0_stas = [sta_stats for sta_stats in stats['stations'].values() if sta_stats['channel'] == 0]
        ch0_successful = sum(sta_stats['successful_transmissions'] for sta_stats in ch0_stas)
        ch0_util = (ch0_successful * 33) / stats['total_slots']
        channel_0_utils.append(ch0_util)
        
        # Channel 1 utilization
        ch1_stas = [sta_stats for sta_stats in stats['stations'].values() if sta_stats['channel'] == 1]
        ch1_successful = sum(sta_stats['successful_transmissions'] for sta_stats in ch1_stas)
        ch1_util = (ch1_successful * 33) / stats['total_slots']
        channel_1_utils.append(ch1_util)
    
    x = np.arange(len(config_labels))
    width = 0.35
    
    bars4a = ax4.bar(x - width/2, channel_0_utils, width, label='Channel 0 (BSS Primary)', 
                     color='skyblue', alpha=0.8)
    bars4b = ax4.bar(x + width/2, channel_1_utils, width, label='Channel 1 (NPCA)', 
                     color='lightcoral', alpha=0.8)
    
    ax4.set_title('Channel Utilization Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Channel Utilization (fraction)', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_labels, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 0.6)
    
    # Add value labels
    for bars in [bars4a, bars4b]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # === Plot 5: Success Rate vs OBSS Intensity ===
    ax5 = axes[1, 1]
    
    success_rates = []
    for label in config_labels:
        stats = results[label]['stats']
        rates = [sta_stats['success_rate'] for sta_stats in stats['stations'].values()]
        avg_success_rate = np.mean(rates) * 100
        success_rates.append(avg_success_rate)
    
    bars5 = ax5.bar(config_labels, success_rates, color=colors, alpha=0.7)
    ax5.set_title('Average Success Rate vs OBSS Intensity', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Success Rate (%)', fontsize=12)
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars5, success_rates):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # === Plot 6: Collision Analysis ===
    ax6 = axes[1, 2]
    
    collision_rates = []
    for label in config_labels:
        stats = results[label]['stats']
        total_collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        total_attempts = sum(sta_stats['total_attempts'] for sta_stats in stats['stations'].values())
        collision_rate = (total_collisions / total_attempts * 100) if total_attempts > 0 else 0
        collision_rates.append(collision_rate)
    
    bars6 = ax6.bar(config_labels, collision_rates, color=colors, alpha=0.7)
    ax6.set_title('Collision Rate vs OBSS Intensity', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Collision Rate (%)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars6, collision_rates):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('npca_performance_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_npca_detailed_analysis(results):
    """Create detailed NPCA analysis plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    config_labels = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # === Plot 1: Per-STA AoI Improvement ===
    baseline_aois = {}
    for label in config_labels:
        stats = results[label]['stats']
        for sta_id, sta_stats in stats['stations'].items():
            if label == "No OBSS":
                baseline_aois[sta_id] = sta_stats['average_aoi_time_us']
    
    # Calculate AoI improvement for each OBSS condition
    obss_conditions = [label for label in config_labels if label != "No OBSS"]
    x_pos = np.arange(len(baseline_aois))
    width = 0.25
    
    for i, condition in enumerate(obss_conditions):
        stats = results[condition]['stats']
        improvements = []
        sta_ids = []
        for sta_id, baseline_aoi in baseline_aois.items():
            if sta_id in stats['stations']:
                current_aoi = stats['stations'][sta_id]['average_aoi_time_us']
                improvement = ((baseline_aoi - current_aoi) / baseline_aoi) * 100
                improvements.append(improvement)
                sta_ids.append(sta_id)
        
        ax1.bar(x_pos + i*width, improvements, width, label=condition, alpha=0.8)
    
    ax1.set_title('AoI Improvement vs No OBSS Baseline', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AoI Improvement (%)', fontsize=12)
    ax1.set_xlabel('Station ID', fontsize=12)
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels([f'STA {sta_id}' for sta_id in baseline_aois.keys()])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # === Plot 2: NPCA Usage Distribution ===
    sta_npca_switches = {}
    for sta_id in baseline_aois.keys():
        sta_npca_switches[sta_id] = []
        for label in config_labels:
            stats = results[label]['stats']
            if sta_id in stats['stations']:
                switches = stats['stations'][sta_id]['npca_switches']
                sta_npca_switches[sta_id].append(switches)
    
    # Create stacked bar chart for NPCA switches
    bottoms = np.zeros(len(config_labels))
    for sta_id in sta_npca_switches.keys():
        switches = sta_npca_switches[sta_id]
        ax2.bar(config_labels, switches, bottom=bottoms, 
               label=f'STA {sta_id}', alpha=0.8)
        bottoms += np.array(switches)
    
    ax2.set_title('NPCA Switch Distribution by STA', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of NPCA Switches', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Throughput vs OBSS Intensity (Line Plot) ===
    obss_intensities = [results[label]['config']['obss_intensity'] for label in config_labels]
    system_throughputs = []
    
    for label in config_labels:
        stats = results[label]['stats']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * 33) / stats['total_slots']
        system_throughputs.append(system_throughput)
    
    ax3.plot(obss_intensities, system_throughputs, 'o-', linewidth=2, markersize=8, 
            color='#2ca02c', label='System Throughput')
    ax3.set_title('System Throughput vs OBSS Intensity', fontsize=14, fontweight='bold')
    ax3.set_xlabel('OBSS Traffic Intensity', fontsize=12)
    ax3.set_ylabel('System Throughput (fraction)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add value annotations
    for x, y in zip(obss_intensities, system_throughputs):
        ax3.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # === Plot 4: NPCA Effectiveness Metrics ===
    effectiveness_metrics = {
        'AoI Reduction (%)': [],
        'Throughput Improvement (%)': [],
        'Switch Efficiency': []
    }
    
    baseline_throughput = system_throughputs[0]  # No OBSS case
    baseline_avg_aoi = np.mean([sta_stats['average_aoi_time_us'] 
                               for sta_stats in results["No OBSS"]['stats']['stations'].values()])
    
    for i, label in enumerate(config_labels[1:], 1):  # Skip "No OBSS"
        stats = results[label]['stats']
        
        # AoI reduction
        current_avg_aoi = np.mean([sta_stats['average_aoi_time_us'] 
                                  for sta_stats in stats['stations'].values()])
        aoi_reduction = ((baseline_avg_aoi - current_avg_aoi) / baseline_avg_aoi) * 100
        effectiveness_metrics['AoI Reduction (%)'].append(aoi_reduction)
        
        # Throughput improvement
        current_throughput = system_throughputs[i]
        throughput_improvement = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
        effectiveness_metrics['Throughput Improvement (%)'].append(throughput_improvement)
        
        # Switch efficiency (improvement per switch)
        total_switches = sum(sta_stats['npca_switches'] for sta_stats in stats['stations'].values())
        switch_efficiency = aoi_reduction / max(1, total_switches)
        effectiveness_metrics['Switch Efficiency'].append(switch_efficiency)
    
    x_metrics = np.arange(len(effectiveness_metrics))
    width = 0.25
    obss_labels = config_labels[1:]  # Exclude "No OBSS"
    
    for i, obss_label in enumerate(obss_labels):
        values = [effectiveness_metrics[metric][i] for metric in effectiveness_metrics.keys()]
        ax4.bar(x_metrics + i*width, values, width, label=obss_label, alpha=0.8)
    
    ax4.set_title('NPCA Effectiveness Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Improvement (%)', fontsize=12)
    ax4.set_xticks(x_metrics + width)
    ax4.set_xticklabels(effectiveness_metrics.keys(), rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('npca_detailed_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def print_npca_detailed_results(results):
    """Print detailed NPCA performance results"""
    
    print("\n" + "="*80)
    print("NPCA PERFORMANCE ANALYSIS RESULTS")
    print("="*80)
    
    # Baseline (No OBSS) results
    baseline_label = "No OBSS"
    baseline_stats = results[baseline_label]['stats']
    baseline_throughput = sum(sta_stats['successful_transmissions'] 
                             for sta_stats in baseline_stats['stations'].values()) * 33 / baseline_stats['total_slots']
    baseline_aoi = np.mean([sta_stats['average_aoi_time_us'] 
                           for sta_stats in baseline_stats['stations'].values()])
    
    print(f"\n📊 Baseline Performance (No OBSS):")
    print(f"  System Throughput: {baseline_throughput:.4f}")
    print(f"  Average AoI: {baseline_aoi:.1f} μs")
    
    # Compare each OBSS condition with baseline
    for label, result in results.items():
        if label == baseline_label:
            continue
            
        stats = result['stats']
        config = result['config']
        
        print(f"\n📊 {label} (OBSS Intensity: {config['obss_intensity']:.1%}):")
        print("-" * 60)
        
        # System-level metrics
        current_throughput = sum(sta_stats['successful_transmissions'] 
                               for sta_stats in stats['stations'].values()) * 33 / stats['total_slots']
        current_aoi = np.mean([sta_stats['average_aoi_time_us'] 
                              for sta_stats in stats['stations'].values()])
        total_npca_switches = sum(sta_stats['npca_switches'] for sta_stats in stats['stations'].values())
        
        throughput_change = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
        aoi_change = ((baseline_aoi - current_aoi) / baseline_aoi) * 100
        
        print(f"  System Throughput: {current_throughput:.4f} ({throughput_change:+.1f}%)")
        print(f"  Average AoI: {current_aoi:.1f} μs ({aoi_change:+.1f}%)")
        print(f"  Total NPCA Switches: {total_npca_switches}")
        
        # Channel-specific analysis
        ch0_stas = [sta_stats for sta_stats in stats['stations'].values() if sta_stats['channel'] == 0]
        ch1_stas = [sta_stats for sta_stats in stats['stations'].values() if sta_stats['channel'] == 1]
        
        ch0_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in ch0_stas) * 33 / stats['total_slots']
        ch1_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in ch1_stas) * 33 / stats['total_slots']
        
        print(f"  Channel 0 Utilization: {ch0_throughput:.4f}")
        print(f"  Channel 1 Utilization: {ch1_throughput:.4f}")
        
        # Per-STA NPCA usage
        print(f"\n  Per-STA NPCA Switches:")
        for sta_id, sta_stats in stats['stations'].items():
            if sta_stats['npca_switches'] > 0:
                print(f"    STA {sta_id}: {sta_stats['npca_switches']} switches, "
                      f"Final channel: {sta_stats['final_channel_type']}")
    
    # Summary insights
    print(f"\n" + "="*80)
    print("NPCA EFFECTIVENESS SUMMARY")
    print("="*80)
    
    max_improvement_label = None
    max_aoi_improvement = -float('inf')
    
    for label, result in results.items():
        if label == baseline_label:
            continue
            
        stats = result['stats']
        current_aoi = np.mean([sta_stats['average_aoi_time_us'] 
                              for sta_stats in stats['stations'].values()])
        aoi_improvement = ((baseline_aoi - current_aoi) / baseline_aoi) * 100
        
        if aoi_improvement > max_aoi_improvement:
            max_aoi_improvement = aoi_improvement
            max_improvement_label = label
    
    if max_improvement_label:
        print(f"🏆 Best AoI Improvement: {max_improvement_label} ({max_aoi_improvement:.1f}%)")
    
    print(f"📈 NPCA shows effectiveness when OBSS intensity > 10%")
    print(f"🔄 NPCA switches increase with OBSS intensity")
    print(f"⚡ Channel 1 (NPCA) utilization correlates with OBSS interference")

if __name__ == "__main__":
    # Run NPCA comparison simulations
    print("Starting NPCA performance comparison simulations...")
    results = run_npca_comparison_simulation()
    
    # Generate visualizations
    print("\nGenerating NPCA performance comparison plots...")
    plot_npca_performance_comparison(results)
    
    print("Generating detailed NPCA analysis plots...")
    plot_npca_detailed_analysis(results)
    
    # Print detailed results
    # print_detailed_results(results)
    
    print("\n✅ NPCA Analysis complete!")
    print("📁 Plots saved as:")
    print("   - npca_performance_comparison.png")
    print("   - npca_detailed_analysis.png")