import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random_access.random_access import SimplifiedCSMACASimulation, SLOTTIME
from random_access.configs import *

def run_obss_comparison(simulation_configs):
    """Run simulations comparing OBSS enabled vs disabled with channel-specific settings"""
    
    results = {}
    
    for config in simulation_configs:
        print(f"Running simulation: {config['label']}")
        
        sim = SimplifiedCSMACASimulation(
            num_channels=config["num_channels"],
            stas_per_channel=config["stas_per_channel"],
            simulation_time=config["simulation_time"],
            frame_size=config["frame_size"],
            obss_enabled_per_channel=config["obss_enabled_per_channel"],
            npca_enabled=config.get("npca_enabled", None),  # 새로 추가 - 없으면 None
            obss_generation_rate=config["obss_generation_rate"]
        )
        
        df = sim.run()
        df.to_csv(f"obss_simulation_{config['label'].replace(' ', '_').lower()}.csv", index=False)
        stats = sim.get_statistics()
        
        results[config['label']] = {
            'config': config,
            'stats': stats,
            'dataframe': df
        }
    
    return results

def plot_obss_comparison(results):
    """Create OBSS comparison plots with channel-specific analysis"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # Plot 1: System Throughput Comparison
    ax1 = axes[0, 0]
    system_throughputs = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * config['frame_size']) / stats['total_slots']
        system_throughputs.append(system_throughput)
    
    bars1 = ax1.bar(range(len(config_names)), system_throughputs, alpha=0.7, color=colors)
    ax1.set_title('System Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Throughput (fraction)', fontsize=12)
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(system_throughputs) * 1.2)
    
    # Add value labels and OBSS status
    for i, (bar, value, config_name) in enumerate(zip(bars1, system_throughputs, config_names)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add OBSS status indicator
        obss_config = results[config_name]['stats']['obss_enabled_per_channel']
        obss_status = f"Ch{[i for i, enabled in enumerate(obss_config) if enabled]}"
        if not any(obss_config):
            obss_status = "No OBSS"
        ax1.text(bar.get_x() + bar.get_width()/2, -0.01,
                obss_status, ha='center', va='top', fontsize=8, style='italic')
    
    # Plot 2: Average AoI Comparison
    ax2 = axes[0, 1]
    avg_aois = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        all_aois = [sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()]
        avg_aoi = np.mean(all_aois)
        avg_aois.append(avg_aoi)
    
    bars2 = ax2.bar(range(len(config_names)), avg_aois, alpha=0.7, color=colors)
    ax2.set_title('Average AoI Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average AoI (μs)', fontsize=12)
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
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
    
    bars3 = ax3.bar(range(len(config_names)), total_collisions, alpha=0.7, color=colors)
    ax3.set_title('Total Collisions Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Collisions', fontsize=12)
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, total_collisions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Channel-specific OBSS Traffic Generation
    ax4 = axes[1, 1]
    
    # Prepare data for stacked bar chart showing OBSS events per channel
    channel_obss_data = {}
    max_channels = max(len(results[name]['stats']['obss_enabled_per_channel']) for name in config_names)
    
    for ch_id in range(max_channels):
        channel_obss_data[f'Ch{ch_id}'] = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            if ch_id < len(stats['obss_enabled_per_channel']):
                ch_obss_stats = stats['obss_per_channel'][ch_id]
                channel_obss_data[f'Ch{ch_id}'].append(ch_obss_stats['generated'])
            else:
                channel_obss_data[f'Ch{ch_id}'].append(0)
    
    # Create stacked bar chart
    x = np.arange(len(config_names))
    width = 0.6
    bottom = np.zeros(len(config_names))
    
    colors_ch = plt.cm.tab10(np.linspace(0, 1, max_channels))
    
    for ch_id, color in enumerate(colors_ch):
        values = channel_obss_data[f'Ch{ch_id}']
        bars = ax4.bar(x, values, width, bottom=bottom, alpha=0.7, 
                      color=color, label=f'Channel {ch_id}')
        
        # Add value labels for non-zero values
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        bottom[i] + value/2,
                        f'{int(value)}', ha='center', va='center', 
                        fontweight='bold', fontsize=9)
        
        bottom += values
    
    ax4.set_title('OBSS Events by Channel', fontsize=14, fontweight='bold')
    ax4.set_ylabel('OBSS Events Generated', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure/obss_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_channel_specific_obss_impact(results):
    """Plot OBSS impact on individual channels with channel-specific analysis"""
    
    config_names = list(results.keys())
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    
    fig, axes = plt.subplots(num_channels, 3, figsize=(16, 6*num_channels))
    if num_channels == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # For each channel, plot throughput, AoI, and OBSS impact
    for ch_id in range(num_channels):
        # Channel throughput comparison
        ax_throughput = axes[ch_id, 0]
        channel_throughputs = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            config = results[config_name]['config']
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            total_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
            channel_throughput = (total_successful * config['frame_size']) / stats['total_slots']
            channel_throughputs.append(channel_throughput)
        
        bars = ax_throughput.bar(range(len(config_names)), channel_throughputs, alpha=0.7, color=colors)
        ax_throughput.set_title(f'Channel {ch_id} Throughput', fontsize=12, fontweight='bold')
        ax_throughput.set_ylabel('Throughput', fontsize=10)
        ax_throughput.set_xticks(range(len(config_names)))
        ax_throughput.set_xticklabels(config_names, rotation=45, ha='right')
        ax_throughput.grid(True, alpha=0.3)
        
        # Add OBSS status indicators
        for i, (bar, value, config_name) in enumerate(zip(bars, channel_throughputs, config_names)):
            ax_throughput.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                              f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            # OBSS status for this channel
            obss_enabled = results[config_name]['stats']['obss_enabled_per_channel'][ch_id]
            status_color = 'green' if obss_enabled else 'red'
            status_text = '●' if obss_enabled else '○'
            ax_throughput.text(bar.get_x() + bar.get_width()/2, -0.005,
                              status_text, ha='center', va='top', color=status_color, fontsize=12)
        
        # Channel AoI comparison
        ax_aoi = axes[ch_id, 1]
        channel_aois = []
        
        for config_name in config_names:
            # stats = results[config_name]['stats']
            # channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
            #                if sta_stats['channel'] == ch_id]
            if channel_stas:
                flattened_aoi = [v for pair in results[config_name]['dataframe']['aoi_ch_' + str(ch_id)] for v in pair]
                avg_aoi = np.mean(flattened_aoi) if flattened_aoi else 0
                channel_aois.append(avg_aoi)
            else:
                channel_aois.append(0)
        
        bars = ax_aoi.bar(range(len(config_names)), channel_aois, alpha=0.7, color=colors)
        ax_aoi.set_title(f'Channel {ch_id} Avg AoI', fontsize=12, fontweight='bold')
        ax_aoi.set_ylabel('AoI (slots)', fontsize=10)
        ax_aoi.set_xticks(range(len(config_names)))
        ax_aoi.set_xticklabels(config_names, rotation=45, ha='right')
        ax_aoi.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, channel_aois):
            ax_aoi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Channel OBSS impact (deferrals and events)
        ax_obss = axes[ch_id, 2]
        
        obss_events = []
        obss_deferrals = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            
            # OBSS events generated in this channel
            ch_obss_stats = stats['obss_per_channel'][ch_id]
            obss_events.append(ch_obss_stats['generated'])
            
            # OBSS deferrals experienced by STAs in this channel
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            total_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
            obss_deferrals.append(total_deferrals)
        
        x = np.arange(len(config_names))
        width = 0.35
        
        bars1 = ax_obss.bar(x - width/2, obss_events, width, alpha=0.7, 
                           color='lightblue', label='OBSS Events Generated')
        bars2 = ax_obss.bar(x + width/2, obss_deferrals, width, alpha=0.7, 
                           color='lightcoral', label='STA OBSS Deferrals')
        
        ax_obss.set_title(f'Channel {ch_id} OBSS Impact', fontsize=12, fontweight='bold')
        ax_obss.set_ylabel('Count', fontsize=10)
        ax_obss.set_xticks(x)
        ax_obss.set_xticklabels(config_names, rotation=45, ha='right')
        ax_obss.legend()
        ax_obss.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax_obss.text(bar.get_x() + bar.get_width()/2., height + 5,
                                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figure/obss_channel_impact.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_fsm_states_analysis(results):
    """Plot FSM states analysis including OBSS_FROZEN with channel-specific info"""
    
    config_names = list(results.keys())
    num_configs = len(config_names)
    
    # Create subplot grid based on number of configurations
    cols = min(3, num_configs)
    rows = (num_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # State distribution for each configuration
    state_types = ['idle', 'backoff', 'backoff_frozen', 'obss_frozen', 'transmitting']
    
    for idx, (config_name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        df = result['dataframe']
        config = result['config']
        stats = result['stats']
        
        ax = axes[idx]
        
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
        values = [count for count in state_counts.values() if count > 0]
        labels = [f'{state}\n({count:,})' for state, count in state_counts.items() if count > 0]
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        if sum(values) > 0:
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                             colors=colors_pie, startangle=90)
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
        
        # Add OBSS configuration info to title
        obss_config = stats['obss_enabled_per_channel']
        active_channels = [i for i, enabled in enumerate(obss_config) if enabled]
        obss_info = f"OBSS: Ch{active_channels}" if active_channels else "No OBSS"
        
        ax.set_title(f'{config_name}\n{obss_info}\nFSM State Distribution', 
                    fontsize=11, fontweight='bold')
    
    # Remove empty subplots
    for i in range(num_configs, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.savefig('figure/obss_fsm_states.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_obss_deferrals_analysis(results):
    """Plot analysis of OBSS deferrals impact with channel-specific insights"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # Plot 1: Total OBSS Deferrals per Configuration
    ax1 = axes[0, 0]
    
    total_deferrals = []
    for config_name in config_names:
        stats = results[config_name]['stats']
        deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        total_deferrals.append(deferrals)
    
    bars1 = ax1.bar(range(len(config_names)), total_deferrals, alpha=0.7, color=colors)
    ax1.set_title('Total OBSS Deferrals', fontsize=12, fontweight='bold')
    ax1.set_ylabel('OBSS Deferrals Count')
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    for i, (bar, value, config_name) in enumerate(zip(bars1, total_deferrals, config_names)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # Add OBSS channel info
        obss_config = results[config_name]['stats']['obss_enabled_per_channel']
        active_channels = [i for i, enabled in enumerate(obss_config) if enabled]
        obss_info = f"Ch{active_channels}" if active_channels else "None"
        ax1.text(bar.get_x() + bar.get_width()/2, -max(total_deferrals)*0.05,
                obss_info, ha='center', va='top', fontsize=8, style='italic')
    
    # Plot 2: OBSS Deferrals vs Throughput
    ax2 = axes[0, 1]
    
    throughputs = []
    deferrals_list = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        throughput = (total_successful * config['frame_size']) / stats['total_slots']
        deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        
        throughputs.append(throughput)
        deferrals_list.append(deferrals)
    
    scatter = ax2.scatter(deferrals_list, throughputs, s=100, alpha=0.7, c=colors)
    for i, config_name in enumerate(config_names):
        ax2.annotate(config_name, (deferrals_list[i], throughputs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_title('OBSS Deferrals vs Throughput', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total OBSS Deferrals')
    ax2.set_ylabel('System Throughput')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Channel-specific OBSS Impact
    ax3 = axes[1, 0]
    
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    
    # Create grouped bar chart for channel-specific deferrals
    x = np.arange(len(config_names))
    width = 0.8 / num_channels
    
    for ch_id in range(num_channels):
        channel_deferrals = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            ch_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
            channel_deferrals.append(ch_deferrals)
        
        offset = (ch_id - num_channels/2 + 0.5) * width
        bars = ax3.bar(x + offset, channel_deferrals, width, alpha=0.7, 
                      label=f'Channel {ch_id}', color=plt.cm.tab10(ch_id))
        
        # Add value labels for non-zero values
        for bar, value in zip(bars, channel_deferrals):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{int(value)}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_title('OBSS Deferrals by Channel', fontsize=12, fontweight='bold')
    ax3.set_ylabel('OBSS Deferrals Count')
    ax3.set_xlabel('Configuration')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: OBSS Generation vs Channel Status
    ax4 = axes[1, 1]
    
    # Show OBSS events generated per channel for each configuration
    obss_generation_data = []
    channel_labels = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config_data = []
        
        for ch_id in range(num_channels):
            ch_obss_stats = stats['obss_per_channel'][ch_id]
            config_data.append(ch_obss_stats['generated'])
        
        obss_generation_data.append(config_data)
        
        # Create label with OBSS status
        obss_config = stats['obss_enabled_per_channel']
        active_channels = [i for i, enabled in enumerate(obss_config) if enabled]
        label = f"{config_name}\n(OBSS: Ch{active_channels})" if active_channels else f"{config_name}\n(No OBSS)"
        channel_labels.append(label)
    
    # Create heatmap
    im = ax4.imshow(obss_generation_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax4.set_xticks(range(num_channels))
    ax4.set_xticklabels([f'Channel {i}' for i in range(num_channels)])
    ax4.set_yticks(range(len(config_names)))
    ax4.set_yticklabels(channel_labels, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('OBSS Events Generated', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(config_names)):
        for j in range(num_channels):
            value = obss_generation_data[i][j]
            text_color = 'white' if value > np.max(obss_generation_data) * 0.5 else 'black'
            ax4.text(j, i, f'{int(value)}', ha='center', va='center', 
                    color=text_color, fontweight='bold')
    
    ax4.set_title('OBSS Generation Heatmap', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure/obss_deferrals_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_mutual_interference_analysis(results):
    """Plot mutual interference analysis between intra-BSS and OBSS with channel-specific insights"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # Plot 1: OBSS Generation vs Deferrals with channel breakdown
    ax1 = axes[0, 0]
    
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    x = np.arange(len(config_names))
    width = 0.8 / num_channels
    
    for ch_id in range(num_channels):
        obss_generated = []
        obss_deferred = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            ch_obss_stats = stats['obss_per_channel'][ch_id]
            
            obss_generated.append(ch_obss_stats['generated'])
            obss_deferred.append(ch_obss_stats['deferred'])
        
        offset = (ch_id - num_channels/2 + 0.5) * width
        
        bars1 = ax1.bar(x + offset - width/4, obss_generated, width/2, alpha=0.7, 
                       color=plt.cm.tab10(ch_id), label=f'Ch{ch_id} Generated' if ch_id == 0 else "")
        bars2 = ax1.bar(x + offset + width/4, obss_deferred, width/2, alpha=0.5, 
                       color=plt.cm.tab10(ch_id), hatch='///', label=f'Ch{ch_id} Deferred' if ch_id == 0 else "")
    
    ax1.set_title('OBSS Generation vs Deferrals by Channel', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cross-channel OBSS Impact
    ax2 = axes[0, 1]
    
    # Show how OBSS in one channel affects STAs in other channels
    cross_channel_impact = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        obss_config = stats['obss_enabled_per_channel']
        
        impact_data = []
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            avg_obss_deferrals = np.mean([sta_stats['obss_deferrals'] for _, sta_stats in channel_stas]) if channel_stas else 0
            impact_data.append(avg_obss_deferrals)
        
        cross_channel_impact.append(impact_data)
    
    # Create heatmap
    im = ax2.imshow(cross_channel_impact, cmap='Reds', aspect='auto', interpolation='nearest')
    
    ax2.set_xticks(range(num_channels))
    ax2.set_xticklabels([f'Channel {i}' for i in range(num_channels)])
    ax2.set_yticks(range(len(config_names)))
    ax2.set_yticklabels(config_names, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Avg OBSS Deferrals per STA', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(config_names)):
        for j in range(num_channels):
            value = cross_channel_impact[i][j]
            text_color = 'white' if value > np.max(cross_channel_impact) * 0.5 else 'black'
            ax2.text(j, i, f'{value:.0f}', ha='center', va='center', 
                    color=text_color, fontweight='bold')
    
    ax2.set_title('Cross-Channel OBSS Impact', fontsize=12, fontweight='bold')
    
    # Plot 3: Channel Asymmetry Analysis
    ax3 = axes[1, 0]
    
    # Compare performance difference between channels in asymmetric OBSS scenarios
    asymmetry_configs = []
    throughput_ratios = []
    aoi_ratios = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        obss_config = stats['obss_enabled_per_channel']
        
        # Only analyze if OBSS is asymmetric (not all channels same)
        if len(set(obss_config)) > 1:  # Mixed True/False
            asymmetry_configs.append(config_name)
            
            # Calculate performance for each channel
            ch_throughputs = []
            ch_aois = []
            
            for ch_id in range(num_channels):
                channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                               if sta_stats['channel'] == ch_id]
                
                if channel_stas:
                    ch_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
                    ch_throughput = (ch_successful * config['frame_size']) / stats['total_slots']
                    ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                    
                    ch_throughputs.append(ch_throughput)
                    ch_aois.append(ch_aoi)
            
            # Calculate ratios (max/min to show asymmetry)
            if len(ch_throughputs) > 1:
                tp_ratio = max(ch_throughputs) / min(ch_throughputs) if min(ch_throughputs) > 0 else 0
                aoi_ratio = max(ch_aois) / min(ch_aois) if min(ch_aois) > 0 else 0
                throughput_ratios.append(tp_ratio)
                aoi_ratios.append(aoi_ratio)
    
    if asymmetry_configs:
        x_asym = np.arange(len(asymmetry_configs))
        width = 0.35
        
        bars1 = ax3.bar(x_asym - width/2, throughput_ratios, width, alpha=0.7, 
                       color='skyblue', label='Throughput Ratio (Max/Min)')
        bars2 = ax3.bar(x_asym + width/2, aoi_ratios, width, alpha=0.7, 
                       color='lightcoral', label='AoI Ratio (Max/Min)')
        
        ax3.set_title('Channel Performance Asymmetry', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Ratio (Max Channel / Min Channel)')
        ax3.set_xlabel('Asymmetric OBSS Configurations')
        ax3.set_xticks(x_asym)
        ax3.set_xticklabels(asymmetry_configs, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Perfect Fairness')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No Asymmetric\nOBSS Configurations', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Channel Performance Asymmetry', fontsize=12, fontweight='bold')
    
    # Plot 4: OBSS Efficiency vs Mutual Interference
    ax4 = axes[1, 1]
    
    # Calculate efficiency metrics for each configuration
    for i, config_name in enumerate(config_names):
        stats = results[config_name]['stats']
        obss_config = stats['obss_enabled_per_channel']
        
        total_generated = stats['obss_events_generated']
        total_deferred = stats['obss_events_deferred']
        mutual_events = stats['mutual_interference_events']
        
        # OBSS success rate
        total_attempts = total_generated + total_deferred
        success_rate = (total_generated / total_attempts * 100) if total_attempts > 0 else 0
        
        # Mutual interference rate per slot
        interference_rate = (mutual_events / stats['total_slots'] * 1000) if stats['total_slots'] > 0 else 0  # per 1000 slots
        
        # Plot point with size based on number of active OBSS channels
        active_channels = sum(obss_config)
        point_size = 50 + active_channels * 100
        
        scatter = ax4.scatter(interference_rate, success_rate, s=point_size, alpha=0.7, 
                             color=colors[i], edgecolors='black', linewidth=1)
        
        # Add configuration label
        ax4.annotate(f'{config_name}\n({active_channels} OBSS Ch)', 
                    (interference_rate, success_rate), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_title('OBSS Efficiency vs Mutual Interference', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mutual Interference Rate (events per 1000 slots)')
    ax4.set_ylabel('OBSS Success Rate (%)')
    ax4.grid(True, alpha=0.3)
    
    # Add legend for point sizes
    legend_elements = []
    for i in range(num_channels + 1):
        size = 50 + i * 100
        legend_elements.append(plt.scatter([], [], s=size, alpha=0.7, color='gray', 
                                         label=f'{i} OBSS Channels'))
    ax4.legend(handles=legend_elements, loc='upper right', title='Active OBSS Channels')
    
    plt.tight_layout()
    plt.savefig('figure/mutual_interference_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_channel_fairness_analysis(results):
    """Plot channel access fairness analysis with channel-specific OBSS considerations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    
    # Plot 1: Per-Channel Utilization with OBSS Status
    ax1 = axes[0, 0]
    
    for idx, (config_name, result) in enumerate(results.items()):
        stats = result['stats']
        config = result['config']
        obss_config = stats['obss_enabled_per_channel']
        
        channel_utils = []
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            total_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
            util = (total_successful * config['frame_size']) / stats['total_slots']
            channel_utils.append(util)
        
        x_pos = np.arange(len(channel_utils)) + idx * 0.15
        bars = ax1.bar(x_pos, channel_utils, width=0.12, alpha=0.7, 
                      color=colors[idx], label=config_name)
        
        # Add value labels and OBSS status
        for i, (util, bar) in enumerate(zip(channel_utils, bars)):
            ax1.text(bar.get_x() + bar.get_width()/2, util + 0.002, 
                    f'{util:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add OBSS indicator
            if obss_config[i]:
                ax1.text(bar.get_x() + bar.get_width()/2, util + 0.008, 
                        '●', ha='center', va='bottom', fontsize=10, color='red')
    
    ax1.set_title('Channel Utilization with OBSS Status', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Channel Utilization')
    ax1.set_xlabel('Channel ID')
    ax1.set_xticks(np.arange(num_channels) + 0.3)
    ax1.set_xticklabels([f'Channel {i}' for i in range(num_channels)])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add legend for OBSS indicator
    ax1.text(0.02, 0.98, '● = OBSS Enabled', transform=ax1.transAxes, 
            va='top', ha='left', fontsize=9, color='red', weight='bold')
    
    # Plot 2: Fairness Index with Channel-specific Analysis
    ax2 = axes[0, 1]
    
    overall_fairness = []
    channel_fairness = {ch_id: [] for ch_id in range(num_channels)}
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        # Overall fairness
        sta_throughputs = []
        for sta_stats in stats['stations'].values():
            throughput = (sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots']
            sta_throughputs.append(throughput)
        
        if sta_throughputs:
            sum_throughputs = sum(sta_throughputs)
            sum_squared = sum(t**2 for t in sta_throughputs)
            n = len(sta_throughputs)
            fairness = (sum_throughputs**2) / (n * sum_squared) if sum_squared > 0 else 0
        else:
            fairness = 0
        overall_fairness.append(fairness)
        
        # Per-channel fairness
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if len(channel_stas) > 1:
                ch_throughputs = [(sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots'] 
                                 for _, sta_stats in channel_stas]
                ch_sum = sum(ch_throughputs)
                ch_sum_sq = sum(t**2 for t in ch_throughputs)
                ch_n = len(ch_throughputs)
                ch_fairness = (ch_sum**2) / (ch_n * ch_sum_sq) if ch_sum_sq > 0 else 0
            else:
                ch_fairness = 1.0  # Perfect fairness for single STA
            
            channel_fairness[ch_id].append(ch_fairness)
    
    # Plot overall fairness
    x = np.arange(len(config_names))
    bars = ax2.bar(x, overall_fairness, alpha=0.7, color=colors, width=0.6)
    ax2.set_title('System Fairness Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Jain\'s Fairness Index')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, overall_fairness):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Inter-Channel Fairness Analysis
    ax3 = axes[1, 0]
    
    # Calculate fairness between channels (not within channels)
    inter_channel_fairness = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        channel_avg_throughputs = []
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                ch_total_tp = sum((sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots'] 
                                for _, sta_stats in channel_stas)
                ch_avg_tp = ch_total_tp / len(channel_stas)
                channel_avg_throughputs.append(ch_avg_tp)
        
        # Calculate inter-channel fairness
        if len(channel_avg_throughputs) > 1:
            ch_sum = sum(channel_avg_throughputs)
            ch_sum_sq = sum(t**2 for t in channel_avg_throughputs)
            ch_n = len(channel_avg_throughputs)
            inter_fairness = (ch_sum**2) / (ch_n * ch_sum_sq) if ch_sum_sq > 0 else 0
        else:
            inter_fairness = 1.0
        
        inter_channel_fairness.append(inter_fairness)
    
    bars3 = ax3.bar(range(len(config_names)), inter_channel_fairness, alpha=0.7, color=colors)
    ax3.set_title('Inter-Channel Fairness', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fairness Index (Between Channels)')
    ax3.set_ylim(0, 1)
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels and OBSS impact indicators
    for i, (bar, value, config_name) in enumerate(zip(bars3, inter_channel_fairness, config_names)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add OBSS asymmetry indicator
        obss_config = results[config_name]['stats']['obss_enabled_per_channel']
        if len(set(obss_config)) > 1:  # Asymmetric OBSS
            ax3.text(bar.get_x() + bar.get_width()/2, 0.05,
                    '⚠', ha='center', va='bottom', fontsize=12, color='orange')
    
    ax3.text(0.02, 0.15, '⚠ = Asymmetric OBSS', transform=ax3.transAxes, 
            va='bottom', ha='left', fontsize=9, color='orange', weight='bold')
    
    # Plot 4: OBSS Impact on Channel Performance Distribution
    ax4 = axes[1, 1]
    
    # Show distribution of channel performance under different OBSS configurations
    performance_data = []
    labels = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        obss_config = stats['obss_enabled_per_channel']
        
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                # Calculate normalized performance (throughput / expected_fair_share)
                ch_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
                ch_throughput = (ch_successful * config['frame_size']) / stats['total_slots']
                
                # Expected fair share (total system throughput / number of channels)
                total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
                system_throughput = (total_successful * config['frame_size']) / stats['total_slots']
                expected_share = system_throughput / num_channels
                
                normalized_perf = ch_throughput / expected_share if expected_share > 0 else 0
                performance_data.append(normalized_perf)
                
                obss_status = "OBSS" if obss_config[ch_id] else "Clean"
                labels.append(f"{config_name}\nCh{ch_id}({obss_status})")
    
    # Create box plot grouped by OBSS status
    obss_perfs = [perf for i, perf in enumerate(performance_data) if "OBSS" in labels[i]]
    clean_perfs = [perf for i, perf in enumerate(performance_data) if "Clean" in labels[i]]
    
    box_data = []
    box_labels = []
    
    if clean_perfs:
        box_data.append(clean_perfs)
        box_labels.append(f'Clean Channels\n(n={len(clean_perfs)})')
    
    if obss_perfs:
        box_data.append(obss_perfs)
        box_labels.append(f'OBSS Channels\n(n={len(obss_perfs)})')
    
    if box_data:
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        colors_box = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Expected Fair Share')
    ax4.set_title('Channel Performance Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Normalized Performance\n(Actual / Expected Fair Share)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('figure/channel_fairness_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

# Print function was already provided in the previous response, so keeping it as is
def print_obss_results(results):
    """Print detailed OBSS comparison results with channel-specific analysis"""
    
    print("\n" + "="*80)
    print("CHANNEL-SPECIFIC OBSS INTERFERENCE ANALYSIS")
    print("="*80)
    
    for config_name, result in results.items():
        stats = result['stats']
        config = result['config']
        
        print(f"\n📊 {config_name}")
        print("-" * 50)
        print(f"OBSS Configuration: {stats['obss_enabled_per_channel']}")
        
        # 채널별 OBSS 상태 출력
        print(f"\n🔧 Channel-specific OBSS Status:")
        total_obss_channels = sum(stats['obss_enabled_per_channel'])
        print(f"Active OBSS channels: {total_obss_channels}/{config['num_channels']}")
        
        for ch_id in range(config['num_channels']):
            obss_ch_stats = stats['obss_per_channel'][ch_id]
            if obss_ch_stats['enabled']:
                print(f"  Channel {ch_id}: OBSS ENABLED")
                print(f"    Generated: {obss_ch_stats['generated']}")
                print(f"    Deferred: {obss_ch_stats['deferred']}")
                print(f"    Blocked by intra-BSS: {obss_ch_stats['blocked_by_intra']}")
                print(f"    Blocked by other OBSS: {obss_ch_stats['blocked_by_other_obss']}")
                
                # OBSS 성공률 계산
                total_attempts = obss_ch_stats['generated'] + obss_ch_stats['deferred']
                success_rate = (obss_ch_stats['generated'] / total_attempts * 100) if total_attempts > 0 else 0
                print(f"    Success rate: {success_rate:.1f}%")
            else:
                print(f"  Channel {ch_id}: OBSS DISABLED")
        
        # 전체 OBSS 통계 (활성화된 채널들의 합계)
        if stats['obss_enabled']:
            print(f"\n🌐 Overall OBSS Statistics:")
            print(f"OBSS Generation Rate: {stats['obss_generation_rate']:.1%} per slot")
            print(f"Total OBSS Events Generated: {stats['obss_events_generated']}")
            print(f"Total OBSS Events Deferred: {stats['obss_events_deferred']}")
            print(f"Total OBSS Duration: {stats['obss_total_duration_slots']} slots ({stats['obss_total_duration_us']/1000:.1f} ms)")
            print(f"OBSS Channel Utilization: {stats['obss_channel_utilization']:.1%}")
            
            # 상호 간섭 통계
            print(f"\n🔄 Mutual Interference:")
            print(f"  OBSS blocked by Intra-BSS: {stats['obss_blocked_by_intra_bss']}")
            print(f"  OBSS blocked by other OBSS: {stats['obss_blocked_by_other_obss']}")
            print(f"  Total mutual interference events: {stats['mutual_interference_events']}")
            
            # 전체 OBSS 성공률
            total_obss_attempts = stats['obss_events_generated'] + stats['obss_events_deferred']
            obss_success_rate = (stats['obss_events_generated'] / total_obss_attempts * 100) if total_obss_attempts > 0 else 0
            print(f"  Overall OBSS success rate: {obss_success_rate:.1f}%")
        else:
            print(f"\n🌐 Overall OBSS Statistics: DISABLED")
        
        # 시뮬레이션 기본 정보
        print(f"\n⏱️ Simulation Parameters:")
        print(f"Simulation time: {stats['total_time_us']/1000:.1f} ms ({stats['total_slots']} slots)")
        print(f"Channels: {config['num_channels']}, STAs per channel: {config['stas_per_channel']}")
        print(f"Frame size: {config['frame_size']} slots")
        
        # 시스템 성능 계산
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        total_attempts = sum(sta_stats['total_attempts'] for sta_stats in stats['stations'].values())
        total_collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        total_obss_deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        total_intra_deferrals = sum(sta_stats['intra_bss_deferrals'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * config['frame_size']) / stats['total_slots']
        system_success_rate = total_successful / total_attempts if total_attempts > 0 else 0
        avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        print(f"\n📈 System Performance:")
        print(f"System throughput: {system_throughput:.4f} ({system_throughput*100:.2f}%)")
        print(f"System success rate: {system_success_rate:.4f} ({system_success_rate*100:.2f}%)")
        print(f"Average AoI: {avg_aoi:.1f} μs")
        print(f"Total successful transmissions: {total_successful}")
        print(f"Total collisions: {total_collisions}")
        print(f"Total STA OBSS deferrals: {total_obss_deferrals}")
        print(f"Total STA Intra-BSS deferrals: {total_intra_deferrals}")
        
        # 공정성 지수 계산
        sta_throughputs = []
        for sta_stats in stats['stations'].values():
            throughput = (sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots']
            sta_throughputs.append(throughput)
        
        if sta_throughputs:
            sum_tp = sum(sta_throughputs)
            sum_sq = sum(t**2 for t in sta_throughputs)
            n = len(sta_throughputs)
            fairness = (sum_tp**2) / (n * sum_sq) if sum_sq > 0 else 0
            print(f"Jain's Fairness Index: {fairness:.3f}")
        
        # 채널별 상세 분석
        print(f"\n🔧 Per-Channel Performance Analysis:")
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                          if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                # 채널별 성능 지표 계산
                ch_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
                ch_attempts = sum(sta_stats['total_attempts'] for _, sta_stats in channel_stas)
                ch_throughput = (ch_successful * config['frame_size']) / stats['total_slots']
                ch_success_rate = ch_successful / ch_attempts if ch_attempts > 0 else 0
                ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                ch_collisions = sum(sta_stats['collisions'] for _, sta_stats in channel_stas)
                ch_obss_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
                ch_intra_deferrals = sum(sta_stats['intra_bss_deferrals'] for _, sta_stats in channel_stas)
                
                # OBSS 상태 표시
                obss_status = "🟢 ENABLED" if stats['obss_enabled_per_channel'][ch_id] else "🔴 DISABLED"
                
                print(f"  Channel {ch_id} ({len(channel_stas)} STAs) - OBSS {obss_status}:")
                print(f"    Throughput: {ch_throughput:.4f} ({ch_throughput*100:.2f}%)")
                print(f"    Success rate: {ch_success_rate:.3f} ({ch_success_rate*100:.1f}%)")
                print(f"    Average AoI: {ch_aoi:.1f} μs")
                print(f"    Collisions: {ch_collisions}")
                print(f"    OBSS deferrals: {ch_obss_deferrals}")
                print(f"    Intra-BSS deferrals: {ch_intra_deferrals}")
                print(f"    Total deferrals: {ch_obss_deferrals + ch_intra_deferrals}")
                
                # 채널 내 공정성
                ch_sta_throughputs = [(sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots'] 
                                     for _, sta_stats in channel_stas]
                if len(ch_sta_throughputs) > 1:
                    ch_sum_tp = sum(ch_sta_throughputs)
                    ch_sum_sq = sum(t**2 for t in ch_sta_throughputs)
                    ch_n = len(ch_sta_throughputs)
                    ch_fairness = (ch_sum_tp**2) / (ch_n * ch_sum_sq) if ch_sum_sq > 0 else 0
                    print(f"    Channel fairness: {ch_fairness:.3f}")
    
    # 전체 비교 분석
    print(f"\n📊 CROSS-CONFIGURATION COMPARISON ANALYSIS")
    print("-" * 50)
    
    # 기준점 찾기 (모든 채널에서 OBSS가 비활성화된 설정)
    baseline_name = None
    for config_name, result in results.items():
        if not any(result['stats']['obss_enabled_per_channel']):
            baseline_name = config_name
            break
    
    if baseline_name:
        baseline_stats = results[baseline_name]['stats']
        baseline_config = results[baseline_name]['config']
        baseline_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in baseline_stats['stations'].values()) * baseline_config['frame_size'] / baseline_stats['total_slots']
        baseline_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in baseline_stats['stations'].values()])
        
        print(f"📍 Baseline: {baseline_name}")
        print(f"   Throughput: {baseline_throughput:.4f}")
        print(f"   Average AoI: {baseline_aoi:.1f} μs")
        
        print(f"\n📉 Performance Impact vs Baseline:")
        
        for config_name, result in results.items():
            if config_name == baseline_name:
                continue
                
            stats = result['stats']
            config = result['config']
            
            current_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values()) * config['frame_size'] / stats['total_slots']
            current_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
            
            throughput_change = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
            aoi_change = ((current_aoi - baseline_aoi) / baseline_aoi) * 100
            
            # OBSS 활성화 상태 요약
            active_channels = [i for i, enabled in enumerate(stats['obss_enabled_per_channel']) if enabled]
            obss_summary = f"OBSS on Ch{active_channels}" if active_channels else "No OBSS"
            
            print(f"  {config_name} ({obss_summary}):")
            print(f"    Throughput change: {throughput_change:+.1f}%")
            print(f"    AoI change: {aoi_change:+.1f}%")
            print(f"    OBSS events generated: {stats['obss_events_generated']}")
            print(f"    OBSS events deferred: {stats['obss_events_deferred']}")
            print(f"    Mutual interference events: {stats['mutual_interference_events']}")
            print(f"    Total STA OBSS deferrals: {sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())}")
    
    # 채널별 비교 분석
    print(f"\n🔍 CHANNEL-WISE IMPACT ANALYSIS:")
    print("-" * 30)
    
    if baseline_name:
        baseline_stats = results[baseline_name]['stats']
        baseline_config = results[baseline_name]['config']
        
        # 각 채널별 기준 성능 계산
        baseline_ch_performance = {}
        for ch_id in range(baseline_config['num_channels']):
            ch_stas = [(sta_id, sta_stats) for sta_id, sta_stats in baseline_stats['stations'].items() 
                      if sta_stats['channel'] == ch_id]
            
            if ch_stas:
                ch_throughput = sum((sta_stats['successful_transmissions'] * baseline_config['frame_size']) / baseline_stats['total_slots'] 
                                  for _, sta_stats in ch_stas)
                ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in ch_stas])
                baseline_ch_performance[ch_id] = {'throughput': ch_throughput, 'aoi': ch_aoi}
        
        for config_name, result in results.items():
            if config_name == baseline_name:
                continue
                
            stats = result['stats']
            config = result['config']
            
            print(f"\n  {config_name}:")
            
            for ch_id in range(config['num_channels']):
                ch_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                          if sta_stats['channel'] == ch_id]
                
                if ch_stas and ch_id in baseline_ch_performance:
                    ch_throughput = sum((sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots'] 
                                      for _, sta_stats in ch_stas)
                    ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in ch_stas])
                    ch_obss_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in ch_stas)
                    
                    baseline_ch_tp = baseline_ch_performance[ch_id]['throughput']
                    baseline_ch_aoi = baseline_ch_performance[ch_id]['aoi']
                    
                    tp_change = ((ch_throughput - baseline_ch_tp) / baseline_ch_tp) * 100 if baseline_ch_tp > 0 else 0
                    aoi_change = ((ch_aoi - baseline_ch_aoi) / baseline_ch_aoi) * 100 if baseline_ch_aoi > 0 else 0
                    
                    obss_status = "🟢" if stats['obss_enabled_per_channel'][ch_id] else "🔴"
                    
                    print(f"    Channel {ch_id} {obss_status}: Throughput {tp_change:+.1f}%, AoI {aoi_change:+.1f}%, OBSS Def: {ch_obss_deferrals}")
    
    print(f"\n✅ Analysis Complete!")
    print(f"📋 Summary: Analyzed {len(results)} configurations across {len(results[list(results.keys())[0]]['config']['stas_per_channel'])} channels")