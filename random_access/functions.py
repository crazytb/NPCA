import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random_access.random_access_rev import SimplifiedCSMACASimulation, SLOTTIME
from random_access.configs import *

def run_obss_comparison(simulation_configs):
    """Run simulations comparing OBSS enabled vs disabled"""
    
    results = {}
    
    for config in simulation_configs:
        print(f"Running simulation: {config['label']}")
        
        sim = SimplifiedCSMACASimulation(
            num_channels=config["num_channels"],
            stas_per_channel=config["stas_per_channel"],
            simulation_time=config["simulation_time"],
            frame_size=config["frame_size"],
            obss_enabled=config["obss_enabled"],
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
    plt.savefig('figure/obss_comparison.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('figure/obss_channel_impact.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('figure/obss_fsm_states.png', dpi=300, bbox_inches='tight')

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
    plt.savefig('figure/obss_deferrals_analysis.png', dpi=300, bbox_inches='tight')

def print_obss_results(results):
    """Print detailed OBSS comparison results"""
    
    print("\n" + "="*80)
    print("MUTUAL OBSS INTERFERENCE ANALYSIS")  # ✨ 제목 수정
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
            print(f"OBSS Events Deferred: {stats['obss_events_deferred']}")  # ✨ 새로 추가
            print(f"OBSS Total Duration: {stats['obss_total_duration_slots']} slots ({stats['obss_total_duration_us']/1000:.1f} ms)")
            print(f"OBSS Channel Utilization: {stats['obss_channel_utilization']:.1%}")
            
            # ✨ 상호 간섭 통계 추가
            print(f"\n🔄 Mutual Interference:")
            print(f"  OBSS blocked by Intra-BSS: {stats['obss_blocked_by_intra_bss']}")
            print(f"  OBSS blocked by other OBSS: {stats['obss_blocked_by_other_obss']}")
            print(f"  Total mutual interference events: {stats['mutual_interference_events']}")
            
            # ✨ OBSS 성공률 추가
            total_obss_attempts = stats['obss_events_generated'] + stats['obss_events_deferred']
            obss_success_rate = (stats['obss_events_generated'] / total_obss_attempts * 100) if total_obss_attempts > 0 else 0
            print(f"  OBSS success rate: {obss_success_rate:.1f}%")
        
        # 기존 시스템 성능 출력...
        print(f"\nSimulation time: {stats['total_time_us']/1000:.1f} ms ({stats['total_slots']} slots)")
        
        # Calculate system metrics
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        total_attempts = sum(sta_stats['total_attempts'] for sta_stats in stats['stations'].values())
        total_collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        total_obss_deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        total_intra_deferrals = sum(sta_stats['intra_bss_deferrals'] for sta_stats in stats['stations'].values())  # ✨ 새로 추가
        system_throughput = (total_successful * frame_size) / stats['total_slots']
        system_success_rate = total_successful / total_attempts if total_attempts > 0 else 0
        avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        print(f"\n📈 System Performance:")  # ✨ 섹션 구분
        print(f"System throughput: {system_throughput:.4f} ({system_throughput*100:.2f}%)")
        print(f"System success rate: {system_success_rate:.4f} ({system_success_rate*100:.2f}%)")
        print(f"Average AoI: {avg_aoi:.1f} μs")
        print(f"Total successful transmissions: {total_successful}")
        print(f"Total collisions: {total_collisions}")
        print(f"Total STA OBSS deferrals: {total_obss_deferrals}")
        print(f"Total STA Intra-BSS deferrals: {total_intra_deferrals}")  # ✨ 새로 추가
        
        # ✨ 공정성 지수 추가
        sta_throughputs = []
        for sta_stats in stats['stations'].values():
            throughput = (sta_stats['successful_transmissions'] * frame_size) / stats['total_slots']
            sta_throughputs.append(throughput)
        
        if sta_throughputs:
            sum_tp = sum(sta_throughputs)
            sum_sq = sum(t**2 for t in sta_throughputs)
            n = len(sta_throughputs)
            fairness = (sum_tp**2) / (n * sum_sq) if sum_sq > 0 else 0
            print(f"Jain's Fairness Index: {fairness:.3f}")
        
        # 기존 채널별 분석...
        print(f"\n🔧 Per-Channel Analysis:")
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                          if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                ch_throughput = sum((sta_stats['successful_transmissions'] * frame_size) / stats['total_slots'] 
                                  for _, sta_stats in channel_stas)
                ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                ch_collisions = sum(sta_stats['collisions'] for _, sta_stats in channel_stas)
                ch_obss_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
                ch_intra_deferrals = sum(sta_stats['intra_bss_deferrals'] for _, sta_stats in channel_stas)  # ✨ 새로 추가
                
                print(f"  Channel {ch_id} ({len(channel_stas)} STAs):")
                print(f"    Throughput: {ch_throughput:.4f}, AoI: {ch_aoi:.1f}μs")
                print(f"    Collisions: {ch_collisions}, OBSS Def: {ch_obss_deferrals}, Intra Def: {ch_intra_deferrals}")  # ✨ 수정
    
    # 기존 성능 영향 분석도 업데이트...
    print(f"\n📊 OBSS PERFORMANCE IMPACT ANALYSIS")  # ✨ 제목 수정
    print("-" * 50)
    
    baseline_name = list(results.keys())[0]
    baseline_stats = results[baseline_name]['stats']
    baseline_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in baseline_stats['stations'].values()) * frame_size / baseline_stats['total_slots']
    baseline_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in baseline_stats['stations'].values()])
    
    for config_name, result in list(results.items())[1:]:
        stats = result['stats']
        
        current_throughput = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values()) * frame_size / stats['total_slots']
        current_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        throughput_degradation = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
        aoi_increase = ((current_aoi - baseline_aoi) / baseline_aoi) * 100
        
        print(f"\n{config_name} vs {baseline_name}:")
        print(f"  Throughput degradation: {throughput_degradation:.1f}%")
        print(f"  AoI increase: {aoi_increase:.1f}%")
        print(f"  OBSS events generated: {stats['obss_events_generated']}")
        print(f"  OBSS events deferred: {stats['obss_events_deferred']}")  # ✨ 새로 추가
        print(f"  Mutual interference events: {stats['mutual_interference_events']}")  # ✨ 새로 추가
        print(f"  Total STA OBSS deferrals: {sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())}")

        
def plot_mutual_interference_analysis(results):
    """Plot mutual interference analysis between intra-BSS and OBSS"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = ['skyblue', 'orange', 'lightcoral']
    
    # Plot 1: OBSS Generation vs Deferrals
    ax1 = axes[0, 0]
    
    obss_generated = []
    obss_deferred = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        generated = stats.get('obss_events_generated', 0)
        deferred = stats.get('obss_events_deferred', 0)
        
        obss_generated.append(generated)
        obss_deferred.append(deferred)
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1a = ax1.bar(x - width/2, obss_generated, width, alpha=0.7, color='lightblue', label='OBSS Generated')
    bars1b = ax1.bar(x + width/2, obss_deferred, width, alpha=0.7, color='lightcoral', label='OBSS Deferred')
    
    ax1.set_title('OBSS Generation vs Deferrals', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1a, bars1b]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Mutual Interference Breakdown
    ax2 = axes[0, 1]
    
    obss_blocked_by_intra = []
    obss_blocked_by_obss = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        blocked_intra = stats.get('obss_blocked_by_intra_bss', 0)
        blocked_obss = stats.get('obss_blocked_by_other_obss', 0)
        
        obss_blocked_by_intra.append(blocked_intra)
        obss_blocked_by_obss.append(blocked_obss)
    
    bars2a = ax2.bar(x - width/2, obss_blocked_by_intra, width, alpha=0.7, 
                     color='steelblue', label='OBSS blocked by Intra-BSS')
    bars2b = ax2.bar(x + width/2, obss_blocked_by_obss, width, alpha=0.7, 
                     color='orange', label='OBSS blocked by other OBSS')
    
    ax2.set_title('OBSS Blocking Causes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Blocking Events')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: STA Deferrals Breakdown (Intra-BSS vs OBSS)
    ax3 = axes[1, 0]
    
    total_intra_deferrals = []
    total_obss_deferrals = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        intra_def = sum(sta_stats['intra_bss_deferrals'] for sta_stats in stats['stations'].values())
        obss_def = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        
        total_intra_deferrals.append(intra_def)
        total_obss_deferrals.append(obss_def)
    
    bars3a = ax3.bar(x - width/2, total_intra_deferrals, width, alpha=0.7, 
                     color='lightgreen', label='STA Intra-BSS Deferrals')
    bars3b = ax3.bar(x + width/2, total_obss_deferrals, width, alpha=0.7, 
                     color='salmon', label='STA OBSS Deferrals')
    
    ax3.set_title('STA Deferrals by Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Deferral Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars3a, bars3b]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Mutual Interference Efficiency
    ax4 = axes[1, 1]
    
    obss_success_rates = []
    mutual_interference_rates = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        generated = stats.get('obss_events_generated', 0)
        deferred = stats.get('obss_events_deferred', 0)
        mutual_events = stats.get('mutual_interference_events', 0)
        
        # OBSS success rate (generated / (generated + deferred))
        total_attempts = generated + deferred
        success_rate = (generated / total_attempts * 100) if total_attempts > 0 else 0
        obss_success_rates.append(success_rate)
        
        # Mutual interference rate per slot
        interference_rate = (mutual_events / stats['total_slots'] * 100) if stats['total_slots'] > 0 else 0
        mutual_interference_rates.append(interference_rate)
    
    bars4a = ax4.bar(x - width/2, obss_success_rates, width, alpha=0.7, 
                     color='gold', label='OBSS Success Rate (%)')
    bars4b = ax4.bar(x + width/2, mutual_interference_rates, width, alpha=0.7, 
                     color='crimson', label='Mutual Interference Rate (%)')
    
    ax4.set_title('OBSS Efficiency & Mutual Interference', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Rate (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars4a, bars4b]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figure/mutual_interference_analysis.png', dpi=300, bbox_inches='tight')

def plot_channel_fairness_analysis(results):
    """Plot channel access fairness analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(results.keys())
    colors = ['skyblue', 'orange', 'lightcoral']
    
    # Plot 1: Per-Channel Utilization
    ax1 = axes[0, 0]
    
    for idx, (config_name, result) in enumerate(results.items()):
        stats = result['stats']
        config = result['config']
        
        channel_utils = []
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            total_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
            util = (total_successful * frame_size) / stats['total_slots']
            channel_utils.append(util)
        
        x_pos = np.arange(len(channel_utils)) + idx * 0.25
        ax1.bar(x_pos, channel_utils, width=0.2, alpha=0.7, 
               color=colors[idx], label=config_name)
        
        # Add value labels
        for i, util in enumerate(channel_utils):
            ax1.text(x_pos[i], util + 0.002, f'{util:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_title('Channel Utilization Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Channel Utilization')
    ax1.set_xlabel('Channel ID')
    ax1.set_xticks(np.arange(2) + 0.25)
    ax1.set_xticklabels(['Channel 0', 'Channel 1'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fairness Index (Jain's Fairness Index)
    ax2 = axes[0, 1]
    
    fairness_indices = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        # Calculate throughput for each STA
        sta_throughputs = []
        for sta_stats in stats['stations'].values():
            throughput = (sta_stats['successful_transmissions'] * frame_size) / stats['total_slots']
            sta_throughputs.append(throughput)
        
        # Calculate Jain's Fairness Index
        if sta_throughputs:
            sum_throughputs = sum(sta_throughputs)
            sum_squared = sum(t**2 for t in sta_throughputs)
            n = len(sta_throughputs)
            fairness = (sum_throughputs**2) / (n * sum_squared) if sum_squared > 0 else 0
        else:
            fairness = 0
        
        fairness_indices.append(fairness)
    
    bars2 = ax2.bar(config_names, fairness_indices, alpha=0.7, color=colors)
    ax2.set_title('Jain\'s Fairness Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fairness Index (0-1)')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, fairness_indices):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Access Delay Distribution
    ax3 = axes[1, 0]
    
    for idx, (config_name, result) in enumerate(results.items()):
        stats = result['stats']
        
        # Calculate average access delay (deferrals) per STA
        access_delays = []
        for sta_stats in stats['stations'].values():
            total_deferrals = sta_stats['total_deferrals']
            attempts = sta_stats['total_attempts']
            avg_delay = total_deferrals / attempts if attempts > 0 else 0
            access_delays.append(avg_delay)
        
        # Create box plot data
        if access_delays:
            positions = [idx + 1]
            bp = ax3.boxplot(access_delays, positions=positions, widths=0.6, 
                           patch_artist=True, labels=[config_name])
            bp['boxes'][0].set_facecolor(colors[idx])
            bp['boxes'][0].set_alpha(0.7)
    
    ax3.set_title('Access Delay Distribution', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg Deferrals per Attempt')
    ax3.set_xlabel('Configuration')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: OBSS Impact on Different Channels
    ax4 = axes[1, 1]
    
    if len([r for r in results.values() if r['config']['obss_enabled']]) > 0:
        # Only show this for OBSS-enabled configs
        obss_configs = [(name, result) for name, result in results.items() 
                       if result['config']['obss_enabled']]
        
        for idx, (config_name, result) in enumerate(obss_configs):
            stats = result['stats']
            config = result['config']
            
            channel_obss_impact = []
            for ch_id in range(config['num_channels']):
                channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                               if sta_stats['channel'] == ch_id]
                
                if channel_stas:
                    avg_obss_deferrals = np.mean([sta_stats['obss_deferrals'] for _, sta_stats in channel_stas])
                    channel_obss_impact.append(avg_obss_deferrals)
                else:
                    channel_obss_impact.append(0)
            
            x_pos = np.arange(len(channel_obss_impact)) + idx * 0.3
            ax4.bar(x_pos, channel_obss_impact, width=0.25, alpha=0.7, 
                   color=colors[idx + 1], label=config_name)
            
            # Add value labels
            for i, impact in enumerate(channel_obss_impact):
                ax4.text(x_pos[i], impact + 5, f'{impact:.0f}', 
                        ha='center', va='bottom', fontsize=8)
        
        ax4.set_title('OBSS Impact per Channel', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Avg OBSS Deferrals per STA')
        ax4.set_xlabel('Channel ID')
        ax4.set_xticks(np.arange(2) + 0.15)
        ax4.set_xticklabels(['Channel 0', 'Channel 1'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No OBSS Data Available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('OBSS Impact per Channel', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure/channel_fairness_analysis.png', dpi=300, bbox_inches='tight')