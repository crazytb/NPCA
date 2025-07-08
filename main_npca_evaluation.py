# main_npca_evaluation.py
"""
Main NPCA Performance Evaluation Script
Evaluates NPCA vs Legacy STA performance in OBSS environments

Usage: python main_npca_evaluation.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from random_access.npca_simulation import NPCASimulation
from random_access.configs import frame_size

def create_npca_evaluation_configs():
    """Create NPCA evaluation configurations"""
    
    simulation_time = 100000  # 100k slots for comprehensive evaluation
    
    configs = [
        # === HIGH OBSS (0.2%) SCENARIOS ===
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 0],  # All Legacy baseline
            "obss_enabled": True,
            "obss_generation_rate": 0.002,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "High OBSS - All Legacy"
        },
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 1],  # 1 NPCA + 9 Legacy
            "obss_enabled": True,
            "obss_generation_rate": 0.002,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "High OBSS - 1 NPCA + 9 Legacy"
        },
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 5],  # 5 NPCA + 5 Legacy
            "obss_enabled": True,
            "obss_generation_rate": 0.002,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "High OBSS - 5 NPCA + 5 Legacy"
        },
        
        # === EXTREMELY HIGH OBSS (2%) SCENARIOS ===
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 0],  # All Legacy baseline
            "obss_enabled": True,
            "obss_generation_rate": 0.02,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "Extremely High OBSS - All Legacy"
        },
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 1],  # 1 NPCA + 9 Legacy
            "obss_enabled": True,
            "obss_generation_rate": 0.02,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "Extremely High OBSS - 1 NPCA + 9 Legacy"
        },
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 5],  # 5 NPCA + 5 Legacy
            "obss_enabled": True,
            "obss_generation_rate": 0.02,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "Extremely High OBSS - 5 NPCA + 5 Legacy"
        }
    ]
    
    return configs

def run_npca_evaluation():
    """Run comprehensive NPCA evaluation"""
    
    configs = create_npca_evaluation_configs()
    results = {}
    
    print("🚀 Starting NPCA Performance Evaluation...")
    print("🎯 Evaluating NPCA vs Legacy STA performance in OBSS environments")
    print("=" * 80)
    
    for i, config in enumerate(configs):
        print(f"\n📊 Running Scenario {i+1}/{len(configs)}: {config['label']}")
        print(f"   OBSS Rate: {config['obss_generation_rate']:.1%}")
        print(f"   Channel 1 STAs: {config['stas_per_channel'][1]} total, {config['npca_stas_per_channel'][1]} NPCA")
        
        try:
            # Create and run simulation
            sim = NPCASimulation(
                num_channels=config["num_channels"],
                stas_per_channel=config["stas_per_channel"],
                npca_stas_per_channel=config["npca_stas_per_channel"],
                simulation_time=config["simulation_time"],
                frame_size=config["frame_size"],
                obss_enabled=config["obss_enabled"],
                obss_generation_rate=config["obss_generation_rate"]
            )
            
            # Run simulation
            print("   🔄 Running simulation...")
            df = sim.run()
            stats = sim.get_enhanced_statistics()
            
            # Store results
            results[config['label']] = {
                'config': config,
                'stats': stats,
                'dataframe': df
            }
            
            # Save individual CSV
            csv_filename = f"npca_results_{config['label'].replace(' ', '_').replace('-', '').lower()}.csv"
            df.to_csv(csv_filename, index=False)
            
            # Print quick summary
            print(f"   ✅ Completed!")
            print(f"      OBSS events generated: {stats['obss_events_generated']}")
            print(f"      Total NPCA STAs: {stats['total_npca_stas']}")
            print(f"      Total Legacy STAs: {stats['total_legacy_stas']}")
            
            if stats['total_npca_stas'] > 0:
                perf = stats['performance_comparison']
                print(f"      📈 Throughput improvement: {perf['throughput_improvement']:+.2f}%")
                print(f"      📉 AoI improvement: {perf['aoi_improvement']:+.2f}%")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def print_detailed_results(results):
    """Print detailed evaluation results"""
    
    print("\n" + "="*80)
    print("📊 DETAILED CROSS-CHANNEL NPCA PERFORMANCE EVALUATION RESULTS")  # 제목 변경
    print("="*80)
    
    # Group results by OBSS level
    high_obss_results = {k: v for k, v in results.items() if "High OBSS -" in k and "Extremely" not in k}
    extreme_obss_results = {k: v for k, v in results.items() if "Extremely High OBSS" in k}
    
    for group_name, group_results in [("HIGH OBSS (0.2%)", high_obss_results), 
                                      ("EXTREMELY HIGH OBSS (2%)", extreme_obss_results)]:
        
        print(f"\n🎯 {group_name} ANALYSIS")
        print("-" * 60)
        
        for label, result in group_results.items():
            stats = result['stats']
            config = result['config']
            
            print(f"\n📋 {label}")
            print(f"   Configuration: {config['stas_per_channel']} STAs per channel")
            print(f"   NPCA STAs: {stats['total_npca_stas']}, Legacy STAs: {stats['total_legacy_stas']}")
            print(f"   OBSS Events Generated: {stats['obss_events_generated']}")
            print(f"   Simulation Duration: {stats['total_time_us']/1000:.1f}ms ({stats['total_slots']} slots)")
            
            # Calculate overall system performance
            total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
            system_throughput = (total_successful * config['frame_size']) / stats['total_slots']
            avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
            
            print(f"   📊 System Performance:")
            print(f"      System Throughput: {system_throughput:.6f} ({system_throughput*100:.4f}%)")
            print(f"      Average AoI: {avg_aoi:.1f} μs")
            
            # NPCA specific metrics
            if stats['total_npca_stas'] > 0:
                perf = stats['performance_comparison']
                npca_sum = stats['npca_summary']
                
                print(f"   🚀 NPCA vs Legacy Performance:")
                print(f"      NPCA Avg Throughput: {perf['npca_avg_throughput']:.6f}")
                print(f"      Legacy Avg Throughput: {perf['legacy_avg_throughput']:.6f}")
                print(f"      Throughput Improvement: {perf['throughput_improvement']:+.2f}%")
                print(f"      NPCA Avg AoI: {perf['npca_avg_aoi']:.1f} μs")
                print(f"      Legacy Avg AoI: {perf['legacy_avg_aoi']:.1f} μs")
                print(f"      AoI Improvement: {perf['aoi_improvement']:+.2f}%")
                
                # Cross-Channel NPCA 메커니즘 출력 (기존 코드 교체)
                print(f"   ⚡ Cross-Channel NPCA Mechanism Performance:")
                print(f"      OBSS Immunity Activations: {npca_sum.get('total_obss_immunity_activations', 0)}")
                print(f"      Channel Switches: {npca_sum.get('total_channel_switches', 0)}")
                print(f"      Alternative Channel Transmissions: {npca_sum.get('total_alternative_channel_transmissions', 0)}")
                print(f"      Alternative Channel Success Rate: {npca_sum.get('avg_alternative_channel_success_rate', 0):.3f}")
                
                # Cross-Channel OBSS 통계 추가
                if 'cross_channel_obss' in stats:
                    cc_obss = stats['cross_channel_obss']
                    print(f"   🔄 Cross-Channel OBSS Impact:")
                    print(f"      NPCA OBSS Events: {cc_obss.get('total_npca_obss_events', 0)}")
                    print(f"      Channel 0 NPCA OBSS Events: {cc_obss.get('channel_0_npca_obss_events', 0)}")
                    print(f"      Channel 0 NPCA OBSS Duration: {cc_obss.get('channel_0_npca_obss_duration', 0)} slots")
                
                # 채널별 성능 분석 추가
                if 'channel_specific_performance' in stats:
                    ch_perf = stats['channel_specific_performance']
                    print(f"   📊 Channel-Specific Performance:")
                    print(f"      Channel 0 Legacy Throughput: {ch_perf.get('channel_0_legacy_throughput', 0):.6f}")
                    print(f"      Channel 1 NPCA Throughput: {ch_perf.get('channel_1_npca_throughput', 0):.6f}")
                    print(f"      Channel 1 Legacy Throughput: {ch_perf.get('channel_1_legacy_throughput', 0):.6f}")

def generate_summary_csv(results):
    """Generate summary CSV for analysis"""
    
    summary_data = []
    
    for label, result in results.items():
        stats = result['stats']
        config = result['config']
        
        # Calculate system metrics
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * config['frame_size']) / stats['total_slots']
        avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        row = {
            'scenario': label,
            'obss_rate_percent': config['obss_generation_rate'] * 100,
            'npca_stas': stats['total_npca_stas'],
            'legacy_stas': stats['total_legacy_stas'],
            'obss_events': stats['obss_events_generated'],
            'system_throughput': system_throughput,
            'system_avg_aoi_us': avg_aoi,
        }
        
        # Add NPCA-specific metrics if available
        if stats['total_npca_stas'] > 0:
            perf = stats['performance_comparison']
            npca_sum = stats['npca_summary']
            
            row.update({
                'npca_avg_throughput': perf['npca_avg_throughput'],
                'legacy_avg_throughput': perf['legacy_avg_throughput'],
                'throughput_improvement_pct': perf['throughput_improvement'],
                'npca_avg_aoi_us': perf['npca_avg_aoi'],
                'legacy_avg_aoi_us': perf['legacy_avg_aoi'],
                'aoi_improvement_pct': perf['aoi_improvement'],
                'obss_immunity_activations': npca_sum.get('total_obss_immunity_activations', 0),
                # Cross-Channel 메트릭 추가
                'channel_switches': npca_sum.get('total_channel_switches', 0),
                'alternative_channel_transmissions': npca_sum.get('total_alternative_channel_transmissions', 0),
                'alternative_channel_success_rate': npca_sum.get('avg_alternative_channel_success_rate', 0),
            })
            
            # Cross-Channel OBSS 메트릭 추가
            if 'cross_channel_obss' in stats:
                cc_obss = stats['cross_channel_obss']
                row.update({
                    'npca_obss_events': cc_obss.get('total_npca_obss_events', 0),
                    'channel_0_npca_obss_events': cc_obss.get('channel_0_npca_obss_events', 0),
                    'channel_0_npca_obss_duration': cc_obss.get('channel_0_npca_obss_duration', 0),
                })
            
            # 채널별 성능 메트릭 추가
            if 'channel_specific_performance' in stats:
                ch_perf = stats['channel_specific_performance']
                row.update({
                    'channel_0_legacy_throughput': ch_perf.get('channel_0_legacy_throughput', 0),
                    'channel_1_npca_throughput': ch_perf.get('channel_1_npca_throughput', 0),
                    'channel_1_legacy_throughput': ch_perf.get('channel_1_legacy_throughput', 0),
                })
        else:
            # Fill with zeros for all-legacy scenarios
            row.update({
                'npca_avg_throughput': 0,
                'legacy_avg_throughput': system_throughput,
                'throughput_improvement_pct': 0,
                'npca_avg_aoi_us': 0,
                'legacy_avg_aoi_us': avg_aoi,
                'aoi_improvement_pct': 0,
                'obss_immunity_activations': 0,
                'channel_switches': 0,
                'alternative_channel_transmissions': 0,
                'alternative_channel_success_rate': 0,
                'npca_obss_events': 0,
                'channel_0_npca_obss_events': 0,
                'channel_0_npca_obss_duration': 0,
                'channel_0_legacy_throughput': 0,
                'channel_1_npca_throughput': 0,
                'channel_1_legacy_throughput': 0,
            })
        
        summary_data.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('npca_evaluation_summary.csv', index=False)
    
    print(f"\n💾 Files saved:")
    print(f"   📊 Summary: npca_evaluation_summary.csv")
    print(f"   📁 Individual results: npca_results_*.csv")

def run_quick_test():
    """Run a quick test with detailed log analysis"""
    
    print("🧪 Running Quick NPCA Test with Detailed Analysis...")
    
    test_config = {
        "num_channels": 2,
        "stas_per_channel": [1, 4],  # Smaller test
        "npca_stas_per_channel": [0, 2],  # 2 NPCA STAs
        "obss_enabled": True,
        "obss_generation_rate": 0.01,
        "simulation_time": 10000,  # Shorter simulation
        "frame_size": frame_size,
        "label": "Quick Test"
    }
    
    try:
        # Import the log analyzer
        from npca_log_analyzer import analyze_npca_logs
        
        sim = NPCASimulation(
            num_channels=test_config["num_channels"],
            stas_per_channel=test_config["stas_per_channel"],
            npca_stas_per_channel=test_config["npca_stas_per_channel"],
            simulation_time=test_config["simulation_time"],
            frame_size=test_config["frame_size"],
            obss_enabled=test_config["obss_enabled"],
            obss_generation_rate=test_config["obss_generation_rate"]
        )
        
        print("🔄 Running simulation...")
        df = sim.run()
        stats = sim.get_enhanced_statistics()
        
        print(f"✅ Quick test completed!")
        print(f"   NPCA STAs: {stats['total_npca_stas']}")
        print(f"   Legacy STAs: {stats['total_legacy_stas']}")
        print(f"   OBSS events: {stats['obss_events_generated']}")
        print(f"   Throughput improvement: {stats['performance_comparison']['throughput_improvement']:.2f}%")

        # Cross-Channel 메트릭 추가
        if stats['total_npca_stas'] > 0:
            print(f"   Channel switches: {stats['npca_summary'].get('total_channel_switches', 0)}")
            print(f"   Alternative channel transmissions: {stats['npca_summary'].get('total_alternative_channel_transmissions', 0)}")
            if 'cross_channel_obss' in stats:
                cc_obss = stats['cross_channel_obss']
                print(f"   NPCA OBSS events generated: {cc_obss.get('total_npca_obss_events', 0)}")
        
        # Save detailed logs for analysis
        df.to_csv('quick_test_detailed_log.csv', index=False)
        print(f"📁 Detailed log saved to: quick_test_detailed_log.csv")
        
        # Run detailed log analysis
        print("\n" + "="*60)
        print("🔍 STARTING DETAILED LOG ANALYSIS")
        print("="*60)
        
        analyzer = analyze_npca_logs(sim, df, stats)
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_basic_quick_test():
    """Run basic quick test without detailed analysis"""
    
    print("🧪 Running Basic Quick NPCA Test...")
    
    test_config = {
        "num_channels": 2,
        "stas_per_channel": [1, 4],
        "npca_stas_per_channel": [0, 2],
        "obss_enabled": True,
        "obss_generation_rate": 0.01,
        "simulation_time": 10000,
        "frame_size": frame_size,
        "label": "Quick Test"
    }
    
    try:
        sim = NPCASimulation(
            num_channels=test_config["num_channels"],
            stas_per_channel=test_config["stas_per_channel"],
            npca_stas_per_channel=test_config["npca_stas_per_channel"],
            simulation_time=test_config["simulation_time"],
            frame_size=test_config["frame_size"],
            obss_enabled=test_config["obss_enabled"],
            obss_generation_rate=test_config["obss_generation_rate"]
        )
        
        df = sim.run()
        stats = sim.get_enhanced_statistics()

        print(f"✅ Quick test completed!")
        print(f"   NPCA STAs: {stats['total_npca_stas']}")
        print(f"   Legacy STAs: {stats['total_legacy_stas']}")
        print(f"   OBSS events: {stats['obss_events_generated']}")
        print(f"   Throughput improvement: {stats['performance_comparison']['throughput_improvement']:.2f}%")

        # Cross-Channel 메트릭 추가
        if stats['total_npca_stas'] > 0:
            print(f"   Channel switches: {stats['npca_summary'].get('total_channel_switches', 0)}")
            print(f"   Alternative channel transmissions: {stats['npca_summary'].get('total_alternative_channel_transmissions', 0)}")
            if 'cross_channel_obss' in stats:
                cc_obss = stats['cross_channel_obss']
                print(f"   NPCA OBSS events generated: {cc_obss.get('total_npca_obss_events', 0)}")

        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Update the main section
if __name__ == "__main__":
    print("🎯 NPCA Performance Evaluation in OBSS Environments")
    print("   Comparing 1 NPCA + 9 Legacy vs 5 NPCA + 5 Legacy scenarios")
    print("   Under High OBSS (0.2%) and Extremely High OBSS (2%) conditions")
    print()
    
    # Ask user for test mode
    mode = input("Choose mode:\n1. Quick test (basic)\n2. Quick test (with detailed analysis)\n3. Full evaluation\nEnter choice (1, 2, or 3): ").strip()
    
    if mode == "1":
        # Basic quick test
        if run_basic_quick_test():
            print("\n🎉 Basic quick test successful! Ready for detailed analysis or full evaluation.")
        else:
            print("\n❌ Basic quick test failed. Please check your implementation.")
    
    elif mode == "2":
        # Detailed quick test
        if run_quick_test():
            print("\n🎉 Detailed quick test successful! Check the analysis above for insights.")
        else:
            print("\n❌ Detailed quick test failed. Please check your implementation.")
    
    elif mode == "3":
        # Full evaluation mode
        try:
            # Run full evaluation
            results = run_npca_evaluation()
            
            # Print detailed results
            print_detailed_results(results)
            
            # Generate summary CSV
            generate_summary_csv(results)
            
            print(f"\n🎉 NPCA Evaluation Complete!")
            print(f"✅ {len(results)} scenarios evaluated successfully!")
            
        except Exception as e:
            print(f"\n❌ Evaluation Failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("Invalid choice. Please run again with 1, 2, or 3.")