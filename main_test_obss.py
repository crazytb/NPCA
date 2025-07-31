from random_access.functions import *
from random_access.configs import simulation_configs

if __name__ == "__main__":
    # Run OBSS comparison simulations
    print("Starting mutual OBSS interference simulations...")  # ✨ 제목 수정
    results = run_obss_comparison(simulation_configs)
    
    # # Generate visualizations
    # print("\nGenerating OBSS comparison plots...")
    # plot_obss_comparison(results)
    
    # print("Generating channel-specific OBSS impact plots...")
    # plot_channel_specific_obss_impact(results)
    
    # print("Generating FSM states analysis...")
    # plot_fsm_states_analysis(results)
    
    # print("Generating mutual interference analysis...")  # ✨ 새로 추가
    # plot_mutual_interference_analysis(results)
    
    # print("Generating channel fairness analysis...")  # 수정새로 추가
    # plot_channel_fairness_analysis(results)
    
    # plot_npca_cross_channel_impact(results)  # ✨ NPCA 교차 채널 영향 분석
    
    # print("Generating OBSS deferrals analysis...")  # ✨ 제거하거나 유지 (선택사항)
    # plot_obss_deferrals_analysis(results)
    
    # # Print detailed results
    # print_obss_results(results)
    
    # print("\n✅ Mutual OBSS Analysis complete!")  # ✨ 제목 수정
    # print("📁 Plots saved as:")
    # print("   - obss_comparison.png")
    # print("   - obss_channel_impact.png")
    # print("   - obss_fsm_states.png")
    # print("   - mutual_interference_analysis.png")  # ✨ 새로 추가
    # print("   - channel_fairness_analysis.png")     # ✨ 새로 추가
    # print("   - obss_deferrals_analysis.png")       # ✨ 기존 유지 (선택사항)