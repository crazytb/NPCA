import copy

# Configuration
simulation_time = 1_000_000  # Total simulation time in us
simulation_slot = simulation_time // 9  # Slot duration in us (9us for 802.11ax)
frame_size = 33  # Size of each frame in slots
# stas_per_channel = [2, 2]  # Number of STAs per channel

# 공통 설정
base_config = {
    "num_channels": 2,
    "simulation_time": simulation_time,
    "obss_enabled_per_channel": [False, True],
    # "npca_enabled": [False, True],
    "obss_generation_rate": 0.01
}

# 후보 값
sta_values = [2, 6, 10] 
frame_sizes = [frame_size, frame_size * 10]
frame_labels = {33: "fshort", 33 * 10: "flong"}
npca_options = [[False, True], [False, False]]  # 추가된 부분

# 시뮬레이션 설정 생성
simulation_configs = []
for ch0 in sta_values:
    for ch1 in sta_values:
        for fs in frame_sizes:
            for npca_enabled in npca_options:
                config = copy.deepcopy(base_config)
                config["stas_per_channel"] = [ch0, ch1]
                config["frame_size"] = fs
                config["npca_enabled"] = npca_enabled
                config["label"] = f"s{ch0}_{ch1}_{frame_labels[fs]}_npca_{int(npca_enabled[1])}"
                simulation_configs.append(config)