# Configuration

simulation_time = 10000  # Total simulation time in slots
frame_size = 33  # Size of each frame in slots

# simulation_configs = [
#     {"num_channels": 2, "stas_per_channel": [2, 10], "obss_enabled": False, 
#         "obss_generation_rate": 0.0, "simulation_time": simulation_time, "frame_size": frame_size, 
#         "label": "OBSS Disabled"},
#     {"num_channels": 2, "stas_per_channel": [2, 10], "obss_enabled": True, 
#         "obss_generation_rate": 0.002, "simulation_time": simulation_time, "frame_size": frame_size, 
#         "label": "High OBSS (0.2%)"},
#     {"num_channels": 2, "stas_per_channel": [2, 10], "obss_enabled": True, 
#         "obss_generation_rate": 0.02, "simulation_time": simulation_time, "frame_size": frame_size, 
#         "label": "Extremely High OBSS (2%)"},
#     ]

# simulation_configs = [
#     {
#         "label": "No OBSS",
#         "num_channels": 2,
#         "stas_per_channel": [2, 10],
#         "simulation_time": simulation_time,
#         "frame_size": frame_size,
#         "obss_enabled_per_channel": [False, False],  # 모든 채널에서 OBSS 비활성화
#         "obss_generation_rate": 0.01
#     },
#     {
#         "label": "OBSS on Channel 0 only",
#         "num_channels": 2,
#         "stas_per_channel": [2, 10],
#         "simulation_time": simulation_time,
#         "frame_size": frame_size,
#         "obss_enabled_per_channel": [True, False],   # 채널 0에서만 OBSS 활성화
#         "obss_generation_rate": 0.01
#     },
#     {
#         "label": "OBSS on Channel 1 only",
#         "num_channels": 2,
#         "stas_per_channel": [2, 10],
#         "simulation_time": simulation_time,
#         "frame_size": frame_size,
#         "obss_enabled_per_channel": [False, True],   # 채널 1에서만 OBSS 활성화
#         "obss_generation_rate": 0.01
#     },
#     {
#         "label": "OBSS on Both Channels",
#         "num_channels": 2,
#         "stas_per_channel": [2, 10],
#         "simulation_time": simulation_time,
#         "frame_size": frame_size,
#         "obss_enabled_per_channel": [True, True],    # 모든 채널에서 OBSS 활성화
#         "obss_generation_rate": 0.01
#     }
# ]

simulation_configs = [
    {
        "label": "Channel 1: 5 STAs with NPCA",
        "num_channels": 2,
        "stas_per_channel": [2, 5],
        "simulation_time": simulation_time,
        "frame_size": frame_size,
        "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
        "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
        "obss_generation_rate": 0.01
    },
    {
        "label": "Channel 1: 10 STAs with NPCA", 
        "num_channels": 2,
        "stas_per_channel": [2, 10],
        "simulation_time": simulation_time,
        "frame_size": frame_size,
        "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
        "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
        "obss_generation_rate": 0.01
    },
    {
        "label": "Channel 1: 20 STAs with NPCA",
        "num_channels": 2,
        "stas_per_channel": [2, 20],
        "simulation_time": simulation_time,
        "frame_size": frame_size,
        "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
        "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
        "obss_generation_rate": 0.01
    }
]