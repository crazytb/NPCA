# Configuration

simulation_time = 10000  # Total simulation time in us
simulation_slot = simulation_time // 9  # Slot duration in us (9us for 802.11ax)
frame_size = 33  # Size of each frame in slots

simulation_configs = [
    # {
    #     "label": "No NPCA Short Frame Duration", 
    #     "num_channels": 2,
    #     "stas_per_channel": [10, 10],
    #     "simulation_time": simulation_time,
    #     "frame_size": frame_size,
    #     "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
    #     "npca_enabled": [False, False],  # 채널 1에서만 NPCA 활성화
    #     "obss_generation_rate": 0.01
    # },
    # {
    #     "label": "No NPCA Long Frame Duration",
    #     "num_channels": 2,
    #     "stas_per_channel": [10, 10],
    #     "simulation_time": simulation_time,
    #     "frame_size": frame_size*10,
    #     "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
    #     "npca_enabled": [False, False],  # 채널 1에서만 NPCA 활성화
    #     "obss_generation_rate": 0.01
    # },
    {
        "label": "NPCA Short Frame Duration", 
        "num_channels": 2,
        "stas_per_channel": [2, 2],
        "simulation_time": simulation_time,
        "frame_size": frame_size,
        "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
        "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
        "obss_generation_rate": 0.01
    },
    # {
    #     "label": "NPCA Long Frame Duration",
    #     "num_channels": 2,
    #     "stas_per_channel": [10, 10],
    #     "simulation_time": simulation_time,
    #     "frame_size": frame_size*10,
    #     "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
    #     "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
    #     "obss_generation_rate": 0.01
    # }
]

# simulation_configs = [
#     {
#         "label": "Channel 1: 5 STAs with NPCA",
#         "num_channels": 2,
#         "stas_per_channel": [10, 5],
#         "simulation_time": simulation_time,
#         "frame_size": frame_size,
#         "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
#         "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
#         "obss_generation_rate": 0.01
#     },
#     {
#         "label": "Channel 1: 10 STAs with NPCA", 
#         "num_channels": 2,
#         "stas_per_channel": [10, 10],
#         "simulation_time": simulation_time,
#         "frame_size": frame_size,
#         "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
#         "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
#         "obss_generation_rate": 0.01
#     },
#     {
#         "label": "Channel 1: 20 STAs with NPCA",
#         "num_channels": 2,
#         "stas_per_channel": [10, 20],
#         "simulation_time": simulation_time,
#         "frame_size": frame_size,
#         "obss_enabled_per_channel": [False, True],  # 채널 1에서만 OBSS 활성화
#         "npca_enabled": [False, True],  # 채널 1에서만 NPCA 활성화
#         "obss_generation_rate": 0.01
#     }
# ]