# Configuration

simulation_time = 100000  # Total simulation time in slots
frame_size = 33  # Size of each frame in bytes

simulation_configs = [
    {"num_channels": 2, "stas_per_channel": [2, 10], "obss_enabled": False, 
        "obss_generation_rate": 0.0, "simulation_time": simulation_time, "frame_size": frame_size, 
        "label": "OBSS Disabled"},
    {"num_channels": 2, "stas_per_channel": [2, 10], "obss_enabled": True, 
        "obss_generation_rate": 0.002, "simulation_time": simulation_time, "frame_size": frame_size, 
        "label": "High OBSS (0.2%)"},
    {"num_channels": 2, "stas_per_channel": [2, 10], "obss_enabled": True, 
        "obss_generation_rate": 0.02, "simulation_time": simulation_time, "frame_size": frame_size, 
        "label": "Extremely High OBSS (2%)"},
    ]