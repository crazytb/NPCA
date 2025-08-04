from random_access.random_access import *

def create_sta_scenario(n_legacy: int, n_npca: int) -> Tuple[List[Channel], List[STA]]:
    channels = [
        Channel(0, obss_generation_rate=0.1, obss_duration_range=(20, 40)),
        Channel(1, obss_generation_rate=0.1, obss_duration_range=(20, 40))
    ]
    stas = []
    sta_id_counter = 0

    # Legacy STAs on Channel 0
    for _ in range(n_legacy):
        sta = STA(
            sta_id=sta_id_counter,
            channel_id=0,
            primary_channel=channels[0],
            npca_channel=None,
            npca_enabled=False
        )
        stas.append(sta)
        sta_id_counter += 1

    # NPCA STAs on Channel 1, using channel 0 as NPCA channel
    for _ in range(n_npca):
        sta = STA(
            sta_id=sta_id_counter,
            channel_id=1,
            primary_channel=channels[1],
            npca_channel=channels[0],
            npca_enabled=True
        )
        stas.append(sta)
        sta_id_counter += 1

    return channels, stas

channels, stas = create_sta_scenario(n_legacy=3, n_npca=5)
sim = Simulator(num_slots=200, stas=stas, channels=channels)
sim.run()
df = sim.get_dataframe()
df.to_csv("sim_result.csv", index=False)
