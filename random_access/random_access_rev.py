import numpy as np
import pandas as pd
import random

# IEEE 802.11 CW 규격: CWmin = 15, CWmax = 1023
CONTENTION_WINDOW = [2 ** (i + 4) - 1 for i in range(7)]  # [15, 31, 63, 127, 255, 511, 1023]
SLOTTIME = 9

class STA:
    def __init__(self, sta_id, channel_id, npca=False):
        self.sta_id = sta_id
        self.primary_channel = channel_id
        self.current_channel = channel_id
        self.npca = npca
        if npca:
            self.npca_channel = 1 - channel_id  # assume 2 channels only

        self.bo_stage = 0
        self.bo_counter = self.get_new_backoff()
        self.aoitimestamp = 0
        self.tx_attempt = False
        self.has_packet = True

    def get_new_backoff(self):
        cw = CONTENTION_WINDOW[min(self.bo_stage, len(CONTENTION_WINDOW) - 1)]
        return np.random.randint(0, cw) + 1  # 1부터 시작하는 backoff

    def tick(self, channel_busy):
        if not channel_busy and self.bo_counter > 0:
            self.bo_counter -= 1
        self.tx_attempt = (self.bo_counter == 0 and self.has_packet)

    def on_tx_result(self, result, current_slot):
        if result == 'succ':
            self.bo_stage = 0
            self.bo_counter = self.get_new_backoff()
            self.has_packet = True
            self.aoitimestamp = current_slot
        elif result in ('coll', 'fail'):
            self.bo_stage = min(self.bo_stage + 1, len(CONTENTION_WINDOW) - 1)
            self.bo_counter = self.get_new_backoff()

class Channel:
    def __init__(self, channel_id):
        self.channel_id = channel_id
        self.transmissions = []
        self.occupied_until = 0  # 채널 점유 끝 슬롯

    def clear(self):
        self.transmissions = []

    def add_tx(self, sta):
        self.transmissions.append(sta)

    def resolve(self):
        if len(self.transmissions) == 1:
            return [(self.transmissions[0], 'succ')]
        elif len(self.transmissions) > 1:
            return [(sta, 'coll') for sta in self.transmissions]
        return []

def simulate_csma(num_channels, stas_per_channel, beaconinterval, num_episodes, frametxslot, per, npca_flags=None):
    total_slots = int(num_episodes * beaconinterval / SLOTTIME)

    if npca_flags is None:
        npca_flags = [[False] * n for n in stas_per_channel]

    # 채널 및 STA 초기화
    channels = [Channel(i) for i in range(num_channels)]
    stas = []
    for ch_id in range(num_channels):
        for sta_id in range(stas_per_channel[ch_id]):
            stas.append(STA(sta_id, ch_id, npca=npca_flags[ch_id][sta_id]))

    df_log = []
    pending_results = []

    for slot in range(total_slots):
        for pr in list(pending_results):
            end_slot, sta, result, ch_id = pr
            if end_slot == slot:
                df_log.append({
                    'time': end_slot * SLOTTIME,
                    'node': sta.sta_id,
                    'timestamp': sta.aoitimestamp * SLOTTIME,
                    'result': result,
                    'channel': ch_id,
                    'duration': frametxslot * SLOTTIME
                })
                sta.on_tx_result(result, end_slot)
                pending_results.remove(pr)

        for ch in channels:
            ch.clear()

        for sta in stas:
            ch = channels[sta.current_channel]
            channel_busy = (slot < ch.occupied_until)
            sta.tick(channel_busy)

        for sta in stas:
            if sta.tx_attempt:
                channels[sta.current_channel].add_tx(sta)

        for ch in channels:
            results = ch.resolve()
            for sta, result in results:
                # PER 적용
                if result == 'succ' and random.random() < per[ch.channel_id]:
                    result = 'fail'

                end_slot = slot + frametxslot
                pending_results.append((end_slot, sta, result, ch.channel_id))
                ch.occupied_until = end_slot

    return pd.DataFrame(df_log)
