import numpy as np
import pandas as pd
import random

CONTENTION_WINDOW = [2**(i+4)-1 for i in range(7)]  # CW_min = 15, CW_max = 1023

class STA:
    def __init__(self, sta_id, channel_id, npca=False):
        self.sta_id = sta_id
        self.current_channel = channel_id
        self.primary_channel = channel_id
        self.npca = npca
        if self.npca:
            self.npca_channel = 1-channel_id  # NPCA 채널은 1-채널 ID로 설정
        
        self.bo_stage = 0
        self.bo_counter = self.get_new_backoff()
        self.aoitimestamp = 0
        # self.has_packet = True
        self.tx_attempt = False

    def get_new_backoff(bo_stage: int) -> int:
        """
        Return a random backoff counter based on the backoff stage.
        Parameters:
            bo_stage (int): Backoff stage, must be between 0 and 6.
        Returns:
            int: Random backoff counter in range [0, CW[bo_stage]]
        """
        try:
            cw = CONTENTION_WINDOW[bo_stage]
        except IndexError:
            raise ValueError(f"Invalid bo_stage={bo_stage}. Must be between 0 and {len(CONTENTION_WINDOW) - 1}.")
        return np.random.randint(0, cw + 1)

    def update_backoff(self, channel_busy):
        if not channel_busy:
            if self.bo_counter > 0:
                self.bo_counter -= 1
        self.tx_attempt = (self.bo_counter == 0)

    def tick(self, channel_busy):
        if (not channel_busy) or self.npca:
            if self.bo_counter > 0:
                self.bo_counter -= 1

    def on_tx_result(self, result):
        if result == 'succ':
            self.bo_stage = 0
        elif result in ('coll', 'fail'):
            self.bo_stage = min(self.bo_stage + 1, 6)
        self.bo_counter = self.get_new_backoff()


class Channel:
    def __init__(self, channel_id):
        self.channel_id = channel_id
        self.transmissions = []

    def add_tx(self, sta):
        self.transmissions.append(sta)

    def resolve(self):
        if len(self.transmissions) == 1:
            return [(self.transmissions[0], 'succ')]
        elif len(self.transmissions) > 1:
            return [(sta, 'coll') for sta in self.transmissions]
        return []

    def clear(self):
        self.transmissions = []


def tick(stas, channels, slot, slottime, frametxslot, per):
    for ch in channels:
        ch.clear()

    for sta in stas:
        current_channel = sta.current_channel
        channel_busy = any(
            other.tx_attempt for other in stas if other != sta and other.current_channel == current_channel
        )
        sta.tick(channel_busy)
        if sta.tx_attempt:
            channels[current_channel].add_tx(sta)

    log = []
    for ch in channels:
        results = ch.resolve()
        for sta, result in results:
            if result == 'succ' and random.random() < per[ch.channel_id]:
                result = 'fail'
            log.append({
                'time': slot * slottime,
                'node': sta.sta_id,
                'timestamp': sta.aoitimestamp * slottime,
                'result': result,
                'channel': ch.channel_id,
                'duration': frametxslot
            })
            sta.on_tx_result(result)
    return log


def simulate_csma(num_channels, stas_per_channel, beaconinterval, num_episodes, frametxslot, per, npca_flags=None):
    slottime = 9
    total_slots = int(num_episodes * beaconinterval / slottime)

    if npca_flags is None:
        npca_flags = [[False] * n for n in stas_per_channel]

    channels = [Channel(i) for i in range(num_channels)]
    stas = []
    for ch_id in range(num_channels):
        for sta_id in range(stas_per_channel[ch_id]):
            sta = STA(
                sta_id=sta_id,
                channel_id=ch_id,
                cw_min=15,
                cw_max=1023,
                npca=npca_flags[ch_id][sta_id]
            )
            stas.append(sta)

    df_log = []
    for slot in range(total_slots):
        for ch in channels:
            ch.clear()

        # STA 백오프 진행
        for sta in stas:
            current_channel = sta.current_channel
            channel_busy = any(other.tx_attempt for other in stas if other.current_channel == current_channel and other != sta)
            sta.update_backoff(channel_busy)
            if sta.tx_attempt:
                channels[current_channel].add_tx(sta)

        # 전송 해석
        for ch in channels:
            results = ch.resolve()
            for sta, result in results:
                if result == 'succ' and random.random() < per[ch.channel_id]:
                    result = 'fail'  # PHY 오류 반영
                df_log.append({
                    'time': slot * slottime,
                    'node': sta.sta_id,
                    'timestamp': sta.aoitimestamp * slottime,
                    'result': result,
                    'channel': ch.channel_id,
                    'duration': frametxslot
                })
                sta.on_tx_result(result)

    return pd.DataFrame(df_log)
