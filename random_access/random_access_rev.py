import numpy as np
import pandas as pd
import random

CONTENTION_WINDOW = [2 ** (i + 4) - 1 for i in range(7)]
SLOTTIME = 9


class STA:
    def __init__(self, sta_id, channel_id, npca=False):
        self.sta_id = sta_id
        self.primary_channel = channel_id
        self.current_channel = channel_id
        self.npca = npca
        if npca:
            self.npca_channel = 1 - channel_id

        self.status = 'backoff'
        self.bo_stage = 0
        self.bo_counter = self.get_new_backoff()
        self.aoitimestamp = 0
        self.tx_attempt = False
        self.has_packet = True

    def get_new_backoff(self):
        cw = CONTENTION_WINDOW[min(self.bo_stage, len(CONTENTION_WINDOW) - 1)]
        return np.random.randint(0, cw + 1)

    def tick(self, channel_busy):
        if not self.has_packet:
            self.status = 'idle'
            self.tx_attempt = False
            return

        if self.status in ['transmitting', 'waiting_result']:
            return

        if channel_busy:
            self.status = 'frozen'
            self.tx_attempt = False
        else:
            if self.bo_counter > 0:
                self.bo_counter -= 1
                self.status = 'backoff'
                self.tx_attempt = False
            else:
                self.status = 'transmitting'
                self.tx_attempt = True

    def on_tx_result(self, result, current_slot):
        if result == 'succ':
            self.status = 'backoff'
            self.bo_stage = 0
            self.bo_counter = self.get_new_backoff()
            if self.bo_counter == 0:
                self.bo_counter = 1
            self.has_packet = True
            self.aoitimestamp = current_slot
        elif result in ('coll', 'fail'):
            self.status = 'backoff'
            self.bo_stage = min(self.bo_stage + 1, len(CONTENTION_WINDOW) - 1)
            self.bo_counter = self.get_new_backoff()
        else:
            self.status = 'idle'


class Channel:
    def __init__(self, channel_id):
        self.channel_id = channel_id
        self.transmissions = []
        self.occupied_until = 0

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


class CSMANetwork:
    def __init__(self, num_channels, stas_per_channel, beaconinterval, num_episodes, frametxslot, per, npca_flags=None):
        self.num_channels = num_channels
        self.stas_per_channel = stas_per_channel
        self.beaconinterval = beaconinterval
        self.num_episodes = num_episodes
        self.frametxslot = frametxslot
        self.per = per
        self.total_slots = int(num_episodes * beaconinterval / SLOTTIME) + frametxslot

        if npca_flags is None:
            npca_flags = [[False] * n for n in stas_per_channel]

        self.channels = [Channel(i) for i in range(num_channels)]
        self.stas = [STA(sta_id, ch_id, npca=npca_flags[ch_id][sta_id])
                     for ch_id in range(num_channels)
                     for sta_id in range(stas_per_channel[ch_id])]
        self.pending_results = {}
        self.logs = []
        self.slot = 0

    def run(self):
        for self.slot in range(self.total_slots):
            self.tick()
        return pd.DataFrame(self.logs)

    def tick(self):
        self._commit_pending_results()
        self._clear_channels()
        self._update_sta_statuses()
        self._collect_tx_attempts()
        self._resolve_channel_access()

    def _commit_pending_results(self):
        to_delete = []
        for (sta, end_slot), (result, ch_id) in self.pending_results.items():
            if self.slot == end_slot:
                self.logs.append({
                    'time': end_slot * SLOTTIME,
                    'node': sta.sta_id,
                    'timestamp': sta.aoitimestamp * SLOTTIME,
                    'result': result,
                    'channel': ch_id,
                    'duration': self.frametxslot * SLOTTIME
                })
                sta.on_tx_result(result, end_slot)
                to_delete.append((sta, end_slot))
        for key in to_delete:
            del self.pending_results[key]

    def _clear_channels(self):
        for ch in self.channels:
            ch.clear()

    def _update_sta_statuses(self):
        for sta in self.stas:
            ch = self.channels[sta.current_channel]
            channel_busy = (self.slot < ch.occupied_until)
            sta.tick(channel_busy)

    def _collect_tx_attempts(self):
        for sta in self.stas:
            if sta.tx_attempt:
                self.channels[sta.current_channel].add_tx(sta)

    def _resolve_channel_access(self):
        for ch in self.channels:
            results = ch.resolve()
            for sta, result in results:
                if result == 'succ' and random.random() < self.per[ch.channel_id]:
                    result = 'fail'
                end_slot = self.slot + self.frametxslot
                key = (sta, end_slot)
                if key not in self.pending_results:
                    self.pending_results[key] = (result, ch.channel_id)
                    ch.occupied_until = end_slot
