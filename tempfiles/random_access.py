import numpy as np
import pandas as pd
import random

class Channel:
    def __init__(self, channel_id, num_sta, contentionwindowsize, frametxslot):
        self.channel_id = channel_id
        self.num_sta = num_sta
        self.frametxslot = frametxslot
        self.contentionwindowsize = contentionwindowsize
        self.currentslot = 0
        self.bo_stage = np.zeros(num_sta, dtype=int)
        self.bo_counter = np.array([np.random.randint(self.contentionwindowsize[x]) for x in self.bo_stage])
        self.retx = np.zeros(num_sta, dtype=int)
        self.aoitimestamp = np.zeros(num_sta, dtype=int)

    def is_ready(self):
        return np.any(self.bo_counter == 0)

    def advance_idle(self, skip_slots):
        self.currentslot += skip_slots
        self.aoitimestamp[self.retx == 0] = self.currentslot
        self.bo_counter -= skip_slots

    def handle_tx(self, per):
        results = []
        per_rv = random.random()
        ready_sta = np.where(self.bo_counter == 0)[0]
        if len(ready_sta) == 1:
            # Tx success or fail
            idx = ready_sta[0]
            if per_rv > per:
                result = 'succ'
                self.retx[idx] = 0
                self.bo_stage[idx] = 0
            else:
                result = 'fail'
                self.retx[idx] = 1
                self.bo_stage[idx] = min(self.bo_stage[idx] + 1, 5)
            self.bo_counter[idx] = np.random.randint(self.contentionwindowsize[self.bo_stage[idx]])
        elif len(ready_sta) > 1:
            # Collision
            result = 'coll'
            for idx in ready_sta:
                self.retx[idx] = 1
                self.bo_stage[idx] = min(self.bo_stage[idx] + 1, 5)
                self.bo_counter[idx] = np.random.randint(self.contentionwindowsize[self.bo_stage[idx]])
        else:
            return results  # no action

        self.currentslot += self.frametxslot
        for idx in ready_sta:
            results.append({
                'time': self.currentslot,
                'node': idx,
                'timestamp': self.aoitimestamp[idx],
                'result': result,
                'channel': self.channel_id,
                'duration': self.frametxslot
            })
        self.aoitimestamp[self.retx == 0] = self.currentslot
        return results

def randomaccess_eventdriven(num_channels, stas_per_channel, beaconinterval, num_episodes, frametxslot, per):
    slottime = 9
    sim_slots = int(num_episodes * beaconinterval / slottime)
    contentionwindowsize = [2 ** x for x in range(5, 11)]
    channels = [Channel(i, stas_per_channel[i], contentionwindowsize, frametxslot) for i in range(num_channels)]
    df = pd.DataFrame(columns=['time', 'node', 'timestamp', 'result', 'channel', 'duration'])

    while max(ch.currentslot for ch in channels) < sim_slots - frametxslot:
        if all(not ch.is_ready() for ch in channels):
            # 모든 채널에서 backoff 중 → 최소 bo_counter만큼 skip
            skip_slots = min(np.min(ch.bo_counter) for ch in channels)
            for ch in channels:
                ch.advance_idle(skip_slots)
        else:
            for ch in channels:
                if ch.is_ready():
                    results = ch.handle_tx(per[ch.channel_id])
                    if results:
                        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)

    return df
