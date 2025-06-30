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

    def handle_tx(self, per, slottime):
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
                    results = ch.handle_tx(per[ch.channel_id], slottime)
                    if results:
                        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)

    return df

# def randomaccess(num_channels, stas_per_channel, beaconinterval, num_episodes, frametxslot, per):
#     slottime = 9
#     currentslot = np.zeros(num_channels, dtype=int)
#     # print(f"Data frame length: {frametxslot}")
#     contentionwindowsize = [2**x for x in range(5, 11)]
#     bo_stage = np.empty(num_channels, dtype=object)
#     bo_counter = np.empty(num_channels, dtype=object)
#     succ_timestamp = np.empty(num_channels, dtype=object)
#     succ_namestamp = np.empty(num_channels, dtype=object)
#     num_succ = np.zeros(num_channels, dtype=int)
#     num_fail = np.zeros(num_channels, dtype=int)
#     num_coll = np.zeros(num_channels, dtype=int)
#     # Initialize backoff stage and counter for each channel
#     retx = np.empty(num_channels, dtype=object)
#     aoitimestamp = np.empty(num_channels, dtype=object)
#     for channel in range(num_channels):
#         # Backoff counter initialize
#         bo_stage[channel] = np.zeros(stas_per_channel[channel], dtype=int)
#         bo_counter[channel] = np.array([np.random.randint(contentionwindowsize[x]) for x in bo_stage[channel]], dtype=int)
#         retx[channel] = np.zeros(stas_per_channel[channel], dtype=int)
#         aoitimestamp[channel] = np.zeros(stas_per_channel[channel], dtype=int)
#     df = pd.DataFrame(columns=['time', 'node', 'timestamp', 'result', 'channel'])  # type: ignore
    
#     while np.min(currentslot) < num_episodes*beaconinterval/slottime - frametxslot:
#         for channel in range(num_channels):
#             per_rv = random.random()
#             # Idle
#             if np.min(bo_counter[channel]) != 0:
#                 currentslot[channel] += np.min(bo_counter[channel])
#                 aoitimestamp[channel, retx[channel] == 0] = currentslot[channel]
#                 bo_counter[channel] = bo_counter[channel] - np.min(bo_counter[channel])
#             # Tx succ
#             elif (per_rv > per[channel]) and ((np.min(bo_counter[channel]) == 0) and (np.size(bo_counter[channel]) - np.count_nonzero(bo_counter[channel]) == 1)):
#                 currentslot[channel] += frametxslot
#                 ind, = np.where(bo_counter[channel] == 0)
#                 retx[channel, ind] = 0
#                 # print(f"Time: {currentslot}, Tx success from {ind+1} with AoI {aoi[ind]}")
#                 df2 = pd.DataFrame({'time': currentslot[channel]*slottime, 
#                                     'node': ind + 1, 
#                                     'timestamp': aoitimestamp[channel, ind]*slottime, 
#                                     'result': 'succ',
#                                     'channel': channel})
#                 df = pd.concat([df, df2], ignore_index=True, axis=0)
#                 aoitimestamp[channel, retx[channel] == 0] = currentslot[channel]
#                 succ_timestamp = np.append(succ_timestamp, currentslot[channel])
#                 succ_namestamp = np.append(succ_namestamp, ind[0])
#                 bo_stage[ind] = 0
#                 bo_counter[channel, ind] = np.random.randint(contentionwindowsize[0])
#                 num_succ += 1
#             # Tx failed
#             elif (per_rv <= per[channel]) and ((np.min(bo_counter[channel]) == 0) and (np.size(bo_counter[channel]) - np.count_nonzero(bo_counter[channel]) == 1)):
#                 # aoi[retx == 0] += frametxslot
#                 currentslot[channel] += frametxslot
#                 ind, = np.where(bo_counter[channel] == 0)
#                 retx[channel, ind] = 1
#                 for x in ind:
#                     if bo_stage[x] < 5:
#                         bo_stage[x] += 1
#                     bo_counter[channel, x] = np.random.randint(contentionwindowsize[bo_stage[x]])
#                 # print(f"Time: {currentslot}, Tx collision from {ind+1} with AoI {aoi[ind]}")
#                 df2 = pd.DataFrame({'time': currentslot[channel]*slottime, 
#                                     'node': ind + 1, 
#                                     'timestamp': aoitimestamp[channel, ind]*slottime, 
#                                     'result': 'fail',
#                                     'channel': channel})
#                 df = pd.concat([df, df2], ignore_index=True, axis=0)
#                 aoitimestamp[channel, retx[channel] == 0] = currentslot[channel]
#                 num_fail += 1
#             # Tx coll
#             elif np.min(bo_counter[channel]) == 0 and (np.size(bo_counter[channel]) - np.count_nonzero(bo_counter[channel]) > 1):
#                 # aoi[retx == 0] += frametxslot
#                 currentslot[channel] += frametxslot
#                 ind, = np.where(bo_counter[channel] == 0)
#                 retx[channel, ind] = 1
#                 for x in ind:
#                     if bo_stage[x] < 5:
#                         bo_stage[x] += 1
#                     bo_counter[channel, x] = np.random.randint(contentionwindowsize[bo_stage[x]])
#                 # print(f"Time: {currentslot}, Tx collision from {ind+1} with AoI {aoi[ind]}")
#                 df2 = pd.DataFrame({'time': currentslot[channel]*slottime, 
#                                     'node': ind + 1, 
#                                     'timestamp': aoitimestamp[channel, ind]*slottime, 
#                                     'result': 'coll',
#                                     'channel': channel})
#                 df = pd.concat([df, df2], ignore_index=True, axis=0)
#                 aoitimestamp[channel, retx[channel] == 0] = currentslot[channel]
#                 num_coll += 1
#     return df

# def randomaccess_objectified(num_channels, stas_per_channel, beaconinterval, num_episodes, frametxslot, per):
#     slottime = 9
#     sim_slots = int(num_episodes * beaconinterval / slottime)
#     contentionwindowsize = [2 ** x for x in range(5, 11)]
#     channels = [Channel(i, stas_per_channel[i], contentionwindowsize, frametxslot) for i in range(num_channels)]
#     df = pd.DataFrame(columns=['time', 'node', 'timestamp', 'result', 'channel'])

#     for slot in range(sim_slots - frametxslot):
#         any_tx = False
#         for ch in channels:
#             if ch.is_ready():
#                 results = ch.handle_tx(per[ch.channel_id], slottime)
#                 if results:
#                     df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
#                     any_tx = True
#         if not any_tx:
#             for ch in channels:
#                 ch.advance_idle()

#     return df