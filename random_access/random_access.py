import pandas as pd
import random
from typing import List

# Copy-pasting the given code components into the environment to enable simulation
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

# Constants
CONTENTION_WINDOW = [2 ** (i + 4) - 1 for i in range(7)]  # CW from 15 to 1023
SLOTTIME = 9  # μs

class STAState(Enum):
    PRIMARY_BACKOFF = auto()
    PRIMARY_FROZEN = auto()
    PRIMARY_TX = auto()
    NPCA_BACKOFF = auto()
    NPCA_FROZEN = auto()
    NPCA_TX = auto()

@dataclass
class OBSSTraffic:
    obss_id: str
    start_slot: int
    duration: int
    source_bss: Optional[int] = None

    @property
    def end_slot(self):
        return self.start_slot + self.duration

class Channel:
    def __init__(self, 
                 channel_id: int, 
                 obss_generation_rate: float = 0.0, 
                 obss_duration_range: Tuple[int, int] = (20, 40)):
        self.channel_id = channel_id
        self.current_occupancy: Optional[Tuple[int, int]] = None
        self.obss_traffic: List[OBSSTraffic] = []
        # OBSS generation config
        self.obss_generation_rate = obss_generation_rate
        self.obss_duration_range = obss_duration_range

    def is_busy(self, slot: int) -> bool:
        return (self.current_occupancy is not None and self.current_occupancy[1] > slot) \
            or any(obss.start_slot <= slot < obss.end_slot for obss in self.obss_traffic)

    def is_busy_by_intra_bss(self, bss_id: int, slot: int) -> bool:
        return self.current_occupancy is not None and self.current_occupancy[1] > slot

    def is_busy_by_obss(self, slot: int) -> bool:
        return any(obss.start_slot <= slot < obss.end_slot for obss in self.obss_traffic)

    def occupy(self, slot: int, duration: int, sta_id: int):
        self.current_occupancy = (sta_id, slot + duration)

    def release(self, slot: int):
        if self.current_occupancy and self.current_occupancy[1] <= slot:
            self.current_occupancy = None

    def update(self, slot: int):
        self.obss_traffic = [obss for obss in self.obss_traffic if obss.end_slot > slot]
        self.release(slot)

    def add_obss_traffic(self, obss: OBSSTraffic):
        self.obss_traffic.append(obss)

    def get_latest_obss(self, slot: int) -> Optional[OBSSTraffic]:
        active = [obss for obss in self.obss_traffic if obss.start_slot <= slot < obss.end_slot]
        return active[0] if active else None
    
    def generate_obss(self, slot: int):
        """빈 채널일 때 확률적으로 OBSS 트래픽을 생성"""
        if (
            self.current_occupancy is None and
            not self.is_busy_by_obss(slot) and
            random.random() < self.obss_generation_rate
            ):
            duration = random.randint(*self.obss_duration_range)
            obss = OBSSTraffic(
                obss_id=f"obss_{self.channel_id}_{slot}",
                start_slot=slot,
                duration=duration
            )
            self.add_obss_traffic(obss)

class STA:
    def __init__(self, sta_id: int, channel_id: int, primary_channel: Channel, npca_channel: Optional[Channel] = None, npca_enabled: bool = False, radio_transition_time: int = 1):
        self.sta_id = sta_id
        self.channel_id = channel_id
        self.primary_channel = primary_channel
        self.npca_channel = npca_channel
        self.npca_enabled = npca_enabled
        self.radio_transition_time = radio_transition_time

        self.state = STAState.PRIMARY_BACKOFF
        self.backoff = self.generate_backoff()
        self.tx_remaining = 0
        self.ppdu_duration = 0
        self.current_obss: Optional[OBSSTraffic] = None

    def generate_backoff(self, cw_min: int = 15) -> int:
        return random.randint(0, cw_min)

    def get_tx_duration(self) -> int:
        return 10

    def step(self, slot: int):
        if self.state == STAState.PRIMARY_BACKOFF:
            self._handle_primary_backoff(slot)
        elif self.state == STAState.PRIMARY_FROZEN:
            if not self.primary_channel.is_busy(slot):
                self.state = STAState.PRIMARY_BACKOFF
        elif self.state == STAState.PRIMARY_TX:
            self._handle_primary_tx(slot)
        elif self.state == STAState.NPCA_BACKOFF:
            self._handle_npca_backoff(slot)
        elif self.state == STAState.NPCA_FROZEN:
            self._handle_npca_frozen(slot)
        elif self.state == STAState.NPCA_TX:
            self._handle_npca_tx(slot)

    def _handle_primary_backoff(self, slot: int):
        # (1) NPCA OFF인 경우: 어떤 busy든 무조건 frozen
        if not self.npca_enabled:
            if self.primary_channel.is_busy(slot):
                self.state = STAState.PRIMARY_FROZEN
            elif self.backoff == 0:
                self.tx_remaining = self.get_tx_duration()
                self.primary_channel.occupy(slot, self.tx_remaining, self.sta_id)
                self.state = STAState.PRIMARY_TX
            else:
                self.backoff -= 1
            return  # 종료

        # (2) NPCA ON인 경우
        if self.primary_channel.is_busy_by_intra_bss(self.channel_id, slot):
            self.state = STAState.PRIMARY_FROZEN

        elif self.primary_channel.is_busy_by_obss(slot):
            if self.npca_channel:
                self.current_obss = self.primary_channel.get_latest_obss(slot)
                self.backoff = self.generate_backoff()
                if self.npca_channel.is_busy(slot):
                    self.state = STAState.NPCA_FROZEN
                else:
                    self.state = STAState.NPCA_BACKOFF

        elif not self.primary_channel.is_busy(slot):
            if self.backoff == 0:
                self.tx_remaining = self.get_tx_duration()
                self.primary_channel.occupy(slot, self.tx_remaining, self.sta_id)
                self.state = STAState.PRIMARY_TX
            else:
                self.backoff -= 1


    def _handle_primary_tx(self, slot: int):
        if self.tx_remaining > 0:
            self.tx_remaining -= 1
        if self.tx_remaining == 0:
            self.state = STAState.PRIMARY_BACKOFF
            self.backoff = self.generate_backoff()

    def _handle_npca_backoff(self, slot: int):
        if self.npca_channel.is_busy_by_intra_bss(self.channel_id, slot):
            self.state = STAState.NPCA_FROZEN
        elif self.backoff > 0:
            self.backoff -= 1
        else:
            remaining = self.current_obss.end_slot - slot
            self.ppdu_duration = max(0, remaining - self.radio_transition_time)

            if self.ppdu_duration > 0:
                self.tx_remaining = self.ppdu_duration
                self.npca_channel.occupy(slot, self.tx_remaining, self.sta_id)

                # ✅ NPCA로 인해 primary 채널에도 OBSS 트래픽을 추가
                source_obss = OBSSTraffic(
                    obss_id=f"npca_source_{self.sta_id}_{slot}",
                    start_slot=slot,
                    duration=self.tx_remaining
                )
                self.primary_channel.add_obss_traffic(source_obss)

                self.state = STAState.NPCA_TX
            else:
                self.state = STAState.PRIMARY_BACKOFF
                self.backoff = self.generate_backoff()



    def _handle_npca_frozen(self, slot: int):
        if not self.npca_channel.is_busy(slot):
            self.state = STAState.NPCA_BACKOFF
        elif not self.primary_channel.is_busy(slot):
            self.state = STAState.PRIMARY_BACKOFF

    def _handle_npca_tx(self, slot: int):
        if self.tx_remaining > 0:
            self.tx_remaining -= 1
        if self.tx_remaining == 0:
            self.state = STAState.PRIMARY_BACKOFF
            self.backoff = self.generate_backoff()

# Simple simulator to test
class Simulator:
    def __init__(self, num_slots: int, stas: List[STA], channels: List[Channel]):
        self.num_slots = num_slots
        self.stas = stas
        self.channels = channels
        self.log = []

    def run(self):
        for slot in range(self.num_slots):
            # Generate OBSS traffic
            for ch in self.channels:
                ch.generate_obss(slot)

            # Update channels and STAs
            for ch in self.channels:
                ch.update(slot)
            
            # Step through each STA
            for sta in self.stas:
                sta.step(slot)
                # self.log.append({
                #     "slot": slot,
                #     "sta_id": sta.sta_id,
                #     "state": sta.state.name,
                #     "backoff": sta.backoff,
                #     "tx_remaining": sta.tx_remaining,
                #     "channel": sta.channel_id,
                # })
            self.log_slot(slot)

    def log_slot(self, slot: int):
        row = {
            "slot": slot,
            "time": slot * SLOTTIME,
        }

        for ch_id, ch in enumerate(self.channels):
            stas_in_ch = [sta for sta in self.stas if sta.channel_id == ch_id]

            row[f"states_ch_{ch_id}"] = [sta.state.name.lower() for sta in stas_in_ch]
            row[f"backoff_ch_{ch_id}"] = [sta.backoff for sta in stas_in_ch]
            row[f"npca_enabled_ch_{ch_id}"] = [sta.npca_enabled for sta in stas_in_ch]
            row[f"channel_{ch_id}_occupied_remained"] = (
                max(ch.current_occupancy[1] - slot, 0) if ch.current_occupancy else 0
                )
            row[f"channel_{ch_id}_obss_occupied_remained"] = (
                max((obss.end_slot - slot for obss in ch.obss_traffic), default=0)
                )

        self.log.append(row)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

