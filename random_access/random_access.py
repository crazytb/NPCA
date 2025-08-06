import pandas as pd
import random
from typing import List

# Copy-pasting the given code components into the environment to enable simulation
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import defaultdict

# Fix random seed for reproducibility
random.seed(42)

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

# @dataclass
# class OBSSTraffic:
#     obss_id: str
#     start_slot: int
#     duration: int
#     source_bss: Optional[int] = None
#     @property
#     def end_slot(self):
#         return self.start_slot + self.duration

@dataclass
class OccupyRequest:
    channel_id: int
    duration: int
    is_obss: bool = False

class Channel:
    def __init__(self, channel_id: int):
        self.channel_id = channel_id

        # intra-BSS 점유 상태
        self.intra_occupied = False
        self.intra_end_slot = -1
        self.occupied_by_sta = None

        # OBSS (inter-BSS) 트래픽 리스트 (튜플 기반)
        self.obss_traffic: List[Tuple[str, int, int, int]] = []  # (obss_id, start_slot, duration, source_bss)

    # ===============================
    # 점유 상태 관련 메서드
    # ===============================
    def is_busy_by_intra_bss(self, slot: int) -> bool:
        return self.intra_occupied and self.intra_end_slot > slot

    def is_busy_by_obss(self, slot: int) -> bool:
        return any(start <= slot < start + dur for _, start, dur, _ in self.obss_traffic)

    def is_busy(self, slot: int) -> bool:
        return self.is_busy_by_intra_bss(slot) or self.is_busy_by_obss(slot)

    def occupy_by_intra(self, slot: int, duration: int, sta_id: int):
        self.intra_occupied = True
        self.intra_end_slot = slot + duration
        self.occupied_by_sta = sta_id

    def release_intra(self, slot: int):
        if self.intra_occupied and self.intra_end_slot <= slot:
            self.intra_occupied = False
            self.occupied_by_sta = None

    # ===============================
    # OBSS 관련 메서드
    # ===============================
    def add_obss_traffic(self, obss_tuple: Tuple[str, int, int, int]):
        self.obss_traffic.append(obss_tuple)

    def get_latest_obss(self, slot: int) -> Optional[Tuple[str, int, int, int]]:
        for obss in self.obss_traffic:
            obss_id, start, dur, _ = obss
            if start <= slot < start + dur:
                return obss
        return None
    
    def generate_obss(self, slot: int):
        # Intra-BSS or OBSS 트래픽이 있으면 OBSS 새로 생성하지 않음
        if self.is_busy(slot):
            return

        # 확률적으로 OBSS 트래픽 생성
        if random.random() < self.obss_generation_rate:
            duration = random.randint(*self.obss_duration_range)
            obss_tuple = (
                f"obss_{self.channel_id}_{slot}",  # obss_id
                slot,                              # start_slot
                duration,                          # duration
                -1                                 # source_bss: 외부 OBSS
            )
            self.obss_traffic.append(obss_tuple)


    # ===============================
    # 슬롯별 상태 업데이트
    # ===============================
    def update(self, slot: int):
        self.release_intra(slot)
        self.obss_traffic = [obss for obss in self.obss_traffic if obss[1] + obss[2] > slot]


class STA:
    def __init__(self, sta_id: int, channel_id: int, primary_channel: Channel, npca_channel: Optional[Channel] = None, npca_enabled: bool = False, radio_transition_time: int = 1):
        self.sta_id = sta_id
        self.channel_id = channel_id
        self.primary_channel = primary_channel
        self.npca_channel = npca_channel
        self.npca_enabled = npca_enabled
        self.radio_transition_time = radio_transition_time
        self.occupy_request: Optional[OccupyRequest] = None

        self.state = STAState.PRIMARY_BACKOFF
        self.next_state = self.state
        self.cw_index = 0
        self.backoff = self.generate_backoff()
        self.tx_remaining = 0
        self.ppdu_duration = 0
        # self.current_obss: Optional[OBSSTraffic] = None
        self.intent = None

    def generate_backoff(self) -> int:
        cw = CONTENTION_WINDOW[self.cw_index]
        return random.randint(0, cw)
    
    def handle_collision(self):
        self.cw_index = min(self.cw_index + 1, len(CONTENTION_WINDOW) - 1)
        self.backoff = self.generate_backoff()
        self.tx_remaining = 0
        self.next_state = STAState.PRIMARY_BACKOFF

    def handle_success(self):
        self.cw_index = 0
        self.backoff = self.generate_backoff()
        self.next_state = STAState.PRIMARY_BACKOFF
    
    def decide_action(self, slot):
        self.intent = None
        if self.state == STAState.PRIMARY_BACKOFF and self.backoff == 0:
            self.intent = "primary_tx"
        return self.intent

    def get_tx_duration(self) -> int:
        return 10

    def step(self, slot: int):
        if self.state == STAState.PRIMARY_BACKOFF:
            self._handle_primary_backoff(slot)
        elif self.state == STAState.PRIMARY_FROZEN:
            self._handle_primary_frozen(slot)
        elif self.state == STAState.PRIMARY_TX:
            self._handle_primary_tx(slot)
        elif self.state == STAState.NPCA_BACKOFF:
            self._handle_npca_backoff(slot)
        elif self.state == STAState.NPCA_FROZEN:
            self._handle_npca_frozen(slot)
        elif self.state == STAState.NPCA_TX:
            self._handle_npca_tx(slot)

    def _handle_primary_backoff(self, slot: int):
        # 1. Primary 채널이 intra-BSS busy: frozen
        if self.primary_channel.is_busy_by_intra_bss(self.channel_id, slot):
            self.next_state = STAState.PRIMARY_FROZEN
            return
        # 2. Primary 채널이 OBSS busy: NPCA enabled 여부에 따라 다름
        if self.primary_channel.is_busy_by_obss(slot):
            # NPCA enabled인 경우
            if self.npca_enabled and self.npca_channel:
                self.current_obss = self.primary_channel.get_latest_obss(slot)
                self.backoff = self.generate_backoff()
                # NPCA 채널이 busy인지 확인
                if self.npca_channel.is_busy_by_intra_bss(self.npca_channel.channel_id, slot):
                    self.next_state = STAState.NPCA_FROZEN
                # NPCA 채널이 busy하지 않으면 backoff
                else:
                    self.next_state = STAState.NPCA_BACKOFF
        # 3. Primary 채널이 busy하지 않으면 전송 시도
        if not self.primary_channel.is_busy(slot):
            if self.backoff == 0:
                self.ppdu_duration = self.get_tx_duration()
                self.tx_remaining = self.ppdu_duration
                self.occupy_request = (self.primary_channel.channel_id, self.tx_remaining)
                self.next_state = STAState.PRIMARY_TX
            else:
                self.backoff -= 1

    def _handle_primary_frozen(self, slot: int):
        if not self.primary_channel.is_busy(slot):
            self.next_state = STAState.PRIMARY_BACKOFF

    def _handle_primary_tx(self, slot: int):
        # 전송 시작 시 occupy_request 설정 (딱 한 번)
        if self.tx_remaining == self.ppdu_duration:  # 시작 시점
            self.occupy_request = OccupyRequest(
                channel_id=self.primary_channel.channel_id, 
                duration=self.tx_remaining, 
                is_obss=False)

        # 전송 중
        if self.tx_remaining > 0:
            self.tx_remaining -= 1

        # 전송 종료 후
        if self.tx_remaining == 0:
            self.state = STAState.PRIMARY_BACKOFF
            self.backoff = self.generate_backoff()
            self.cw_index = 0

    def _handle_npca_backoff(self, slot: int):
        # 1. npca 채널이 intra-BSS로 busy → NPCA_FROZEN
        if self.npca_channel.is_busy_by_intra_bss(slot):
            self.state = STAState.NPCA_FROZEN
            return
        # 2. backoff countdown
        if self.backoff > 0:
            self.backoff -= 1
            return
        # 3. backoff == 0 → 전송 시도
        if self.current_obss is None:
            # OBSS duration이 사라졌다면 전송 불가 → stay in NPCA_BACKOFF
            self.state = STAState.PRIMARY_BACKOFF
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            return

        obss_start, obss_dur = self.current_obss[1], self.current_obss[2]
        obss_end = obss_start + obss_dur
        self.ppdu_duration = obss_end - slot

        if self.ppdu_duration <= 0:
            # OBSS duration이 끝났음 → stay in NPCA_BACKOFF
            return

        # 4. 전송 준비 완료 → occupy 대상은 원래 primary 채널 (e.g., channel 1의 STA → channel 0 점유)
        self.tx_remaining = self.ppdu_duration
        self.occupy_request = (self.npca_channel.channel_id, self.tx_remaining)
        self.state = STAState.NPCA_TX

    def _handle_npca_frozen(self, slot: int):
        # OBSS 정보가 더 이상 유효하지 않으면 primary로 복귀
        if self.current_obss is None:
            self.state = STAState.PRIMARY_BACKOFF
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            return

        obss_start, obss_dur = self.current_obss[1], self.current_obss[2]
        obss_end = obss_start + obss_dur

        # OBSS duration이 끝나면 primary로 복귀
        if slot >= obss_end:
            self.state = STAState.PRIMARY_BACKOFF
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            self.current_obss = None
            return

        # NPCA 채널이 idle → backoff 재개
        if not self.npca_channel.is_busy_by_intra_bss(slot):
            self.state = STAState.NPCA_BACKOFF


    def _handle_npca_tx(self, slot: int):
        # 전송 시작 시 OBSS 점유 요청 (딱 한 번)
        if self.tx_remaining == self.ppdu_duration:
            self.occupy_request = OccupyRequest(
                channel_id=self.npca_channel.channel_id,  # NPCA 채널 ID
                duration=self.tx_remaining,               # duration
                is_obss=True                              # OBSS 전송
            )

        if self.tx_remaining > 0:
            self.tx_remaining -= 1
            return

        if self.tx_remaining == 0:
            self.current_obss = None  # 전송 종료 → cleanup
            self.state = STAState.PRIMARY_BACKOFF
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            return


class Simulator:
    def __init__(self, num_slots: int, stas: List['STA'], channels: List['Channel']):
        self.num_slots = num_slots
        self.stas = stas
        self.channels = channels
        self.log = []

    def run(self):
        for slot in range(self.num_slots):
            # OBSS 트래픽 생성
            for ch in self.channels:
                ch.generate_obss(slot)

            # 채널 상태 업데이트
            for ch in self.channels:
                ch.update(slot)

            # STA들 동작 준비
            for sta in self.stas:
                sta.occupy_request = None
                sta.step(slot)

            # Occupy 요청 수집
            channel_requests = defaultdict(list)
            for sta in self.stas:
                if sta.occupy_request is not None:
                    req = sta.occupy_request
                    channel_requests[req.channel_id].append((sta, req))

            # Occupy 요청 처리 (채널별)
            for ch_id, reqs in channel_requests.items():
                if len(reqs) == 1:
                    sta, req = reqs[0]
                    if req.is_obss:
                        # NPCA 트래픽은 OBSS로 기록
                        obss_tuple = (
                            f"npca_sta_{sta.sta_id}_slot{slot}",
                            slot,
                            req.duration,
                            sta.channel_id  # original BSS
                        )
                        self.channels[ch_id].add_obss_traffic(obss_tuple)
                    else:
                        self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)

                    sta.tx_success = True
                else:
                    # 충돌 발생
                    for sta, _ in reqs:
                        sta.handle_collision()

            # 상태 전이 완료
            for sta in self.stas:
                sta.state = sta.next_state
                sta.occupy_request = None

            # 로그 기록
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

            # OBSS 점유 여부 (obss_traffic은 tuple로 관리됨)
            obss_remain = 0
            for obss_id, start, dur, _ in ch.obss_traffic:
                if start <= slot < start + dur:
                    obss_remain = start + dur - slot
                    break

            # 점유 남은 시간 기록
            row[f"channel_{ch_id}_occupied_remained"] = ch.current_occupancy[1] - slot if ch.current_occupancy else 0
            row[f"channel_{ch_id}_obss_occupied_remained"] = obss_remain

        self.log.append(row)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)
