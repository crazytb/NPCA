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
    def __init__(self, channel_id: int, obss_generation_rate: float = 0.0, obss_duration_range: Tuple[int, int] = (20, 40)):
        self.channel_id = channel_id
        self.obss_generation_rate = obss_generation_rate
        self.obss_duration_range = obss_duration_range

        # Intra-BSS 점유 상태
        self.intra_occupied = False
        self.intra_end_slot = 0

        # OBSS 트래픽 리스트: (obss_id, start_slot, duration, source_bss)
        self.obss_traffic: List[Tuple[str, int, int, int]] = []

    def occupy(self, slot: int, duration: int, sta_id: int):
        """STA가 채널을 점유함 (intra-BSS 점유)"""
        self.intra_occupied = True
        self.intra_end_slot = slot + duration

    def add_obss_traffic(self, req: OccupyRequest, slot: int):
        """NPCA 전송을 OBSS 트래픽으로 기록"""
        obss_tuple = (
            f"obss_gen_{self.channel_id}_slot{slot}",
            slot,
            req.duration,
            req.source_bss if hasattr(req, "source_bss") else -1  # fallback
        )
        self.obss_traffic.append(obss_tuple)

    def is_busy_by_intra_bss(self, slot: int) -> bool:
        return self.intra_occupied and self.intra_end_slot > slot

    def is_busy_by_obss(self, slot: int) -> bool:
        return any(start <= slot < start + dur for _, start, dur, _ in self.obss_traffic)

    def is_busy(self, slot: int) -> bool:
        return self.is_busy_by_intra_bss(slot) or self.is_busy_by_obss(slot)

    def update(self, slot: int):
        """슬롯마다 상태 갱신: 점유 만료 및 OBSS 제거"""
        if self.intra_occupied and self.intra_end_slot <= slot:
            self.intra_occupied = False

        self.obss_traffic = [
            t for t in self.obss_traffic if t[1] + t[2] > slot
        ]

    def generate_obss(self, slot: int):
        """OBSS 트래픽을 확률적으로 생성"""
        if self.obss_generation_rate == 0:
            return

        if not self.is_busy(slot):
            if random.random() < self.obss_generation_rate:
                duration = random.randint(*self.obss_duration_range)
                obss_tuple = (
                    f"obss_gen_{self.channel_id}_slot{slot}",
                    slot,
                    duration,
                    -1  # source_bss unknown
                )
                self.obss_traffic.append(obss_tuple)
                
    def get_latest_obss(self, slot: int) -> Optional[Tuple[str, int, int, int]]:
        """현재 slot에 유효한 OBSS 중 가장 최근에 시작된 것을 반환"""
        active = [
            obss for obss in self.obss_traffic
            if obss[1] <= slot < obss[1] + obss[2]  # start <= slot < end
        ]
        if not active:
            return None
        return max(active, key=lambda x: x[1])  # start_slot 기준으로 가장 최근



class STA:
    def __init__(self, 
                 sta_id: int, 
                 channel_id: int, 
                 primary_channel: Channel, 
                 npca_channel: Optional[Channel] = None, 
                 npca_enabled: bool = False, 
                 radio_transition_time: int = 1,
                 ppdu_duration: int = 10):
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
        self.ppdu_duration = ppdu_duration
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
        return self.ppdu_duration

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
        if self.primary_channel.is_busy_by_intra_bss(slot):
            self.next_state = STAState.PRIMARY_FROZEN
            return
        # 2. Primary 채널이 OBSS busy: NPCA enabled 여부에 따라 다름
        if self.primary_channel.is_busy_by_obss(slot):
            # NPCA enabled인 경우
            if self.npca_enabled and self.npca_channel:
                self.current_obss = self.primary_channel.get_latest_obss(slot)
                self.backoff = self.generate_backoff()
                # NPCA 채널이 busy인지 확인
                if self.npca_channel.is_busy_by_intra_bss(slot):
                    self.next_state = STAState.NPCA_FROZEN
                # NPCA 채널이 busy하지 않으면 backoff
                else:
                    self.next_state = STAState.NPCA_BACKOFF
            else:
                self.next_state = STAState.PRIMARY_FROZEN
            return
        # 3. Primary 채널이 busy하지 않으면 전송 시도
        if not self.primary_channel.is_busy(slot):
            if self.backoff == 0:
                self.ppdu_duration = self.get_tx_duration()
                self.tx_remaining = self.ppdu_duration
                self.occupy_request = OccupyRequest(
                    channel_id=self.primary_channel.channel_id, 
                    duration=self.tx_remaining, 
                    is_obss=False)
                self.next_state = STAState.PRIMARY_TX
            else:
                self.backoff -= 1
            return

    def _handle_primary_frozen(self, slot: int):
        if not self.primary_channel.is_busy(slot):
            self.next_state = STAState.PRIMARY_BACKOFF

    def _handle_primary_tx(self, slot: int):
        # 전송 시작 시 occupy_request 설정 (딱 한 번)
        # if self.tx_remaining == self.ppdu_duration:  # 시작 시점
        #     self.occupy_request = OccupyRequest(
        #         channel_id=self.primary_channel.channel_id, 
        #         duration=self.tx_remaining, 
        #         is_obss=False)

        # 전송 중
        if self.tx_remaining > 0:
            self.tx_remaining -= 1

        # 전송 종료 후
        if self.tx_remaining == 0:
            self.next_state = STAState.PRIMARY_BACKOFF
            self.backoff = self.generate_backoff()
            self.cw_index = 0

    def _handle_npca_backoff(self, slot: int):
        # 1. npca 채널이 intra-BSS로 busy → NPCA_FROZEN
        if self.npca_channel.is_busy_by_intra_bss(slot):
            self.next_state = STAState.NPCA_FROZEN
            return
        # 2. backoff countdown
        if self.backoff > 0:
            self.backoff -= 1
            return
        # 3. backoff == 0 → 전송 시도
        if self.current_obss is None:
            # OBSS duration이 사라졌다면 전송 불가 → stay in NPCA_BACKOFF
            self.next_state = STAState.PRIMARY_BACKOFF
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
        self.occupy_request = OccupyRequest(
                channel_id=self.npca_channel.channel_id,  # NPCA 채널 ID
                duration=self.tx_remaining,               # duration
                is_obss=True                              # OBSS 전송
            )
        self.next_state = STAState.NPCA_TX

    def _handle_npca_frozen(self, slot: int):
        # OBSS 정보가 더 이상 유효하지 않으면 primary로 복귀
        if self.current_obss is None:
            self.next_state = STAState.PRIMARY_BACKOFF
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            return

        obss_start, obss_dur = self.current_obss[1], self.current_obss[2]
        obss_end = obss_start + obss_dur

        # OBSS duration이 끝나면 primary로 복귀
        if slot >= obss_end:
            self.next_state = STAState.PRIMARY_BACKOFF
            self.cw_index = 0
            self.backoff = self.generate_backoff()
            self.current_obss = None
            return

        # NPCA 채널이 idle → backoff 재개
        if not self.npca_channel.is_busy_by_intra_bss(slot):
            self.next_state = STAState.NPCA_BACKOFF


    def _handle_npca_tx(self, slot: int):
        # 전송 시작 시 OBSS 점유 요청 (딱 한 번)
        # if self.tx_remaining == self.ppdu_duration:
        #     self.occupy_request = OccupyRequest(
        #         channel_id=self.npca_channel.channel_id,  # NPCA 채널 ID
        #         duration=self.tx_remaining,               # duration
        #         is_obss=True                              # OBSS 전송
        #     )

        if self.tx_remaining > 0:
            self.tx_remaining -= 1
            return

        if self.tx_remaining == 0:
            self.current_obss = None  # 전송 종료 → cleanup
            self.next_state = STAState.PRIMARY_BACKOFF
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
            # ① 채널 업데이트
            for ch in self.channels:
                ch.update(slot)

            # ② STA 상태 업데이트
            for sta in self.stas:
                sta.occupy_request = None
                sta.step(slot)

            # ③ 채널 OBSS request 수집
            obss_reqs = []
            for ch in self.channels:
                obss_req = ch.generate_obss(slot)
                if obss_req:
                    obss_reqs.append((None, obss_req))

            # ④ STA 전송 요청 수집
            sta_reqs = [(sta, sta.occupy_request) for sta in self.stas if sta.occupy_request is not None]

            # ⑤ 전체 요청 통합
            all_reqs = sta_reqs + obss_reqs

            # ⑥ 채널별로 OccupyRequest 분류
            channel_requests = defaultdict(list)
            for sta, req in all_reqs:
                channel_requests[req.channel_id].append((sta, req))

            # ⑦ Occupy 요청 처리
            for ch_id, reqs in channel_requests.items():
                if len(reqs) == 1:
                    sta, req = reqs[0]
                    if req.is_obss:
                        self.channels[ch_id].add_obss_traffic(req, slot)
                    else:
                        self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                    if sta:
                        sta.tx_success = True
                else:
                    for sta, _ in reqs:
                        if sta:
                            sta.handle_collision()

            # ⑧ 상태 전이 및 초기화
            for sta in self.stas:
                sta.state = sta.next_state
                sta.occupy_request = None

            # ⑨ 로그 저장
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

            # OBSS 점유 시간
            obss_remain = 0
            for _, start, dur, _ in ch.obss_traffic:
                if start <= slot < start + dur:
                    obss_remain = start + dur - slot
                    break

            # intra-BSS 점유 시간
            occupied_remain = ch.intra_end_slot - slot if ch.intra_occupied else 0

            row[f"channel_{ch_id}_occupied_remained"] = occupied_remain
            row[f"channel_{ch_id}_obss_occupied_remained"] = obss_remain

        self.log.append(row)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

