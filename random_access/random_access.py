import numpy as np
import pandas as pd
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Constants
CONTENTION_WINDOW = [2 ** (i + 4) - 1 for i in range(7)]  # CW from 15 to 1023
SLOTTIME = 9  # μs

class STAState(Enum):
    """Simplified Station states in CSMA/CA FSM"""
    IDLE = "idle"
    BACKOFF = "backoff"
    BACKOFF_FROZEN = "backoff_frozen"
    OBSS_FROZEN = "obss_frozen"
    NPCA_BACKOFF = "npca_backoff"
    NPCA_BACKOFF_FROZEN = "npca_backoff_frozen"  # ✅ 새로 추가
    PRIMARY_TRANSMITTING = "primary_transmitting"
    NPCA_TRANSMITTING = "npca_transmitting"

@dataclass
class FrameInfo:
    """Frame information for transmission"""
    frame_id: int
    source: int
    size: int  # in slots
    timestamp: int  # When frame was originally created
    creation_slot: int  # Slot when frame was created for AoI calculation

@dataclass
class OBSSTraffic:
    """OBSS traffic information"""
    obss_id: int
    start_slot: int
    duration: int  # in slots
    source_channel: int  # Which channel this OBSS traffic originates from
    
@dataclass
class PendingOBSSTraffic:
    """Pending OBSS traffic information"""
    obss_id: int
    duration: int
    source_channel: int
    creation_slot: int

class OBSSGenerator:
    """OBSS traffic generator with backoff mechanism"""
    
    def __init__(self, source_channel: int, generation_rate: float, frame_size_range: Tuple[int, int]):
        self.source_channel = source_channel
        self.generation_rate = generation_rate
        self.frame_size_range = frame_size_range
        
        # OBSS backoff state
        self.pending_traffic = None
        self.backoff_counter = 0
        self.obss_id_counter = 0
        
        # Statistics
        self.obss_generated = 0
        self.obss_deferred = 0
        self.obss_blocked_by_intra_bss = 0
        self.obss_blocked_by_other_obss = 0
    
    def attempt_generation(self, current_slot: int, intra_bss_busy: bool, other_obss_busy: bool) -> Optional[OBSSTraffic]:
        """Attempt to generate OBSS traffic with backoff mechanism"""
    
        channel_busy = intra_bss_busy or other_obss_busy
        
        # === Phase 1: Handle pending OBSS traffic ===
        if self.pending_traffic:
            if channel_busy:
                # Channel busy - freeze backoff and count deferrals
                if intra_bss_busy:
                    self.obss_blocked_by_intra_bss += 1
                if other_obss_busy:
                    self.obss_blocked_by_other_obss += 1
                self.obss_deferred += 1
                return None
            else:
                # Channel clear - proceed with backoff
                if self.backoff_counter > 0:
                    self.backoff_counter -= 1
                    return None
                else:
                    # Backoff completed - generate OBSS traffic
                    traffic = OBSSTraffic(
                        obss_id=self.pending_traffic.obss_id,
                        start_slot=current_slot,
                        duration=self.pending_traffic.duration,
                        source_channel=self.pending_traffic.source_channel
                    )
                    self.pending_traffic = None
                    self.obss_generated += 1
                    return traffic
        
        # === Phase 2: Attempt new OBSS traffic generation ===
        if np.random.random() < self.generation_rate:
            obss_size = np.random.randint(self.frame_size_range[0], self.frame_size_range[1] + 1)
            
            if channel_busy:
                # Channel busy - create pending traffic and start backoff
                self.pending_traffic = PendingOBSSTraffic(
                    obss_id=self.obss_id_counter,
                    duration=obss_size,
                    source_channel=self.source_channel,
                    creation_slot=current_slot
                )
                self.obss_id_counter += 1
                self.backoff_counter = np.random.randint(0, 32)  # Simple backoff window
                
                if intra_bss_busy:
                    self.obss_blocked_by_intra_bss += 1
                if other_obss_busy:
                    self.obss_blocked_by_other_obss += 1
                self.obss_deferred += 1
                return None
            else:
                # Channel clear - generate immediately
                traffic = OBSSTraffic(
                    obss_id=self.obss_id_counter,
                    start_slot=current_slot,
                    duration=obss_size,
                    source_channel=self.source_channel
                )
                self.obss_id_counter += 1
                self.obss_generated += 1
                return traffic
        
        return None

class STAFiniteStateMachine:
    """Simplified 802.11 CSMA/CA Station implemented as FSM with NPCA support"""
    
    def __init__(self, sta_id: int, channel_id: int, npca_enabled: bool = False):
        self.sta_id = sta_id
        self.channel_id = channel_id
        self.npca_enabled = npca_enabled
        
        # FSM State
        self.state = STAState.IDLE
        
        # CSMA/CA Parameters
        self.backoff_stage = 0
        self.backoff_counter = 0
        self.max_retries = 7
        
        # Frame handling
        self.tx_queue = []
        self.current_frame = None
        self.transmitting_until = -1
        
        # NPCA 관련 변수들
        self.npca_target_channels = []
        self.npca_transmission_channel = None
        self.npca_max_duration = 0
        
        # NPCA 통계
        self.npca_attempts = 0
        self.npca_successful = 0
        self.npca_blocked = 0
        
        # AoI tracking
        self.frame_creation_slot = 0
        self.last_successful_tx_slot = 0
        
        # Statistics
        self.successful_transmissions = 0
        self.collision_count = 0
        self.total_attempts = 0
        self.obss_deferrals = 0
        self.intra_bss_deferrals = 0
        
        # Flags
        self.has_frame_to_send = False
        self.tx_attempt = False
    
    def set_npca_channels(self, available_channels: List[int]):
        """NPCA로 접근 가능한 채널 목록 설정 (자신의 primary 채널 제외)"""
        if self.npca_enabled:
            self.npca_target_channels = [ch for ch in available_channels if ch != self.channel_id]
        else:
            self.npca_target_channels = []

    def get_new_backoff(self) -> int:
        """Generate new backoff value based on current stage"""
        cw_index = min(self.backoff_stage, len(CONTENTION_WINDOW) - 1)
        cw = CONTENTION_WINDOW[cw_index]
        return np.random.randint(0, cw + 1)
    
    def add_frame(self, frame: FrameInfo):
        """Add frame to transmission queue"""
        self.tx_queue.append(frame)
        self.has_frame_to_send = True
    
    def update(self, current_slot: int, channel_status: Dict[int, Dict], obss_occupied_until: Dict[int, int]) -> Tuple[bool, int]:
        """Update FSM state"""
        
        self.tx_attempt = False
        target_channel = self.channel_id
        
        # Skip update if currently transmitting and transmission not finished
        if self.state in [STAState.PRIMARY_TRANSMITTING, STAState.NPCA_TRANSMITTING] and \
        self.transmitting_until > current_slot:
            return False, target_channel
        
        # Primary 채널 상태 가져오기
        primary_status = channel_status.get(self.channel_id, {
            'intra_busy': False, 
            'obss_busy': False, 
            'any_busy': False
        })
        
        primary_intra_busy = primary_status['intra_busy']
        primary_obss_busy = primary_status['obss_busy']
        primary_any_busy = primary_intra_busy or primary_obss_busy
        
        # 지연 통계 카운팅
        if self.state in [STAState.BACKOFF, STAState.BACKOFF_FROZEN, STAState.OBSS_FROZEN, 
                         STAState.NPCA_BACKOFF, STAState.NPCA_BACKOFF_FROZEN]:
            if primary_intra_busy:
                self.intra_bss_deferrals += 1
            elif primary_obss_busy:
                self.obss_deferrals += 1
                
        # Main FSM logic
        if self.state == STAState.IDLE:
            self._handle_idle_state(current_slot)
            
        elif self.state == STAState.BACKOFF:
            target_channel = self._handle_backoff_with_npca_exception(
                primary_intra_busy, primary_obss_busy, primary_any_busy,
                current_slot, channel_status, obss_occupied_until)
            
        elif self.state == STAState.BACKOFF_FROZEN:
            self._handle_backoff_frozen_with_npca_exception(
                primary_intra_busy, primary_obss_busy, primary_any_busy,
                current_slot, channel_status, obss_occupied_until)
            
        elif self.state == STAState.OBSS_FROZEN:
            target_channel = self._handle_obss_frozen(
                primary_intra_busy, primary_obss_busy, 
                current_slot, channel_status, obss_occupied_until)
            
        elif self.state == STAState.NPCA_BACKOFF:
            target_channel = self._handle_npca_backoff(current_slot, channel_status, obss_occupied_until)
            
        elif self.state == STAState.NPCA_BACKOFF_FROZEN:
            target_channel = self._handle_npca_backoff_frozen(current_slot, channel_status, obss_occupied_until)
            
        elif self.state in [STAState.PRIMARY_TRANSMITTING, STAState.NPCA_TRANSMITTING]:
            self._handle_transmitting(current_slot)
        
        return self.tx_attempt, target_channel
    
    def _handle_idle_state(self, current_slot: int):
        """Handle IDLE state"""
        if self.has_frame_to_send and self.tx_queue:
            self.current_frame = self.tx_queue.pop(0)
            self.has_frame_to_send = len(self.tx_queue) > 0
            self.frame_creation_slot = self.current_frame.creation_slot
            self.backoff_counter = self.get_new_backoff()
            self.state = STAState.BACKOFF
    
    def _handle_backoff_with_npca_exception(self, primary_intra_busy: bool, primary_obss_busy: bool, 
                                       primary_any_busy: bool, current_slot: int,
                                       channel_status: Dict[int, Dict], obss_occupied_until: Dict[int, int]) -> int:
        """Handle BACKOFF state with NPCA exception"""
        
        if primary_any_busy:
            # NPCA 예외 조건 확인
            if (self.npca_enabled and 
                not primary_intra_busy and 
                primary_obss_busy and 
                self._has_idle_target_channel(channel_status)):
                
                # ✅ 직접 NPCA 백오프로 전환
                return self._setup_npca_and_start_backoff(current_slot, channel_status, obss_occupied_until)
            else:
                # 일반적인 경우
                self.state = STAState.BACKOFF_FROZEN
                return self.channel_id
        else:
            # Primary 채널이 idle → 백오프 계속
            self._continue_backoff()
            return self.channel_id

    def _handle_backoff_frozen_with_npca_exception(self, primary_intra_busy: bool, primary_obss_busy: bool,
                                              primary_any_busy: bool, current_slot: int,
                                              channel_status: Dict[int, Dict], obss_occupied_until: Dict[int, int]):
        """Handle BACKOFF_FROZEN state with NPCA exception"""
        
        if not primary_any_busy:
            # Primary 채널이 idle → 백오프 재개
            self.state = STAState.BACKOFF
            return
        
        # Primary 채널이 여전히 busy
        if (self.npca_enabled and 
            not primary_intra_busy and 
            primary_obss_busy and 
            self._has_idle_target_channel(channel_status)):
            
            # NPCA 조건 만족 → NPCA 백오프로 전환
            self._setup_npca_and_start_backoff(current_slot, channel_status, obss_occupied_until)
        # else: BACKOFF_FROZEN 상태 유지

    def _setup_npca_and_start_backoff(self, current_slot: int, channel_status: Dict[int, Dict], 
                                     obss_occupied_until: Dict[int, int]) -> int:
        """NPCA 파라미터 설정 및 NPCA_BACKOFF 상태로 직접 전환"""
        
        # 1. OBSS 지속 시간 확인
        primary_obss_until = obss_occupied_until.get(self.channel_id, current_slot)
        remaining_obss_slots = max(0, primary_obss_until - current_slot)
        
        min_npca_duration = 5
        if remaining_obss_slots < min_npca_duration:
            self.state = STAState.BACKOFF_FROZEN
            return self.channel_id
        
        # 2. 타겟 채널 선택
        available_npca_channels = []
        for target_ch in self.npca_target_channels:
            # ✅ 수정: 기본값을 idle로 설정
            target_status = channel_status.get(target_ch, {
                'intra_busy': False, 
                'obss_busy': False, 
                'any_busy': False
            })
            if not target_status['any_busy']:
                available_npca_channels.append(target_ch)
        
        if not available_npca_channels:
            self.npca_blocked += 1
            self.state = STAState.BACKOFF_FROZEN
            return self.channel_id
        
        # 3. NPCA 파라미터 설정
        selected_npca_channel = available_npca_channels[0]
        self.npca_transmission_channel = selected_npca_channel
        self.npca_max_duration = remaining_obss_slots
        self.npca_attempts += 1
        
        # ✅ 직접 NPCA_BACKOFF 상태로 전환
        self.state = STAState.NPCA_BACKOFF
        
        return self.channel_id

    def _handle_obss_frozen(self, primary_intra_busy: bool, primary_obss_busy: bool,
                           current_slot: int, channel_status: Dict[int, Dict], 
                           obss_occupied_until: Dict[int, int]) -> int:
        """Handle OBSS_FROZEN state"""
        
        if primary_intra_busy:
            self.state = STAState.BACKOFF_FROZEN
            return self.channel_id
            
        elif not primary_obss_busy:
            self.state = STAState.BACKOFF
            return self.channel_id
            
        else:
            # Primary에 OBSS만 있는 상황 → NPCA 시도
            if self._has_idle_target_channel(channel_status):
                return self._setup_npca_and_start_backoff(current_slot, channel_status, obss_occupied_until)
            else:
                self.state = STAState.BACKOFF_FROZEN
                return self.channel_id
    
    def _handle_npca_backoff(self, current_slot: int, channel_status: Dict[int, Dict], 
                            obss_occupied_until: Dict[int, int]) -> int:
        """Handle NPCA_BACKOFF state with freeze mechanism"""
        
        # 1. Primary 채널의 OBSS 상태 확인 (NPCA 유지 조건)
        primary_status = channel_status.get(self.channel_id, {
            'obss_busy': False, 
            'intra_busy': False
        })
        
        if primary_status['intra_busy']:
            # Primary에 intra-BSS 트래픽 발생 - NPCA 포기
            self.state = STAState.BACKOFF_FROZEN
            self._reset_npca_params()
            return self.channel_id
        
        if not primary_status['obss_busy']:
            # Primary의 OBSS가 사라짐 - NPCA 포기
            self.state = STAState.BACKOFF
            self._reset_npca_params()
            return self.channel_id
        
        # 2. NPCA target 채널 상태 확인
        if self.npca_transmission_channel is None:
            self.state = STAState.OBSS_FROZEN
            return self.channel_id
        
        # ✅ 수정: 기본값을 idle로 설정
        target_status = channel_status.get(self.npca_transmission_channel, {
            'intra_busy': False, 
            'obss_busy': False, 
            'any_busy': False
        })
        
        # ✅ NPCA 채널이 busy하면 NPCA_BACKOFF_FROZEN으로 전환
        if target_status['any_busy']:
            self.state = STAState.NPCA_BACKOFF_FROZEN
            return self.channel_id
        
        # 3. NPCA 전송 가능 시간 재확인
        primary_obss_until = obss_occupied_until.get(self.channel_id, current_slot)
        remaining_obss_slots = max(0, primary_obss_until - current_slot)
        
        if remaining_obss_slots < 2:
            self.state = STAState.BACKOFF
            self._reset_npca_params()
            return self.channel_id
        
        # 4. ✅ NPCA 채널이 idle하면 백오프 카운터 진행
        if self.backoff_counter == 0:
            # 백오프 완료 - NPCA 전송 시도
            self.npca_max_duration = remaining_obss_slots
            self.state = STAState.NPCA_TRANSMITTING
            self.tx_attempt = True
            self.total_attempts += 1
            return self.npca_transmission_channel
        else:
            # 백오프 카운터 감소
            self.backoff_counter -= 1
            if self.backoff_counter == 0:
                self.npca_max_duration = remaining_obss_slots
                self.state = STAState.NPCA_TRANSMITTING
                self.tx_attempt = True
                self.total_attempts += 1
                return self.npca_transmission_channel
            return self.channel_id

    def _handle_npca_backoff_frozen(self, current_slot: int, channel_status: Dict[int, Dict], 
                                   obss_occupied_until: Dict[int, int]) -> int:
        """Handle NPCA_BACKOFF_FROZEN state"""
        
        # 1. Primary 채널의 OBSS 상태 확인
        primary_status = channel_status.get(self.channel_id, {
            'obss_busy': False, 
            'intra_busy': False
        })
        
        if primary_status['intra_busy']:
            # Primary에 intra-BSS 트래픽 발생 - NPCA 포기
            self.state = STAState.BACKOFF_FROZEN
            self._reset_npca_params()
            return self.channel_id
        
        if not primary_status['obss_busy']:
            # Primary의 OBSS가 사라짐 - NPCA 포기
            self.state = STAState.BACKOFF
            self._reset_npca_params()
            return self.channel_id
        
        # 2. NPCA target 채널 상태 확인
        if self.npca_transmission_channel is None:
            self.state = STAState.OBSS_FROZEN
            return self.channel_id
        
        # ✅ 수정: 기본값을 idle로 설정
        target_status = channel_status.get(self.npca_transmission_channel, {
            'intra_busy': False, 
            'obss_busy': False, 
            'any_busy': False
        })
        
        # ✅ NPCA 채널이 idle해지면 NPCA_BACKOFF로 복귀
        if not target_status['any_busy']:
            # NPCA 전송 가능 시간 재확인
            primary_obss_until = obss_occupied_until.get(self.channel_id, current_slot)
            remaining_obss_slots = max(0, primary_obss_until - current_slot)
            
            if remaining_obss_slots >= 2:
                # 충분한 시간 있음 - NPCA 백오프 재개
                self.state = STAState.NPCA_BACKOFF
            else:
                # 시간 부족 - 일반 백오프로 복귀
                self.state = STAState.BACKOFF
                self._reset_npca_params()
            
            return self.channel_id
        
        # ✅ NPCA 채널이 여전히 busy - FROZEN 상태 유지
        return self.channel_id
    
    def _handle_transmitting(self, current_slot: int):
        """Handle both PRIMARY_TRANSMITTING and NPCA_TRANSMITTING states"""
        if self.transmitting_until == -1:
            if self.state == STAState.NPCA_TRANSMITTING and self.npca_max_duration > 0:
                actual_duration = min(self.current_frame.size, self.npca_max_duration)
                self.transmitting_until = current_slot + actual_duration
            else:
                self.transmitting_until = current_slot + self.current_frame.size
        
        if current_slot >= self.transmitting_until:
            # 전송 완료 - 결과는 on_transmission_result에서 처리
            pass
    
    def _continue_backoff(self):
        """백오프 카운터 감소 및 전송 시도"""
        if self.backoff_counter == 0:
            self.state = STAState.PRIMARY_TRANSMITTING
            self.tx_attempt = True
            self.total_attempts += 1
        else:
            self.backoff_counter -= 1
            if self.backoff_counter == 0:
                self.state = STAState.PRIMARY_TRANSMITTING
                self.tx_attempt = True
                self.total_attempts += 1
        
    def _has_idle_target_channel(self, channel_status: Dict[int, Dict]) -> bool:
        """타겟 채널 중 idle한 채널이 있는지 확인"""
        if not self.npca_enabled or not self.npca_target_channels:
            return False
        
        for target_ch in self.npca_target_channels:
            # ✅ 수정: 기본값을 idle로 설정
            target_status = channel_status.get(target_ch, {
                'intra_busy': False, 
                'obss_busy': False, 
                'any_busy': False
            })
            if not target_status['any_busy']:
                return True
        
        return False
    
    def _reset_npca_params(self):
        """NPCA 관련 파라미터 리셋"""
        self.npca_transmission_channel = None
        self.npca_max_duration = 0
        self.transmitting_until = -1

    def _reset_transmission_params(self, keep_frame: bool = False):
        """Reset transmission parameters"""
        if not keep_frame:
            self.backoff_stage = 0
            self.current_frame = None
            self.frame_creation_slot = 0
        
        self.npca_transmission_channel = None
        self.npca_max_duration = 0
        self.transmitting_until = -1
        self.tx_attempt = False
    
    def on_transmission_result(self, result: str, completion_slot: int):
        """Handle transmission result from channel"""
        if self.state not in [STAState.PRIMARY_TRANSMITTING, STAState.NPCA_TRANSMITTING]:
            return
            
        if result == 'success':
            self.state = STAState.IDLE
            self.successful_transmissions += 1
            
            # NPCA 성공 통계
            if self.state == STAState.NPCA_TRANSMITTING:
                self.npca_successful += 1
                
            self.last_successful_tx_slot = completion_slot
            self._reset_transmission_params()
        elif result == 'collision':
            self.backoff_stage = min(self.backoff_stage + 1, len(CONTENTION_WINDOW) - 1)
            self.backoff_counter = self.get_new_backoff()
            self.collision_count += 1
            self.state = STAState.BACKOFF
            self._reset_transmission_params(keep_frame=True)

    def get_current_aoi(self, current_slot: int) -> int:
        """Calculate current Age of Information in slots"""
        if self.current_frame is None:
            return current_slot - self.last_successful_tx_slot
        else:
            return current_slot - self.frame_creation_slot


class ChannelFSM:
    """Simplified Channel state machine for CSMA/CA with OBSS support"""
    
    def __init__(self, channel_id: int):
        self.channel_id = channel_id
        self.transmitting_stations = []
        self.occupied_until = -1
        self.current_frame = None
        self.pending_results = []  # Store pending transmission results
        
        # OBSS traffic management
        self.obss_traffic = []  # List of active OBSS transmissions
        self.obss_occupied_until = -1  # When OBSS traffic ends
        
    def update(self, current_slot: int):
        """Update channel state and return completed transmission results"""
        results = []
        
        # Update OBSS traffic
        self._update_obss_traffic(current_slot)
        
        # Check if transmission completed
        if current_slot >= self.occupied_until and self.occupied_until != -1:
            # Transmission completed - return results with completion slot
            results = [(sta_id, result, current_slot) for sta_id, result in self.pending_results]
            self.pending_results.clear()
            self.transmitting_stations.clear()
            self.current_frame = None
            self.occupied_until = -1
        
        return results
    
    def _update_obss_traffic(self, current_slot: int):
        """Update OBSS traffic status"""
        # Remove expired OBSS traffic
        self.obss_traffic = [obss for obss in self.obss_traffic 
                            if current_slot < obss.start_slot + obss.duration]
        
        # Update OBSS occupied until time
        if self.obss_traffic:
            self.obss_occupied_until = max(obss.start_slot + obss.duration 
                                          for obss in self.obss_traffic)
        else:
            self.obss_occupied_until = -1
    
    def add_obss_traffic(self, obss_traffic: OBSSTraffic):
        """Add OBSS traffic to this channel"""
        self.obss_traffic.append(obss_traffic)
        self.obss_occupied_until = max(self.obss_occupied_until, 
                                      obss_traffic.start_slot + obss_traffic.duration)
    
    def add_transmission(self, sta_id: int, frame: FrameInfo):
        """Add transmission attempt"""
        self.transmitting_stations.append((sta_id, frame))
    
    def resolve_access(self, current_slot: int):
        """Resolve channel access and schedule results"""
        if len(self.transmitting_stations) == 0:
            return
        
        # Only process if channel is not already occupied by intra-BSS traffic
        if current_slot >= self.occupied_until:
            if len(self.transmitting_stations) == 1:
                # Single transmission - will be successful
                sta_id, frame = self.transmitting_stations[0]
                self.pending_results.append((sta_id, 'success'))
                self.occupied_until = current_slot + frame.size
                self.current_frame = frame
            else:
                # Multiple transmissions - collision
                max_duration = max(frame.size for _, frame in self.transmitting_stations)
                for sta_id, frame in self.transmitting_stations:
                    self.pending_results.append((sta_id, 'collision'))
                self.occupied_until = current_slot + max_duration
        
        # Clear transmission attempts after resolving
        self.transmitting_stations.clear()
    
    def is_busy(self, current_slot: int) -> bool:
        """Check if channel is busy due to intra-BSS traffic"""
        return current_slot < self.occupied_until
    
    def is_obss_busy(self, current_slot: int) -> bool:
        """Check if channel is busy due to OBSS traffic"""
        return current_slot < self.obss_occupied_until
    
    def is_any_busy(self, current_slot: int) -> bool:
        """Check if channel is busy due to any traffic (intra-BSS or OBSS)"""
        return self.is_busy(current_slot) or self.is_obss_busy(current_slot)        
        
class SimplifiedCSMACASimulation:
    """Simplified CSMA/CA Network Simulation with mutual OBSS interference"""
    
    def __init__(self, num_channels: int, stas_per_channel: List[int], 
                 simulation_time: int, frame_size: int, 
                 obss_enabled_per_channel: List[bool] = None,
                 npca_enabled: List[bool] = None,
                 obss_generation_rate: float = 0.001, 
                 obss_frame_size_range: Tuple[int, int] = (20, 500)):
        
        self.num_channels = num_channels
        self.stas_per_channel = stas_per_channel
        self.simulation_time = simulation_time
        self.frame_size = frame_size
        
        # 채널별 OBSS 설정 처리
        if obss_enabled_per_channel is None:
            self.obss_enabled_per_channel = [False] * num_channels
        else:
            assert len(obss_enabled_per_channel) == num_channels, \
                f"obss_enabled_per_channel length ({len(obss_enabled_per_channel)}) must match num_channels ({num_channels})"
            self.obss_enabled_per_channel = obss_enabled_per_channel
        
        # 채널별 NPCA 설정 처리
        if npca_enabled is None:
            self.npca_enabled = [False] * num_channels
        else:
            assert len(npca_enabled) == num_channels, \
                f"npca_enabled length ({len(npca_enabled)}) must match num_channels ({num_channels})"
            self.npca_enabled = npca_enabled
        
        self.obss_enabled = any(self.obss_enabled_per_channel)
        self.obss_generation_rate = obss_generation_rate
        self.obss_frame_size_range = obss_frame_size_range
        
        # Initialize channels
        self.channels = [ChannelFSM(i) for i in range(num_channels)]
        
        # Initialize stations with NPCA support
        self.stations = []
        sta_id = 0
        for ch_id in range(num_channels):
            for _ in range(stas_per_channel[ch_id]):
                # 해당 채널에서 NPCA 활성화 여부 확인
                npca_enabled_for_sta = self.npca_enabled[ch_id]
                sta = STAFiniteStateMachine(sta_id, ch_id, npca_enabled_for_sta)
                
                # NPCA 채널 설정
                if npca_enabled_for_sta:
                    available_channels = list(range(num_channels))
                    sta.set_npca_channels(available_channels)
                
                self.stations.append(sta)
                sta_id += 1
        
        # Initialize OBSS generators
        self.obss_generators = []
        for ch_id in range(num_channels):
            if self.obss_enabled_per_channel[ch_id]:
                generator = OBSSGenerator(
                    source_channel=ch_id,
                    generation_rate=obss_generation_rate,
                    frame_size_range=obss_frame_size_range
                )
                self.obss_generators.append(generator)
            else:
                self.obss_generators.append(None)
        
        # Simulation state
        self.current_slot = 0
        self.logs = []
        self.frame_counter = 0
        
        # Generate initial frames for all stations
        self._generate_initial_frames()
    
    def _generate_initial_frames(self):
        """Generate initial frames for all stations"""
        for sta in self.stations:
            frame = FrameInfo(
                frame_id=self.frame_counter,
                source=sta.sta_id,
                size=self.frame_size,
                timestamp=0,
                creation_slot=0
            )
            self.frame_counter += 1
            sta.add_frame(frame)
    
    def _get_affected_channels(self, source_channel: int) -> List[int]:
        """Each channel has independent OBSS - no cross-channel interference"""
        return [source_channel]
    
    def _generate_obss_traffic(self, current_slot: int):
        """Generate OBSS traffic with mutual interference consideration"""
        if not self.obss_enabled:
            return
        
        # Generate OBSS traffic for each channel (활성화된 채널만)
        for ch_id, generator in enumerate(self.obss_generators):
            # 해당 채널에서 OBSS가 활성화되지 않았으면 스킵
            if generator is None or not self.obss_enabled_per_channel[ch_id]:
                continue
                
            # Check channel status for OBSS generation
            intra_bss_busy = self.channels[ch_id].is_busy(current_slot)
            other_obss_busy = self.channels[ch_id].is_obss_busy(current_slot)
            
            # Attempt OBSS generation (with backoff if channel busy)
            obss_traffic = generator.attempt_generation(current_slot, intra_bss_busy, other_obss_busy)
            
            if obss_traffic:
                # Add OBSS traffic to affected channels
                affected_channels = self._get_affected_channels(ch_id)
                for target_ch in affected_channels:
                    if 0 <= target_ch < self.num_channels:
                        self.channels[target_ch].add_obss_traffic(obss_traffic)
    
    def run(self) -> pd.DataFrame:
        """Run the simulation"""
        for self.current_slot in range(self.simulation_time):
            # print(f"Slot: {self.current_slot}/{self.simulation_time - 1}")
            self._tick()
        
        return pd.DataFrame(self.logs)
    
    def _tick(self):
        """One simulation tick with NPCA support - NPCA 전송 시 원래 채널도 busy 처리"""
        # Generate OBSS traffic
        self._generate_obss_traffic(self.current_slot)
        
        # Update channels and get completed transmission results
        completed_results = {}
        for channel in self.channels:
            results = channel.update(self.current_slot)
            if results:
                completed_results[channel.channel_id] = results
        
        # Process completed transmission results
        for ch_id, results in completed_results.items():
            for sta_id, result, completion_slot in results:
                sta = self.stations[sta_id]
                sta.on_transmission_result(result, completion_slot)
                
                # Generate new frame after successful transmission
                if result == 'success':
                    self._generate_new_frame(sta, completion_slot)
        
        # ✨ 1단계: 모든 STA의 전송 시도를 수집
        transmission_attempts = []
        npca_attempts = []
        
        # 임시 채널 상태 (NPCA 전송 반영 전)
        temp_channel_status = {}
        temp_obss_occupied_until = {}
        
        for ch_id in range(self.num_channels):
            channel = self.channels[ch_id]
            temp_channel_status[ch_id] = {
                'intra_busy': channel.is_busy(self.current_slot),
                'obss_busy': channel.is_obss_busy(self.current_slot),
                'any_busy': channel.is_any_busy(self.current_slot)
            }
            temp_obss_occupied_until[ch_id] = channel.obss_occupied_until
        
        # STA 업데이트 (NPCA 전송 반영 전 상태 기준)
        for sta in self.stations:
            tx_attempt, target_channel = sta.update(self.current_slot, temp_channel_status, temp_obss_occupied_until)
            
            if tx_attempt and sta.current_frame:
                if target_channel != sta.channel_id:
                    # NPCA 전송
                    npca_attempts.append((sta, target_channel))
                else:
                    # 일반 전송
                    transmission_attempts.append((sta, target_channel))
        
        # ✨ 2단계: NPCA 전송 처리 및 원래 채널에 OBSS 추가
        valid_npca_stas = []
        
        for sta, target_channel in npca_attempts:
            # 타겟 채널에 intra-BSS 전송이 있는지 확인
            target_has_intra = any(target_ch == target_channel for _, target_ch in transmission_attempts)
            
            if target_has_intra:
                # 타겟 채널 busy → NPCA 차단
                sta.npca_blocked += 1
                sta.state = STAState.BACKOFF_FROZEN
                sta._reset_npca_params()
            else:
                # NPCA 전송 허용
                valid_npca_stas.append((sta, target_channel))
                
                # ✨ 원래 채널에도 OBSS 트래픽 추가 (자신의 전송으로 인해)
                self._add_npca_to_source_channel(sta)
                
                # 타겟 채널에 OBSS 트래픽 추가
                self._add_npca_to_target_channel(sta, target_channel)
        
        # ✨ 3단계: 업데이트된 채널 상태로 다른 STA들 재평가
        # NPCA 전송으로 인해 원래 채널이 busy해진 상태 반영
        final_channel_status = {}
        final_obss_occupied_until = {}
        
        for ch_id in range(self.num_channels):
            channel = self.channels[ch_id]
            final_channel_status[ch_id] = {
                'intra_busy': channel.is_busy(self.current_slot),
                'obss_busy': channel.is_obss_busy(self.current_slot),
                'any_busy': channel.is_any_busy(self.current_slot)
            }
            final_obss_occupied_until[ch_id] = channel.obss_occupied_until
        
        # NPCA 전송하지 않는 STA들의 상태 재조정
        for sta in self.stations:
            # NPCA 전송 중인 STA는 건드리지 않음
            if any(npca_sta.sta_id == sta.sta_id for npca_sta, _ in valid_npca_stas):
                continue
                
            # 다른 STA들은 업데이트된 채널 상태에 따라 상태 조정
            if sta.state in [STAState.BACKOFF, STAState.NPCA_BACKOFF, STAState.OBSS_FROZEN]:
                primary_status = final_channel_status[sta.channel_id]
                
                if primary_status['any_busy']:
                    # 채널이 busy해짐 (NPCA 전송 포함) → frozen 상태로
                    if primary_status['intra_busy']:
                        sta.state = STAState.BACKOFF_FROZEN
                    else:
                        # OBSS만 있는 경우 (NPCA 전송 포함)
                        sta.state = STAState.BACKOFF_FROZEN  # 단순화: OBSS든 NPCA든 정지
        
        # ✨ 4단계: 일반 전송들을 채널에 추가
        for sta, target_channel in transmission_attempts:
            self.channels[target_channel].add_transmission(sta.sta_id, sta.current_frame)
        
        # Resolve channel access for intra-BSS transmissions
        for channel in self.channels:
            channel.resolve_access(self.current_slot)
        
        # Log current state
        self._log_state()
    
    def _add_npca_as_obss(self, npca_sta: STAFiniteStateMachine, target_channel: int):
        """NPCA 전송을 target 채널에 OBSS 트래픽으로 추가"""
        if npca_sta.current_frame and npca_sta.npca_max_duration > 0:
            # NPCA 전송 시간은 min(frame_size, npca_max_duration)
            actual_duration = min(npca_sta.current_frame.size, npca_sta.npca_max_duration)
            
            # OBSS 트래픽 객체 생성
            npca_as_obss = OBSSTraffic(
                obss_id=f"npca_{npca_sta.sta_id}_{self.current_slot}",  # 고유 ID
                start_slot=self.current_slot,
                duration=actual_duration,
                source_channel=npca_sta.channel_id  # 원래 채널에서 온 것으로 표시
            )
            
            # Target 채널에 OBSS 트래픽으로 추가
            self.channels[target_channel].add_obss_traffic(npca_as_obss)
    
    def _add_npca_as_obss_traffic(self, sta_id: int, frame: FrameInfo, target_channel: int):
        """NPCA 전송을 target 채널에 OBSS 트래픽으로 추가"""
        sta = self.stations[sta_id]
        
        # NPCA 전송 시간 제한 적용
        actual_duration = min(frame.size, sta.npca_max_duration)
        
        # OBSS 트래픽 객체 생성
        npca_obss_traffic = OBSSTraffic(
            obss_id=f"npca_{sta_id}_{self.current_slot}",  # NPCA 식별용 ID
            start_slot=self.current_slot,
            duration=actual_duration,
            source_channel=sta.channel_id  # 원래 채널에서 온 트래픽
        )
        
        # Target 채널에 OBSS 트래픽으로 추가
        if 0 <= target_channel < self.num_channels:
            self.channels[target_channel].add_obss_traffic(npca_obss_traffic)
        
        # STA의 전송 완료 처리
        completion_slot = self.current_slot + actual_duration
        sta.on_transmission_result('success', completion_slot)
        
        # 새 프레임 생성
        self._generate_new_frame(sta, completion_slot)

    def _add_npca_as_obss_only(self, npca_sta: STAFiniteStateMachine, target_channel: int):
        """NPCA 전송을 타겟 채널에 OBSS 트래픽으로만 추가 (intra-BSS 전송 없음)"""
        
        if npca_sta.current_frame and npca_sta.npca_max_duration > 0:
            # NPCA 전송 시간 계산
            actual_duration = min(npca_sta.current_frame.size, npca_sta.npca_max_duration)
            
            # 타겟 채널에 OBSS 트래픽으로만 추가
            npca_as_obss = OBSSTraffic(
                obss_id=f"npca_{npca_sta.sta_id}_{self.current_slot}",
                start_slot=self.current_slot,
                duration=actual_duration,
                source_channel=npca_sta.channel_id
            )
            
            if 0 <= target_channel < self.num_channels:
                self.channels[target_channel].add_obss_traffic(npca_as_obss)
            
            # NPCA 전송 STA의 전송 완료 처리
            self._schedule_npca_completion(npca_sta, actual_duration)

    def _add_npca_to_source_channel(self, npca_sta: STAFiniteStateMachine):
        """NPCA 전송 시 원래 채널에도 OBSS 트래픽 추가"""
        if npca_sta.current_frame and npca_sta.npca_max_duration > 0:
            actual_duration = min(npca_sta.current_frame.size, npca_sta.npca_max_duration)
            
            # 원래 채널에 OBSS 트래픽 추가 (자신의 전송으로 인한 busy)
            source_obss = OBSSTraffic(
                obss_id=f"npca_source_{npca_sta.sta_id}_{self.current_slot}",
                start_slot=self.current_slot,
                duration=actual_duration,
                source_channel=npca_sta.channel_id
            )
            
            self.channels[npca_sta.channel_id].add_obss_traffic(source_obss)
    
    def _generate_new_frame(self, sta: STAFiniteStateMachine, creation_slot: int):
        """Generate new frame for station after successful transmission"""
        frame = FrameInfo(
            frame_id=self.frame_counter,
            source=sta.sta_id,
            size=self.frame_size,
            timestamp=creation_slot,
            creation_slot=creation_slot
        )
        self.frame_counter += 1
        sta.add_frame(frame)
    
    def _log_state(self):
        """Log current simulation state with NPCA support"""
        log_entry = {
            'time': self.current_slot * SLOTTIME,
            'slot': self.current_slot
        }
        
        # Channel states (기존과 동일)
        for ch_id, channel in enumerate(self.channels):
            log_entry[f'channel_{ch_id}_busy'] = channel.is_busy(self.current_slot)
            log_entry[f'channel_{ch_id}_obss_busy'] = channel.is_obss_busy(self.current_slot)
            log_entry[f'channel_{ch_id}_any_busy'] = channel.is_any_busy(self.current_slot)
            
            # Show remaining occupation time
            remaining_slots = max(0, channel.occupied_until - self.current_slot)
            obss_remaining_slots = max(0, channel.obss_occupied_until - self.current_slot)
            log_entry[f'channel_{ch_id}_occupied_until'] = remaining_slots
            log_entry[f'channel_{ch_id}_obss_occupied_until'] = obss_remaining_slots
            
            # Count active OBSS traffic
            log_entry[f'channel_{ch_id}_active_obss_count'] = len(channel.obss_traffic)
        
        # Station states by channel (NPCA 상태 포함)
        for ch_id in range(self.num_channels):
            channel_stas = [sta for sta in self.stations if sta.channel_id == ch_id]
            
            log_entry[f'states_ch_{ch_id}'] = [sta.state.value for sta in channel_stas]
            log_entry[f'backoff_ch_{ch_id}'] = [sta.backoff_counter for sta in channel_stas]
            log_entry[f'backoff_stage_ch_{ch_id}'] = [sta.backoff_stage for sta in channel_stas]
            log_entry[f'tx_attempts_ch_{ch_id}'] = [sta.tx_attempt for sta in channel_stas]
            log_entry[f'queue_len_ch_{ch_id}'] = [len(sta.tx_queue) for sta in channel_stas]
            log_entry[f'aoi_ch_{ch_id}'] = [sta.get_current_aoi(self.current_slot) for sta in channel_stas]
            log_entry[f'obss_deferrals_ch_{ch_id}'] = [sta.obss_deferrals for sta in channel_stas]
            log_entry[f'intra_bss_deferrals_ch_{ch_id}'] = [sta.intra_bss_deferrals for sta in channel_stas]
            
            # NPCA 통계 추가
            log_entry[f'npca_attempts_ch_{ch_id}'] = [sta.npca_attempts for sta in channel_stas]
            log_entry[f'npca_successful_ch_{ch_id}'] = [sta.npca_successful for sta in channel_stas]
            log_entry[f'npca_blocked_ch_{ch_id}'] = [sta.npca_blocked for sta in channel_stas]
            log_entry[f'npca_enabled_ch_{ch_id}'] = [sta.npca_enabled for sta in channel_stas]
        
        self.logs.append(log_entry)
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics with NPCA metrics"""
        # Aggregate OBSS generator statistics (활성화된 채널만)
        active_generators = [gen for gen in self.obss_generators if gen is not None]
        
        total_obss_generated = sum(gen.obss_generated for gen in active_generators)
        total_obss_deferred = sum(gen.obss_deferred for gen in active_generators)
        total_obss_blocked_by_intra = sum(gen.obss_blocked_by_intra_bss for gen in active_generators)
        total_obss_blocked_by_other_obss = sum(gen.obss_blocked_by_other_obss for gen in active_generators)
        
        # Calculate total OBSS duration
        total_obss_duration = 0
        if active_generators:
            for gen in active_generators:
                total_obss_duration += gen.obss_generated * np.mean(self.obss_frame_size_range)
        
        # NPCA 통계 수집
        total_npca_attempts = sum(sta.npca_attempts for sta in self.stations)
        total_npca_successful = sum(sta.npca_successful for sta in self.stations)
        total_npca_blocked = sum(sta.npca_blocked for sta in self.stations)
        npca_enabled_stas = sum(1 for sta in self.stations if sta.npca_enabled)
        
        stats = {
            'total_slots': self.current_slot,
            'total_time_us': self.current_slot * SLOTTIME,
            'obss_enabled': self.obss_enabled,
            'obss_enabled_per_channel': self.obss_enabled_per_channel,
            'npca_enabled_per_channel': self.npca_enabled,  # NPCA 설정 추가
            'obss_generation_rate': self.obss_generation_rate,
            'obss_events_generated': total_obss_generated,
            'obss_events_deferred': total_obss_deferred,
            'obss_blocked_by_intra_bss': total_obss_blocked_by_intra,
            'obss_blocked_by_other_obss': total_obss_blocked_by_other_obss,
            'obss_total_duration_slots': int(total_obss_duration),
            'obss_total_duration_us': int(total_obss_duration * SLOTTIME),
            'obss_channel_utilization': total_obss_duration / (self.current_slot * self.num_channels) if self.current_slot > 0 else 0,
            'mutual_interference_events': total_obss_blocked_by_intra + total_obss_deferred,
            
            # NPCA 전체 통계 추가
            'npca_enabled_stas': npca_enabled_stas,
            'npca_total_attempts': total_npca_attempts,
            'npca_total_successful': total_npca_successful,
            'npca_total_blocked': total_npca_blocked,
            'npca_success_rate': (total_npca_successful / total_npca_attempts * 100) if total_npca_attempts > 0 else 0,
            'npca_utilization_rate': (total_npca_attempts / self.current_slot * 100) if self.current_slot > 0 else 0,
            
            'stations': {}
        }
        
        # 채널별 OBSS 통계 (기존)
        stats['obss_per_channel'] = {}
        for ch_id in range(self.num_channels):
            if self.obss_generators[ch_id] is not None:
                gen = self.obss_generators[ch_id]
                stats['obss_per_channel'][ch_id] = {
                    'enabled': True,
                    'generated': gen.obss_generated,
                    'deferred': gen.obss_deferred,
                    'blocked_by_intra': gen.obss_blocked_by_intra_bss,
                    'blocked_by_other_obss': gen.obss_blocked_by_other_obss
                }
            else:
                stats['obss_per_channel'][ch_id] = {
                    'enabled': False,
                    'generated': 0,
                    'deferred': 0,
                    'blocked_by_intra': 0,
                    'blocked_by_other_obss': 0
                }
        
        # 채널별 NPCA 통계 추가
        stats['npca_per_channel'] = {}
        for ch_id in range(self.num_channels):
            channel_stas = [sta for sta in self.stations if sta.channel_id == ch_id]
            npca_stas = [sta for sta in channel_stas if sta.npca_enabled]
            
            if npca_stas:
                ch_npca_attempts = sum(sta.npca_attempts for sta in npca_stas)
                ch_npca_successful = sum(sta.npca_successful for sta in npca_stas)
                ch_npca_blocked = sum(sta.npca_blocked for sta in npca_stas)
                
                stats['npca_per_channel'][ch_id] = {
                    'enabled': True,
                    'npca_stas_count': len(npca_stas),
                    'total_attempts': ch_npca_attempts,
                    'total_successful': ch_npca_successful,
                    'total_blocked': ch_npca_blocked,
                    'success_rate': (ch_npca_successful / ch_npca_attempts * 100) if ch_npca_attempts > 0 else 0
                }
            else:
                stats['npca_per_channel'][ch_id] = {
                    'enabled': False,
                    'npca_stas_count': 0,
                    'total_attempts': 0,
                    'total_successful': 0,
                    'total_blocked': 0,
                    'success_rate': 0
                }
        
        # STA별 통계 (NPCA 정보 포함)
        for sta in self.stations:
            # Calculate average AoI from logs
            avg_aoi_slots = self._calculate_average_aoi(sta.sta_id)
            avg_aoi_time = avg_aoi_slots * SLOTTIME
            
            stats['stations'][sta.sta_id] = {
                'channel': sta.channel_id,
                'npca_enabled': sta.npca_enabled,  # NPCA 활성화 여부
                'successful_transmissions': sta.successful_transmissions,
                'collisions': sta.collision_count,
                'total_attempts': sta.total_attempts,
                'obss_deferrals': sta.obss_deferrals,
                'intra_bss_deferrals': sta.intra_bss_deferrals,
                'total_deferrals': sta.obss_deferrals + sta.intra_bss_deferrals,
                'success_rate': sta.successful_transmissions / max(1, sta.total_attempts),
                
                # NPCA 통계 추가
                'npca_attempts': sta.npca_attempts,
                'npca_successful': sta.npca_successful,
                'npca_blocked': sta.npca_blocked,
                'npca_success_rate': (sta.npca_successful / sta.npca_attempts * 100) if sta.npca_attempts > 0 else 0,
                'npca_ratio': (sta.npca_successful / sta.successful_transmissions * 100) if sta.successful_transmissions > 0 else 0,
                
                'final_state': sta.state.value,
                'final_backoff_stage': sta.backoff_stage,
                'average_aoi_slots': avg_aoi_slots,
                'average_aoi_time_us': avg_aoi_time
            }
        
        return stats
    
    def _calculate_average_aoi(self, sta_id: int) -> float:
        """Calculate average AoI for a specific station from logs"""
        if not self.logs:
            return 0.0
        
        # Find which channel the station belongs to
        sta = self.stations[sta_id]
        ch_id = sta.channel_id
        
        # Find station index within the channel
        channel_stas = [s for s in self.stations if s.channel_id == ch_id]
        sta_index = channel_stas.index(sta)
        
        # Extract AoI values from logs
        aoi_values = []
        for log_entry in self.logs:
            aoi_list = log_entry.get(f'aoi_ch_{ch_id}', [])
            if sta_index < len(aoi_list):
                aoi_values.append(aoi_list[sta_index])
        
        return sum(aoi_values) / len(aoi_values) if aoi_values else 0.0
    
    def _schedule_npca_completion(self, npca_sta: STAFiniteStateMachine, duration: int):
        """NPCA 전송 완료를 예약"""
        # 간단한 방법: 즉시 성공으로 처리
        # 더 정교한 방법: 별도의 완료 스케줄링 시스템 구현
        
        completion_slot = self.current_slot + duration
        npca_sta.transmitting_until = completion_slot
        
        # NPCA 전송 성공으로 간주 (충돌 없음)
        npca_sta.successful_transmissions += 1
        npca_sta.npca_successful += 1
        npca_sta.last_successful_tx_slot = completion_slot
        
        # 새 프레임 생성
        self._generate_new_frame(npca_sta, completion_slot)
        
        # NPCA 파라미터 리셋
        npca_sta._reset_npca_params()
        npca_sta.state = STAState.IDLE