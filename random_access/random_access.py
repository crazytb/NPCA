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
    TRANSMITTING = "transmitting"

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
    """Simplified 802.11 CSMA/CA Station implemented as FSM"""
    
    def __init__(self, sta_id: int, channel_id: int):
        self.sta_id = sta_id
        self.channel_id = channel_id
        
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
        
        # AoI tracking
        self.frame_creation_slot = 0  # When current frame was created
        self.last_successful_tx_slot = 0  # When last successful transmission completed
        
        # Statistics
        self.successful_transmissions = 0
        self.collision_count = 0
        self.total_attempts = 0
        self.obss_deferrals = 0  # Count of deferrals due to OBSS
        self.intra_bss_deferrals = 0
        
        # Flags
        self.has_frame_to_send = False
        self.tx_attempt = False
        
    def get_new_backoff(self) -> int:
        """Generate new backoff value based on current stage"""
        cw_index = min(self.backoff_stage, len(CONTENTION_WINDOW) - 1)
        cw = CONTENTION_WINDOW[cw_index]
        return np.random.randint(0, cw + 1)
    
    def add_frame(self, frame: FrameInfo):
        """Add frame to transmission queue"""
        self.tx_queue.append(frame)
        self.has_frame_to_send = True
    
    def update(self, current_slot: int, channel_busy: bool, obss_busy: bool = False) -> bool:
        """Update FSM state - returns True if attempting transmission"""
        
        self.tx_attempt = False
        
        # Skip update if currently transmitting and transmission not finished
        if self.state == STAState.TRANSMITTING and self.transmitting_until > current_slot:
            return False
        
        # ✅ 지연 통계 카운팅 (기존 코드 완전 교체)
        if self.state in [STAState.BACKOFF, STAState.BACKOFF_FROZEN, STAState.OBSS_FROZEN]:
            if channel_busy and not obss_busy:
                self.intra_bss_deferrals += 1
            elif obss_busy and not channel_busy:
                self.obss_deferrals += 1
            elif channel_busy and obss_busy:
                # 둘 다 있으면 intra-BSS가 우선순위 높음
                self.intra_bss_deferrals += 1
            # channel_busy=False, obss_busy=False인 경우는 지연 없음
            
        # Main FSM logic
        if self.state == STAState.IDLE:
            self._handle_idle_state(current_slot)
            
        elif self.state == STAState.BACKOFF:
            self._handle_backoff(channel_busy, obss_busy)
            
        elif self.state == STAState.BACKOFF_FROZEN:
            self._handle_backoff_frozen(channel_busy, obss_busy)
            
        elif self.state == STAState.OBSS_FROZEN:
            self._handle_obss_frozen(channel_busy, obss_busy)
            
        elif self.state == STAState.TRANSMITTING:
            self._handle_transmitting(current_slot)
        
        return self.tx_attempt
    
    def _handle_idle_state(self, current_slot: int):
        """Handle IDLE state"""
        if self.has_frame_to_send and self.tx_queue:
            self.current_frame = self.tx_queue.pop(0)
            self.has_frame_to_send = len(self.tx_queue) > 0
            # Update AoI tracking - frame creation slot
            self.frame_creation_slot = self.current_frame.creation_slot
            # Start with random backoff
            self.backoff_counter = self.get_new_backoff()
            self.state = STAState.BACKOFF
    
    def _handle_backoff(self, channel_busy: bool, obss_busy: bool):
        """Handle BACKOFF state"""
        if channel_busy:
            # Intra-BSS traffic detected - freeze backoff
            self.state = STAState.BACKOFF_FROZEN
        elif obss_busy:
            # Only OBSS traffic detected - freeze backoff due to OBSS
            self.state = STAState.OBSS_FROZEN
        else:
            # Channel is clear - proceed with backoff countdown
            if self.backoff_counter == 0:
                self.state = STAState.TRANSMITTING
                self.tx_attempt = True
                self.total_attempts += 1
            else:
                # Decrement backoff counter
                self.backoff_counter -= 1
                # Check again after decrementing
                if self.backoff_counter == 0:
                    self.state = STAState.TRANSMITTING
                    self.tx_attempt = True
                    self.total_attempts += 1
    
    def _handle_backoff_frozen(self, channel_busy: bool, obss_busy: bool):
        """Handle BACKOFF_FROZEN state (frozen due to intra-BSS traffic)"""
        if not channel_busy and not obss_busy:
            # Both intra-BSS and OBSS traffic cleared - resume backoff
            self.state = STAState.BACKOFF
        elif not channel_busy and obss_busy:
            # Intra-BSS cleared but OBSS still present - transition to OBSS_FROZEN
            self.state = STAState.OBSS_FROZEN
    
    def _handle_obss_frozen(self, channel_busy: bool, obss_busy: bool):
        """Handle OBSS_FROZEN state (frozen due to OBSS traffic only)"""
        if channel_busy:
            # Intra-BSS traffic appeared - transition to BACKOFF_FROZEN (higher priority)
            self.state = STAState.BACKOFF_FROZEN
        elif not obss_busy:
            # OBSS traffic cleared - resume backoff
            self.state = STAState.BACKOFF
    
    def _handle_transmitting(self, current_slot: int):
        """Handle TRANSMITTING state"""
        if self.transmitting_until == -1:
            self.transmitting_until = current_slot + self.current_frame.size
        
        if current_slot >= self.transmitting_until:
            # Transmission completed - wait for channel resolution
            pass
    
    def on_transmission_result(self, result: str, completion_slot: int):
        """Handle transmission result from channel"""
        if self.state != STAState.TRANSMITTING:
            return
            
        if result == 'success':
            self.state = STAState.IDLE
            self.successful_transmissions += 1
            self.last_successful_tx_slot = completion_slot
            self._reset_transmission_params()
        elif result == 'collision':
            self.backoff_stage = min(self.backoff_stage + 1, len(CONTENTION_WINDOW) - 1)
            self.backoff_counter = self.get_new_backoff()
            self.collision_count += 1
            self.state = STAState.BACKOFF
            self._reset_transmission_params(keep_frame=True)
    
    def _reset_transmission_params(self, keep_frame: bool = False):
        """Reset transmission parameters"""
        if not keep_frame:
            self.backoff_stage = 0
            self.current_frame = None
            # Reset AoI tracking when frame is successfully transmitted
            self.frame_creation_slot = 0
        self.transmitting_until = -1
        self.tx_attempt = False
    
    def get_current_aoi(self, current_slot: int) -> int:
        """Calculate current Age of Information in slots"""
        if self.current_frame is None:
            # No current frame - AoI is time since last successful transmission
            return current_slot - self.last_successful_tx_slot
        else:
            # Current frame exists - AoI is time since frame creation
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
                 simulation_time: int, frame_size: int, obss_enabled: bool = False,
                 obss_generation_rate: float = 0.001, obss_frame_size_range: Tuple[int, int] = (20, 50)):
        self.num_channels = num_channels
        self.stas_per_channel = stas_per_channel
        self.simulation_time = simulation_time
        self.frame_size = frame_size
        self.obss_enabled = obss_enabled
        self.obss_generation_rate = obss_generation_rate
        self.obss_frame_size_range = obss_frame_size_range
        
        # Initialize channels
        self.channels = [ChannelFSM(i) for i in range(num_channels)]
        
        # Initialize stations
        self.stations = []
        sta_id = 0
        for ch_id in range(num_channels):
            for _ in range(stas_per_channel[ch_id]):
                self.stations.append(STAFiniteStateMachine(sta_id, ch_id))
                sta_id += 1
        
        # Initialize OBSS generators (one per channel)
        self.obss_generators = []
        if self.obss_enabled:
            for ch_id in range(num_channels):
                generator = OBSSGenerator(
                    source_channel=ch_id,
                    generation_rate=obss_generation_rate,
                    frame_size_range=obss_frame_size_range
                )
                self.obss_generators.append(generator)
        
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
        
        # Generate OBSS traffic for each channel
        for generator in self.obss_generators:
            source_ch = generator.source_channel
            
            # Check channel status for OBSS generation
            intra_bss_busy = self.channels[source_ch].is_busy(current_slot)
            other_obss_busy = self.channels[source_ch].is_obss_busy(current_slot)
            
            # Attempt OBSS generation (with backoff if channel busy)
            obss_traffic = generator.attempt_generation(current_slot, intra_bss_busy, other_obss_busy)
            
            if obss_traffic:
                # Add OBSS traffic to affected channels
                affected_channels = self._get_affected_channels(source_ch)
                for target_ch in affected_channels:
                    if 0 <= target_ch < self.num_channels:
                        self.channels[target_ch].add_obss_traffic(obss_traffic)
    
    def run(self) -> pd.DataFrame:
        """Run the simulation"""
        for self.current_slot in range(self.simulation_time):
            self._tick()
        
        return pd.DataFrame(self.logs)
    
    def _tick(self):
        """One simulation tick"""
        # Generate OBSS traffic first (with mutual interference consideration)
        self._generate_obss_traffic(self.current_slot)
        
        # Update channels first and get completed transmission results
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
        
        # Update stations and collect transmission attempts
        for sta in self.stations:
            channel = self.channels[sta.channel_id]
            channel_busy = channel.is_busy(self.current_slot)
            obss_busy = channel.is_obss_busy(self.current_slot)
            
            # Update station FSM with separate intra-BSS and OBSS busy signals
            tx_attempt = sta.update(self.current_slot, channel_busy, obss_busy)
            
            # Collect transmission attempts
            if tx_attempt and sta.current_frame:
                channel.add_transmission(sta.sta_id, sta.current_frame)
        
        # Resolve channel access (schedule future results)
        for channel in self.channels:
            channel.resolve_access(self.current_slot)
        
        # Log current state
        self._log_state()
    
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
        """Log current simulation state"""
        log_entry = {
            'time': self.current_slot * SLOTTIME,
            'slot': self.current_slot
        }
        
        # Channel states
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
        
        # Station states by channel
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
        
        self.logs.append(log_entry)
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        # Aggregate OBSS generator statistics
        total_obss_generated = sum(gen.obss_generated for gen in self.obss_generators) if self.obss_generators else 0
        total_obss_deferred = sum(gen.obss_deferred for gen in self.obss_generators) if self.obss_generators else 0
        total_obss_blocked_by_intra = sum(gen.obss_blocked_by_intra_bss for gen in self.obss_generators) if self.obss_generators else 0
        total_obss_blocked_by_other_obss = sum(gen.obss_blocked_by_other_obss for gen in self.obss_generators) if self.obss_generators else 0
        
        # Calculate total OBSS duration
        total_obss_duration = 0
        if self.obss_generators:
            for gen in self.obss_generators:
                total_obss_duration += gen.obss_generated * np.mean(self.obss_frame_size_range)
        
        stats = {
            'total_slots': self.current_slot,
            'total_time_us': self.current_slot * SLOTTIME,
            'obss_enabled': self.obss_enabled,
            'obss_generation_rate': self.obss_generation_rate,
            'obss_events_generated': total_obss_generated,
            'obss_events_deferred': total_obss_deferred,
            'obss_blocked_by_intra_bss': total_obss_blocked_by_intra,
            'obss_blocked_by_other_obss': total_obss_blocked_by_other_obss,
            'obss_total_duration_slots': int(total_obss_duration),
            'obss_total_duration_us': int(total_obss_duration * SLOTTIME),
            'obss_channel_utilization': total_obss_duration / (self.current_slot * self.num_channels) if self.current_slot > 0 else 0,
            'mutual_interference_events': total_obss_blocked_by_intra + total_obss_deferred,
            'stations': {}
        }
        
        for sta in self.stations:
            # Calculate average AoI from logs
            avg_aoi_slots = self._calculate_average_aoi(sta.sta_id)
            avg_aoi_time = avg_aoi_slots * SLOTTIME
            
            stats['stations'][sta.sta_id] = {
                'channel': sta.channel_id,
                'successful_transmissions': sta.successful_transmissions,
                'collisions': sta.collision_count,
                'total_attempts': sta.total_attempts,
                'obss_deferrals': sta.obss_deferrals,
                'intra_bss_deferrals': sta.intra_bss_deferrals,
                'total_deferrals': sta.obss_deferrals + sta.intra_bss_deferrals,
                'success_rate': sta.successful_transmissions / max(1, sta.total_attempts),
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