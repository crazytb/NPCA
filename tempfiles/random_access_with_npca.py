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
    """Station states in CSMA/CA FSM with NPCA support"""
    IDLE = "idle"
    BACKOFF = "backoff"
    BACKOFF_FROZEN = "backoff_frozen"
    TRANSMITTING = "transmitting"
    NPCA_SWITCHING = "npca_switching"

class ChannelType(Enum):
    """Channel type for NPCA operation"""
    BSS_PRIMARY = "bss_primary"
    NPCA_PRIMARY = "npca_primary"

@dataclass
class FrameInfo:
    """Frame information for transmission"""
    frame_id: int
    source: int
    size: int  # in slots
    timestamp: int  # When frame was originally created
    creation_slot: int  # Slot when frame was created for AoI calculation

@dataclass
class OBSSPpdu:
    """OBSS PPDU information"""
    source_bss: int
    bandwidth: int  # MHz (20, 40, 80, 160)
    duration_slots: int
    signal_strength: float
    start_slot: int
    
    def get_occupied_channels(self, base_channel=0):
        """Get list of 20MHz channels occupied by this PPDU"""
        if self.bandwidth == 20:
            return [base_channel]
        elif self.bandwidth == 40:
            return [base_channel, base_channel + 1]
        elif self.bandwidth == 80:
            return [base_channel, base_channel + 1, base_channel + 2, base_channel + 3]
        return [base_channel]

class OBSSTrafficGenerator:
    """Generate OBSS traffic patterns"""
    
    def __init__(self, channel_id: int, traffic_intensity: float = 0.3):
        self.channel_id = channel_id
        self.traffic_intensity = traffic_intensity
        self.next_transmission_slot = random.randint(100, 500)
        self.current_ppdu = None
        
    def generate_transmission(self, current_slot: int) -> Optional[OBSSPpdu]:
        """Generate OBSS transmission if it's time"""
        if current_slot >= self.next_transmission_slot:
            # Generate new OBSS PPDU
            duration = random.choice([50, 100, 200, 500])  # Various durations
            bandwidth = random.choice([20, 40, 80])  # Different bandwidths
            
            ppdu = OBSSPpdu(
                source_bss=99,  # OBSS identifier
                bandwidth=bandwidth,
                duration_slots=duration,
                signal_strength=random.uniform(0.5, 1.0),
                start_slot=current_slot
            )
            
            # Schedule next transmission
            inter_arrival = int(duration / self.traffic_intensity)
            self.next_transmission_slot = current_slot + duration + inter_arrival
            
            return ppdu
        return None

class STAFiniteStateMachine:
    """802.11 CSMA/CA Station with NPCA support"""
    
    def __init__(self, sta_id: int, channel_id: int, bss_id: int = 0):
        self.sta_id = sta_id
        self.channel_id = channel_id
        self.bss_id = bss_id
        
        # FSM State
        self.state = STAState.IDLE
        self.current_channel_type = ChannelType.BSS_PRIMARY
        
        # CSMA/CA Parameters
        self.backoff_stage = 0
        self.backoff_counter = 0
        self.max_retries = 7
        
        # NPCA Parameters
        self.npca_channel_id = channel_id + 1 if channel_id == 0 else channel_id - 1
        self.npca_switching_delay = 10  # slots
        self.npca_switch_back_delay = 10  # slots
        self.min_duration_threshold = 50  # slots
        self.switching_countdown = 0
        
        # NPCA Backoff (separate from BSS backoff)
        self.npca_backoff_stage = 0
        self.npca_backoff_counter = 0
        
        # Frame handling
        self.tx_queue = []
        self.current_frame = None
        self.transmitting_until = -1
        
        # AoI tracking
        self.frame_creation_slot = 0
        self.last_successful_tx_slot = 0
        
        # OBSS detection
        self.obss_nav = 0
        self.current_obss_info = None
        self.obss_checked_this_backoff = False
        
        # Statistics
        self.successful_transmissions = 0
        self.collision_count = 0
        self.total_attempts = 0
        self.npca_switches = 0
        
        # Flags
        self.has_frame_to_send = False
        self.tx_attempt = False
        
    def get_new_backoff(self, use_npca: bool = False) -> int:
        """Generate new backoff value based on current stage"""
        if use_npca:
            cw_index = min(self.npca_backoff_stage, len(CONTENTION_WINDOW) - 1)
        else:
            cw_index = min(self.backoff_stage, len(CONTENTION_WINDOW) - 1)
        cw = CONTENTION_WINDOW[cw_index]
        return np.random.randint(0, cw + 1)
    
    def add_frame(self, frame: FrameInfo):
        """Add frame to transmission queue"""
        self.tx_queue.append(frame)
        self.has_frame_to_send = True
    
    def update(self, current_slot: int, channel_busy: bool, obss_info: Dict = None) -> bool:
        """Update FSM state - returns True if attempting transmission"""
        
        self.tx_attempt = False
        
        # Handle NPCA switching delay
        if self.state == STAState.NPCA_SWITCHING:
            return self._handle_npca_switching(current_slot)
        
        # Skip update if currently transmitting and transmission not finished
        if self.state == STAState.TRANSMITTING and self.transmitting_until > current_slot:
            return False
        
        # OBSS detection and NPCA decision (only when not transmitting)
        if (self.current_channel_type == ChannelType.BSS_PRIMARY and 
            self.state != STAState.TRANSMITTING and
            obss_info and obss_info.get('obss_detected', False)):
            
            if self._should_switch_to_npca(obss_info, current_slot):
                self._initiate_npca_switch(current_slot)
                return False
        
        # Main FSM logic
        if self.state == STAState.IDLE:
            self._handle_idle_state(current_slot)
            
        elif self.state == STAState.BACKOFF:
            self._handle_backoff(channel_busy, obss_info)
            
        elif self.state == STAState.BACKOFF_FROZEN:
            self._handle_backoff_frozen(channel_busy, obss_info)
            
        elif self.state == STAState.TRANSMITTING:
            self._handle_transmitting(current_slot)
        
        return self.tx_attempt
    
    def _handle_idle_state(self, current_slot: int):
        """Handle IDLE state"""
        if self.has_frame_to_send and self.tx_queue:
            self.current_frame = self.tx_queue.pop(0)
            self.has_frame_to_send = len(self.tx_queue) > 0
            self.frame_creation_slot = self.current_frame.creation_slot
            
            # Initialize backoff based on current channel
            if self.current_channel_type == ChannelType.NPCA_PRIMARY:
                self.npca_backoff_counter = self.get_new_backoff(use_npca=True)
            else:
                self.backoff_counter = self.get_new_backoff()
            
            self.state = STAState.BACKOFF
            self.obss_checked_this_backoff = False
    
    def _handle_backoff(self, channel_busy: bool, obss_info: Dict = None):
        """Handle BACKOFF state with OBSS awareness"""
        
        # Check OBSS at the start of backoff
        if not self.obss_checked_this_backoff and obss_info:
            self.obss_checked_this_backoff = True
            if obss_info.get('obss_detected', False):
                self.obss_nav = obss_info.get('remaining_duration', 0)
        
        # Determine if channel is effectively busy
        effective_busy = channel_busy or (self.obss_nav > 0)
        
        if effective_busy:
            self.state = STAState.BACKOFF_FROZEN
            if self.obss_nav > 0:
                self.obss_nav -= 1
        else:
            # Use appropriate backoff counter based on current channel
            if self.current_channel_type == ChannelType.NPCA_PRIMARY:
                current_counter = self.npca_backoff_counter
            else:
                current_counter = self.backoff_counter
            
            if current_counter == 0:
                self.state = STAState.TRANSMITTING
                self.tx_attempt = True
                self.total_attempts += 1
                self.obss_checked_this_backoff = False
            else:
                # Decrement appropriate counter
                if self.current_channel_type == ChannelType.NPCA_PRIMARY:
                    self.npca_backoff_counter -= 1
                    if self.npca_backoff_counter == 0:
                        self.state = STAState.TRANSMITTING
                        self.tx_attempt = True
                        self.total_attempts += 1
                        self.obss_checked_this_backoff = False
                else:
                    self.backoff_counter -= 1
                    if self.backoff_counter == 0:
                        self.state = STAState.TRANSMITTING
                        self.tx_attempt = True
                        self.total_attempts += 1
                        self.obss_checked_this_backoff = False
    
    def _handle_backoff_frozen(self, channel_busy: bool, obss_info: Dict = None):
        """Handle BACKOFF_FROZEN state"""
        # Update OBSS NAV
        if self.obss_nav > 0:
            self.obss_nav -= 1
        
        # Check if can resume backoff
        effective_busy = channel_busy or (self.obss_nav > 0)
        if not effective_busy:
            self.state = STAState.BACKOFF
    
    def _handle_transmitting(self, current_slot: int):
        """Handle TRANSMITTING state"""
        if self.transmitting_until == -1:
            self.transmitting_until = current_slot + self.current_frame.size
        
        if current_slot >= self.transmitting_until:
            # Transmission completed - wait for channel resolution
            pass
    
    def _handle_npca_switching(self, current_slot: int) -> bool:
        """Handle NPCA switching delay"""
        self.switching_countdown -= 1
        if self.switching_countdown <= 0:
            # Switching complete
            self.current_channel_type = ChannelType.NPCA_PRIMARY
            self.state = STAState.IDLE
            self.npca_switches += 1
            
            # Initialize NPCA backoff parameters
            self.npca_backoff_stage = 0
            self.npca_backoff_counter = 0
        return False
    
    def _should_switch_to_npca(self, obss_info: Dict, current_slot: int) -> bool:
        """Check if should switch to NPCA based on IEEE 802.11bn conditions"""
        if not obss_info.get('obss_detected', False):
            return False
        
        # Condition b: Duration check
        remaining_duration = obss_info.get('remaining_duration', 0)
        duration_ok = remaining_duration > self.min_duration_threshold
        
        # Condition c: Channel overlap check
        no_overlap = not obss_info.get('overlaps_npca_channel', False)
        
        return duration_ok and no_overlap
    
    def _initiate_npca_switch(self, current_slot: int):
        """Initiate switch to NPCA channel"""
        self.state = STAState.NPCA_SWITCHING
        self.switching_countdown = self.npca_switching_delay
        self.obss_checked_this_backoff = False
    
    def switch_back_to_bss(self, current_slot: int):
        """Switch back to BSS primary channel"""
        if self.current_channel_type == ChannelType.NPCA_PRIMARY:
            self.current_channel_type = ChannelType.BSS_PRIMARY
            self.state = STAState.IDLE
            # Reset backoff parameters for BSS channel
            self.backoff_stage = 0
            self.backoff_counter = 0
    
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
            # Increment appropriate backoff stage
            if self.current_channel_type == ChannelType.NPCA_PRIMARY:
                self.npca_backoff_stage = min(self.npca_backoff_stage + 1, len(CONTENTION_WINDOW) - 1)
                self.npca_backoff_counter = self.get_new_backoff(use_npca=True)
            else:
                self.backoff_stage = min(self.backoff_stage + 1, len(CONTENTION_WINDOW) - 1)
                self.backoff_counter = self.get_new_backoff()
            
            self.collision_count += 1
            self.state = STAState.BACKOFF
            self._reset_transmission_params(keep_frame=True)
    
    def _reset_transmission_params(self, keep_frame: bool = False):
        """Reset transmission parameters"""
        if not keep_frame:
            if self.current_channel_type == ChannelType.BSS_PRIMARY:
                self.backoff_stage = 0
            else:
                self.npca_backoff_stage = 0
            self.current_frame = None
            self.frame_creation_slot = 0
        self.transmitting_until = -1
        self.tx_attempt = False
        self.obss_checked_this_backoff = False
    
    def get_current_aoi(self, current_slot: int) -> int:
        """Calculate current Age of Information in slots"""
        if self.current_frame is None:
            return current_slot - self.last_successful_tx_slot
        else:
            return current_slot - self.frame_creation_slot
    
    def get_current_backoff_counter(self) -> int:
        """Get current backoff counter based on channel type"""
        if self.current_channel_type == ChannelType.NPCA_PRIMARY:
            return self.npca_backoff_counter
        else:
            return self.backoff_counter

class ChannelFSM:
    """Channel state machine with OBSS support"""
    
    def __init__(self, channel_id: int, obss_traffic_intensity: float = 0.0):
        self.channel_id = channel_id
        self.transmitting_stations = []
        self.occupied_until = -1
        self.current_frame = None
        self.pending_results = []
        
        # OBSS support
        self.obss_generator = None
        if obss_traffic_intensity > 0:
            self.obss_generator = OBSSTrafficGenerator(channel_id, obss_traffic_intensity)
        self.current_obss_ppdu = None
        self.obss_occupied_until = -1
    
    def update(self, current_slot: int):
        """Update channel state and return completed transmission results"""
        results = []
        
        # Check if transmission completed
        if current_slot >= self.occupied_until and self.occupied_until != -1:
            results = [(sta_id, result, current_slot) for sta_id, result in self.pending_results]
            self.pending_results.clear()
            self.transmitting_stations.clear()
            self.current_frame = None
            self.occupied_until = -1
        
        # Update OBSS status
        if current_slot >= self.obss_occupied_until:
            self.current_obss_ppdu = None
            self.obss_occupied_until = -1
        
        # Generate new OBSS traffic
        if self.obss_generator:
            new_obss = self.obss_generator.generate_transmission(current_slot)
            if new_obss:
                self.current_obss_ppdu = new_obss
                self.obss_occupied_until = current_slot + new_obss.duration_slots
        
        return results
    
    def add_transmission(self, sta_id: int, frame: FrameInfo):
        """Add transmission attempt"""
        self.transmitting_stations.append((sta_id, frame))
    
    def resolve_access(self, current_slot: int):
        """Resolve channel access and schedule results"""
        if len(self.transmitting_stations) == 0:
            return
        
        # Only process if channel is not already occupied
        if current_slot >= self.occupied_until:
            if len(self.transmitting_stations) == 1:
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
        
        self.transmitting_stations.clear()
    
    def is_busy(self, current_slot: int) -> bool:
        """Check if channel is busy (our BSS transmissions only)"""
        return current_slot < self.occupied_until
    
    def get_obss_info(self, current_slot: int) -> Dict:
        """Get OBSS information for NPCA decision making"""
        if self.current_obss_ppdu and current_slot < self.obss_occupied_until:
            # Check if OBSS overlaps with NPCA channel
            obss_channels = self.current_obss_ppdu.get_occupied_channels(self.channel_id)
            npca_channel = self.channel_id + 1 if self.channel_id == 0 else self.channel_id - 1
            overlaps_npca = npca_channel in obss_channels
            
            return {
                'obss_detected': True,
                'remaining_duration': self.obss_occupied_until - current_slot,
                'ppdu_bandwidth': self.current_obss_ppdu.bandwidth,
                'overlaps_npca_channel': overlaps_npca,
                'signal_strength': self.current_obss_ppdu.signal_strength,
                'source_bss': self.current_obss_ppdu.source_bss
            }
        
        return {'obss_detected': False}

class SimplifiedCSMACASimulation:
    """CSMA/CA Network Simulation with NPCA support"""
    
    def __init__(self, num_channels: int, stas_per_channel: List[int], 
                 simulation_time: int, frame_size: int, obss_traffic_intensity: float = 0.0):
        self.num_channels = num_channels
        self.stas_per_channel = stas_per_channel
        self.simulation_time = simulation_time
        self.frame_size = frame_size
        self.obss_traffic_intensity = obss_traffic_intensity
        
        # Initialize channels (only channel 0 has OBSS traffic)
        self.channels = []
        for i in range(num_channels):
            obss_intensity = obss_traffic_intensity if i == 0 else 0.0
            self.channels.append(ChannelFSM(i, obss_intensity))
        
        # Initialize stations
        self.stations = []
        sta_id = 0
        for ch_id in range(num_channels):
            for _ in range(stas_per_channel[ch_id]):
                bss_id = 0  # All STAs belong to BSS 0
                self.stations.append(STAFiniteStateMachine(sta_id, ch_id, bss_id))
                sta_id += 1
        
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
    
    def run(self) -> pd.DataFrame:
        """Run the simulation"""
        for self.current_slot in range(self.simulation_time):
            self._tick()
        
        return pd.DataFrame(self.logs)
    
    def _tick(self):
        """One simulation tick"""
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
            # Determine which channel the STA is currently using
            if sta.current_channel_type == ChannelType.BSS_PRIMARY:
                channel = self.channels[sta.channel_id]
            else:  # NPCA_PRIMARY
                if sta.npca_channel_id < len(self.channels):
                    channel = self.channels[sta.npca_channel_id]
                else:
                    channel = self.channels[sta.channel_id]  # Fallback
            
            channel_busy = channel.is_busy(self.current_slot)
            obss_info = channel.get_obss_info(self.current_slot)
            
            # Update station FSM
            tx_attempt = sta.update(self.current_slot, channel_busy, obss_info)
            
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
            remaining_slots = max(0, channel.occupied_until - self.current_slot)
            log_entry[f'channel_{ch_id}_occupied_until'] = remaining_slots
            
            # OBSS information
            obss_info = channel.get_obss_info(self.current_slot)
            log_entry[f'channel_{ch_id}_obss_active'] = obss_info.get('obss_detected', False)
            log_entry[f'channel_{ch_id}_obss_remaining'] = obss_info.get('remaining_duration', 0)
        
        # Station states by channel
        for ch_id in range(self.num_channels):
            # Get STAs currently operating on this channel (considering NPCA)
            channel_stas = []
            for sta in self.stations:
                if ((sta.current_channel_type == ChannelType.BSS_PRIMARY and sta.channel_id == ch_id) or
                    (sta.current_channel_type == ChannelType.NPCA_PRIMARY and sta.npca_channel_id == ch_id)):
                    channel_stas.append(sta)
            
            log_entry[f'states_ch_{ch_id}'] = [sta.state.value for sta in channel_stas]
            log_entry[f'backoff_ch_{ch_id}'] = [sta.get_current_backoff_counter() for sta in channel_stas]
            log_entry[f'backoff_stage_ch_{ch_id}'] = [
                sta.npca_backoff_stage if sta.current_channel_type == ChannelType.NPCA_PRIMARY 
                else sta.backoff_stage for sta in channel_stas
            ]
            log_entry[f'tx_attempts_ch_{ch_id}'] = [sta.tx_attempt for sta in channel_stas]
            log_entry[f'queue_len_ch_{ch_id}'] = [len(sta.tx_queue) for sta in channel_stas]
            log_entry[f'aoi_ch_{ch_id}'] = [sta.get_current_aoi(self.current_slot) for sta in channel_stas]
            log_entry[f'channel_type_ch_{ch_id}'] = [sta.current_channel_type.value for sta in channel_stas]
            log_entry[f'npca_switches_ch_{ch_id}'] = [sta.npca_switches for sta in channel_stas]
        
        self.logs.append(log_entry)
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        stats = {
            'total_slots': self.current_slot,
            'total_time_us': self.current_slot * SLOTTIME,
            'stations': {}
        }
        
        for sta in self.stations:
            # Calculate average AoI from logs
            avg_aoi_slots = self._calculate_average_aoi(sta.sta_id)
            avg_aoi_time = avg_aoi_slots * SLOTTIME
            
            stats['stations'][sta.sta_id] = {
                'channel': sta.channel_id,
                'npca_channel': sta.npca_channel_id,
                'successful_transmissions': sta.successful_transmissions,
                'collisions': sta.collision_count,
                'total_attempts': sta.total_attempts,
                'success_rate': sta.successful_transmissions / max(1, sta.total_attempts),
                'final_state': sta.state.value,
                'final_channel_type': sta.current_channel_type.value,
                'final_backoff_stage': sta.npca_backoff_stage if sta.current_channel_type == ChannelType.NPCA_PRIMARY else sta.backoff_stage,
                'npca_switches': sta.npca_switches,
                'average_aoi_slots': avg_aoi_slots,
                'average_aoi_time_us': avg_aoi_time
            }
        
        return stats
    
    def _calculate_average_aoi(self, sta_id: int) -> float:
        """Calculate average AoI for a specific station from logs"""
        if not self.logs:
            return 0.0
        
        # Find all entries where this STA appears in any channel
        aoi_values = []
        for log_entry in self.logs:
            for ch_id in range(self.num_channels):
                aoi_list = log_entry.get(f'aoi_ch_{ch_id}', [])
                channel_stas = []
                for sta in self.stations:
                    if ((sta.current_channel_type == ChannelType.BSS_PRIMARY and sta.channel_id == ch_id) or
                        (sta.current_channel_type == ChannelType.NPCA_PRIMARY and sta.npca_channel_id == ch_id)):
                        channel_stas.append(sta)
                
                # Find STA index in this channel's list
                for idx, sta in enumerate(channel_stas):
                    if sta.sta_id == sta_id and idx < len(aoi_list):
                        aoi_values.append(aoi_list[idx])
                        break
        
        return sum(aoi_values) / len(aoi_values) if aoi_values else 0.0