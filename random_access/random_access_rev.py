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
    TRANSMITTING = "transmitting"

@dataclass
class FrameInfo:
    """Frame information for transmission"""
    frame_id: int
    source: int
    size: int  # in slots
    timestamp: int

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
        
        # Statistics
        self.successful_transmissions = 0
        self.collision_count = 0
        self.total_attempts = 0
        
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
    
    def update(self, current_slot: int, channel_busy: bool) -> bool:
        """Update FSM state - returns True if attempting transmission"""
        
        self.tx_attempt = False
        
        # Skip update if currently transmitting and transmission not finished
        if self.state == STAState.TRANSMITTING and self.transmitting_until > current_slot:
            return False
            
        # Main FSM logic
        if self.state == STAState.IDLE:
            self._handle_idle_state()
            
        elif self.state == STAState.BACKOFF:
            self._handle_backoff(channel_busy)
            
        elif self.state == STAState.BACKOFF_FROZEN:
            self._handle_backoff_frozen(channel_busy)
            
        elif self.state == STAState.TRANSMITTING:
            self._handle_transmitting(current_slot)
        
        return self.tx_attempt
    
    def _handle_idle_state(self):
        """Handle IDLE state"""
        if self.has_frame_to_send and self.tx_queue:
            self.current_frame = self.tx_queue.pop(0)
            self.has_frame_to_send = len(self.tx_queue) > 0
            # Start with random backoff
            self.backoff_counter = self.get_new_backoff()
            self.state = STAState.BACKOFF
    
    def _handle_backoff(self, channel_busy: bool):
        """Handle BACKOFF state"""
        if channel_busy:
            self.state = STAState.BACKOFF_FROZEN
        else:
            # Check if backoff counter is already 0 at the start of this slot
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
    
    def _handle_backoff_frozen(self, channel_busy: bool):
        """Handle BACKOFF_FROZEN state"""
        if not channel_busy:
            self.state = STAState.BACKOFF
    
    def _handle_transmitting(self, current_slot: int):
        """Handle TRANSMITTING state"""
        if self.transmitting_until == -1:
            self.transmitting_until = current_slot + self.current_frame.size
        
        if current_slot >= self.transmitting_until:
            # Transmission completed - wait for channel resolution
            pass
    
    def on_transmission_result(self, result: str):
        """Handle transmission result from channel"""
        if self.state != STAState.TRANSMITTING:
            return
            
        if result == 'success':
            self.state = STAState.IDLE
            self.successful_transmissions += 1
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
        self.transmitting_until = -1
        self.tx_attempt = False

class ChannelFSM:
    """Simplified Channel state machine for CSMA/CA"""
    
    def __init__(self, channel_id: int):
        self.channel_id = channel_id
        self.transmitting_stations = []
        self.occupied_until = -1
        self.current_frame = None
        self.pending_results = []  # Store pending transmission results
    
    def update(self, current_slot: int):
        """Update channel state and return completed transmission results"""
        results = []
        
        # Check if transmission completed
        if current_slot >= self.occupied_until and self.occupied_until != -1:
            # Transmission completed - return results
            results = self.pending_results.copy()
            self.pending_results.clear()
            self.transmitting_stations.clear()
            self.current_frame = None
            self.occupied_until = -1
        
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
        """Check if channel is busy"""
        return current_slot < self.occupied_until

class SimplifiedCSMACASimulation:
    """Simplified CSMA/CA Network Simulation using FSM"""
    
    def __init__(self, num_channels: int, stas_per_channel: List[int], 
                 simulation_time: int, frame_size: int):
        self.num_channels = num_channels
        self.stas_per_channel = stas_per_channel
        self.simulation_time = simulation_time
        self.frame_size = frame_size
        
        # Initialize channels
        self.channels = [ChannelFSM(i) for i in range(num_channels)]
        
        # Initialize stations
        self.stations = []
        sta_id = 0
        for ch_id in range(num_channels):
            for _ in range(stas_per_channel[ch_id]):
                self.stations.append(STAFiniteStateMachine(sta_id, ch_id))
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
                timestamp=0
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
            for sta_id, result in results:
                sta = self.stations[sta_id]
                sta.on_transmission_result(result)
                
                # Generate new frame after successful transmission
                if result == 'success':
                    self._generate_new_frame(sta)
        
        # Update stations and collect transmission attempts
        for sta in self.stations:
            channel = self.channels[sta.channel_id]
            channel_busy = channel.is_busy(self.current_slot)
            
            # Update station FSM
            tx_attempt = sta.update(self.current_slot, channel_busy)
            
            # Collect transmission attempts
            if tx_attempt and sta.current_frame:
                channel.add_transmission(sta.sta_id, sta.current_frame)
        
        # Resolve channel access (schedule future results)
        for channel in self.channels:
            channel.resolve_access(self.current_slot)
        
        # Log current state
        self._log_state()
    
    def _generate_new_frame(self, sta: STAFiniteStateMachine):
        """Generate new frame for station after successful transmission"""
        frame = FrameInfo(
            frame_id=self.frame_counter,
            source=sta.sta_id,
            size=self.frame_size,
            timestamp=self.current_slot
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
            # Show remaining occupation time instead of absolute time
            remaining_slots = max(0, channel.occupied_until - self.current_slot)
            log_entry[f'channel_{ch_id}_occupied_until'] = remaining_slots
        
        # Station states by channel
        for ch_id in range(self.num_channels):
            channel_stas = [sta for sta in self.stations if sta.channel_id == ch_id]
            
            log_entry[f'states_ch_{ch_id}'] = [sta.state.value for sta in channel_stas]
            log_entry[f'backoff_ch_{ch_id}'] = [sta.backoff_counter for sta in channel_stas]
            log_entry[f'backoff_stage_ch_{ch_id}'] = [sta.backoff_stage for sta in channel_stas]
            log_entry[f'tx_attempts_ch_{ch_id}'] = [sta.tx_attempt for sta in channel_stas]
            log_entry[f'queue_len_ch_{ch_id}'] = [len(sta.tx_queue) for sta in channel_stas]
        
        self.logs.append(log_entry)
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        stats = {
            'total_slots': self.current_slot,
            'total_time_us': self.current_slot * SLOTTIME,
            'stations': {}
        }
        
        for sta in self.stations:
            stats['stations'][sta.sta_id] = {
                'channel': sta.channel_id,
                'successful_transmissions': sta.successful_transmissions,
                'collisions': sta.collision_count,
                'total_attempts': sta.total_attempts,
                'success_rate': sta.successful_transmissions / max(1, sta.total_attempts),
                'final_state': sta.state.value,
                'final_backoff_stage': sta.backoff_stage
            }
        
        return stats