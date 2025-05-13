import math
import numpy as np

from scalesim.memory.read_port import read_port
from typing import List, Set #ds
from typing import Optional, Tuple
class ReadBufferEstimateBw:
    # def __init__(self):
    #     # Buffer parameters
    #     self.word_size = 1
    #     self.active_buf_frac = 0.5
    #     self.total_size_bytes = 1
    #     self.total_size_elems = 1
    #     self.active_buf_size = 1
    #     self.prefetch_buf_size = 1

    #     self.hit_latency = 1

    #     # Backing buffer parameters
    #     self.backing_buffer = read_port()
    #     self.default_bandwidth = 1
    #     self.prefetch_bandwidth = 1

    #     # Access counts
    #     self.num_access = 0

    #     # Trace matrix
    #     self.trace_matrix = np.ones((1, 1))

    #     # Tracking variables
    #     self.num_items_per_set = -1
    #     self.elems_current_set = 0
    #     self.current_set_id = 0
    #     self.read_buffer_set_start_id = -1
    #     self.read_buffer_set_end_id = -1
    #     self.prefetch_buffer_set_start_id = -1
    #     self.prefetch_buffer_set_end_id = -1
    #     self.last_prefetch_start_cycle = -2
    #     self.last_prefetch_end_cycle = -1
    #     self.first_request_rcvd_cycle = 0

    #     # Internal data structures
    #     self.current_set = set()
    #     self.list_of_sets = []
    #     self.num_sets_active_buffer = 1
    #     self.num_sets_prefetch_buffer = 1

    #     # Flags
    #     self.first_request_seen = False
    #     self.params_set_flag = False
    #     self.active_buffer_prefetch_done = False
    #     self.trace_valid = False
    # ds1
    # def __init__(self):
    #     """Initialize buffer with default parameters and state."""
    #     # Buffer configuration parameters
    #     self.word_size = 1                     # Size of each element in bytes
    #     self.active_buf_frac = 0.5             # Fraction of buffer for active portion
    #     self.total_size_bytes = 1              # Total buffer capacity in bytes
    #     self.total_size_elems = 1              # total_size_bytes // word_size
    #     self.active_buf_size = 1               # active_buf_frac * total_size_elems
    #     self.prefetch_buf_size = 1             # total_size_elems - active_buf_size
        
    #     # Timing parameters
    #     self.hit_latency = 1                   # Cycles for buffer hit
        
    #     # Backing interface configuration
    #     # from read_port import read_port        # Avoid circular imports
    #     self.backing_buffer = read_port()
    #     self.default_bandwidth = 1             # Words per cycle (normal ops)
    #     self.prefetch_bandwidth = 1            # Words per cycle (prefetch)
        
    #     # Access tracking
    #     self.num_access = 0                    # Total access count
        
    #     # Trace system initialization
    #     self.trace_matrix = np.ones((1, 1), dtype=np.int32)  # Cycle, address...
        
    #     # Prefetch tracking variables
    #     self.items_per_set = 0                 # Elements per prefetch set
    #     # self.current_set_elements = 0          # Count of elements in current set
    #     self.elems_current_set = 0
    #     self.current_set_id = 0                # ID for set tracking
    #     self.read_set_id = 0                   # Current read set ID
    #     self.prefetch_set_id = 0               # Current prefetch set ID
        
    #     # Cycle tracking
    #     self.prefetch_start_cycle = -1         # Start of prefetch ops
    #     self.prefetch_end_cycle = -1           # End of prefetch ops
    #     self.first_request_cycle = -1           # First request received
        
    #     # Data structures
    #     self.current_set: Set[int] = set()      # Elements in current set
    #     self.list_of_sets: List[Set[int]] = []  # All completed sets
    #     self.num_active_sets = 0               # Sets in active buffer
    #     self.num_prefetch_sets = 0             # Sets in prefetch buffer
        
    #     # Status flags
    #     self.first_request_received = False    # First request flag
    #     self.params_set = False                # Parameters configured flag
    #     self.active_buf_prefetched = False     # Active buffer ready flag
    #     self.trace_valid = False               # Trace data validity
    #     self.active_buffer_prefetch_done = False
    #
    def __init__(self):
        """Initialize buffer with default parameters and state tracking."""
        # Buffer configuration parameters
        self.word_size = 1                     # Size of each element in bytes
        self.active_buf_frac = 0.5            # Fraction of buffer for active portion
        self.total_size_bytes = 1             # Total buffer capacity in bytes
        self.total_size_elems = 1             # total_size_bytes // word_size
        self.active_buf_size = 1              # active_buf_frac * total_size_elems
        self.prefetch_buf_size = 1            # total_size_elems - active_buf_size
        
        # Timing parameters
        self.hit_latency = 1                  # Cycles for buffer hit
        
        # Backing interface configuration
        # from read_port import read_port       # Local import to prevent circular dependencies
        self.backing_buffer = read_port()
        self.default_bandwidth = 1           # Words per cycle (normal operations)
        self.prefetch_bandwidth = 1          # Words per cycle (prefetch operations)
        
        # Access tracking
        self.num_access = 0                   # Total access count
        
        # Trace system initialization
        self.trace_matrix = np.ones((1, 1), dtype=np.int32)  # Cycle, address...
        
        # Prefetch tracking variables
        self.num_items_per_set = -1           # Will be calculated during parameter setting
        self.elems_current_set = 0           # Elements in current set
        self.current_set_id = 0               # Current set identifier
        self.read_set_start_id = -1           # Start of read buffer sets
        self.read_set_end_id = -1             # End of read buffer sets
        self.prefetch_buffer_set_start_id = -1       # Start of prefetch buffer sets
        self.prefetch_buffer_set_end_id = -1         # End of prefetch buffer sets
        
        # Cycle tracking
        self.last_prefetch_start_cycle = -2         # Start cycle of last prefetch
        self.last_prefetch_end_cycle = -1           # End cycle of last prefetch
        self.first_request_rcvd_cycle = 0          # Cycle of first received request
        
        # Data structures
        self.current_set: Set[int] = set()    # Elements in current set
        self.list_of_sets: List[Set[int]] = [] # All completed sets
        self.num_sets_active_buffer = 1       # Sets in active buffer
        self.num_sets_prefetch_buffer = 1     # Sets in prefetch buffer
        
        # Status flags
        self.first_request_seen = False   # First request flag
        self.params_set_flag = False          # Parameters configured flag
        self.active_buffer_prefetch_done = False # Active buffer prefetch complete
        self.trace_valid = False              # Trace data validity
    # def set_params(self, backing_buf_obj,
    #                total_size_bytes=1, word_size=1, active_buf_frac=0.9,
    #                hit_latency=1, backing_buf_default_bw=1):

    #     self.total_size_bytes = total_size_bytes
    #     self.word_size = word_size

    #     assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
    #     self.active_buf_frac = round(active_buf_frac, 2)
    #     self.hit_latency = hit_latency

    #     self.backing_buffer = backing_buf_obj
    #     self.default_bandwidth = backing_buf_default_bw
    #     self.prefetch_bandwidth = self.default_bandwidth

    #     # Calculate these based on the values provided
    #     self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
    #     self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
    #     self.prefetch_buf_size = self.total_size_elems - self.active_buf_size

    #     #
    #     self.num_items_per_set = math.floor(self.total_size_elems / 100)
    #     self.num_sets_active_buffer = int(self.active_buf_frac * 100)
    #     self.num_sets_prefetch_buffer = 100 - self.num_sets_active_buffer

    #     self.current_set = set()
    #     self.current_set_id = 0
    #     self.list_of_sets = []
    #     self.read_buffer_set_start_id = 0
    #     self.read_buffer_set_end_id = self.num_sets_active_buffer - 1
    #     self.last_prefetch_start_cycle = -2
    #     self.last_prefetch_end_cycle = -1  # TODO: Check what the correct value is

    #     #
    #     self.params_set_flag = True
    def set_params(self, 
                backing_buf_obj,
                total_size_bytes: int = 1,
                word_size: int = 1,
                active_buf_frac: float = 0.9,
                hit_latency: int = 1,
                backing_buf_default_bw: int = 1) -> None:
        """Configure buffer parameters and recalculate derived properties.
        
        Args:
            backing_buf_obj: Interface to backing memory
            total_size_bytes: Total buffer capacity in bytes
            word_size: Data word size in bytes
            active_buf_frac: Active buffer allocation fraction [0.5, 1)
            hit_latency: Cycles for buffer hit
            backing_buf_default_bw: Backing interface bandwidth
            
        Raises:
            AssertionError: For invalid parameter values
        """
        # 1. Set basic buffer properties
        self.total_size_bytes = total_size_bytes
        self.word_size = word_size
        
        # 2. Validate and set active buffer fraction
        assert 0.5 <= active_buf_frac < 1.0, "Active buffer fraction must be in [0.5, 1)"
        self.active_buf_frac = round(active_buf_frac, 2)
        
        # 3. Set timing parameters
        assert hit_latency > 0, "Hit latency must be positive"
        self.hit_latency = hit_latency
        
        # 4. Configure backing interface
        assert hasattr(backing_buf_obj, 'service_reads'), "Invalid backing buffer"
        self.backing_buffer = backing_buf_obj
        assert backing_buf_default_bw > 0, "Bandwidth must be positive"
        self.default_bandwidth = backing_buf_default_bw
        self.prefetch_bandwidth = backing_buf_default_bw  # Set equal by default
        
        # 5. Recalculate derived buffer properties
        self._recalculate_buffer_properties()
        
        # 6. Calculate set organization
        self.num_items_per_set = max(1, self.total_size_elems // 100)
        self.num_sets_active_buffer = int(self.active_buf_size / self.num_items_per_set)
        self.num_sets_prefetch_buffer = int(self.prefetch_buf_size / self.num_items_per_set)
        
        # 7. Initialize tracking structures
        self.current_set = set()
        self.current_set_id = 0
        self.list_of_sets = []
        self.read_buffer_set_start_id = 0
        self.read_buffer_set_end_id = self.num_sets_active_buffer - 1
        self.last_prefetch_start_cycle = -2
        self.last_prefetch_end_cycle = -1
        
        # 8. Mark parameters as configured
        self.params_set_flag = True

    def _recalculate_buffer_properties(self) -> None:
        """Internal method to update all size-dependent properties."""
        self.total_size_elems = self.total_size_bytes // self.word_size
        self.active_buf_size = int(self.active_buf_frac * self.total_size_elems)
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size
    #
    # def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
    #     assert self.params_set_flag, 'Parameters are not set yet'
    #     assert incoming_cycles_arr.shape[0] == incoming_requests_arr_np.shape[0], 'Incoming cycles and requests dont match'

    #     outcycles = incoming_cycles_arr + self.hit_latency  # In estimate mode, operation is stall free.
    #     # Therefore its always a hit

    #     # The following to track requests and maintain proper state of the buffer
    #     for i in range(incoming_requests_arr_np.shape[0]):
    #         cycle = int(incoming_cycles_arr[i][0])

    #         requests_this_cycle = incoming_requests_arr_np[i]
    #         if not self.first_request_seen:
    #             if max(requests_this_cycle) > -1:
    #                 self.first_request_rcvd_cycle = cycle
    #                 self.first_request_seen = True

    #         for addr in requests_this_cycle:
    #             if not addr == -1:
    #                 self.manage_prefetches(cycle, addr)

    #     return outcycles
# ICL SUCESS
    def service_reads(self, 
                    incoming_requests_arr_np: np.ndarray, 
                    incoming_cycles_arr: np.ndarray) -> np.ndarray:
        """
        Services read requests while managing prefetch operations and buffer state.
        
        Args:
            incoming_requests_arr_np: Array of read requests (num_cycles x bandwidth)
            incoming_cycles_arr: Array of cycle numbers for each request batch
            
        Returns:
            Array of completion cycles for each request batch
            
        Raises:
            AssertionError: For invalid parameters or shape mismatches
        """
        # --- Parameter Validation ---
        assert self.params_set_flag, "Parameters not set. Call set_params() first"
        assert incoming_requests_arr_np.shape[0] == incoming_cycles_arr.shape[0], \
            "Request and cycle arrays must have same length"
        
        # --- Calculate Output Cycles ---
        outcycles = incoming_cycles_arr + self.hit_latency
        
        # --- Process Each Cycle's Requests ---
        for i in range(incoming_requests_arr_np.shape[0]):
            cycle = incoming_cycles_arr[i]
            requests = incoming_requests_arr_np[i]
            
            # Track first valid request
            if not self.first_request_seen and requests[0] != -1:
                self.first_request_rcvd_cycle = cycle
                self.first_request_seen = True
            
            # Process each address in current cycle
            for addr in requests:
                if addr != -1:  # Skip null requests
                    self.manage_prefetches(cycle, addr)
        
        return outcycles

    # def service_reads(self,
    #                 incoming_requests_arr_np: np.ndarray,
    #                 incoming_cycles_arr: np.ndarray) -> np.ndarray:
    #     """Process read requests with prefetch management and cycle calculation.
        
    #     Args:
    #         incoming_requests_arr_np: 2D array of read requests (-1 for empty slots)
    #         incoming_cycles_arr: 1D array of arrival cycles
            
    #     Returns:
    #         np.ndarray: 1D array of completion cycles
            
    #     Raises:
    #         AssertionError: If parameters not configured or input shapes mismatch
    #     """
    #     # 1. Validate configuration and inputs
    #     assert self.params_set_flag, "Parameters must be configured before servicing reads"
    #     assert len(incoming_requests_arr_np) == len(incoming_cycles_arr), \
    #         "Requests and cycles arrays must have same length"
        
    #     # 2. Initialize output cycles with hit latency
    #     out_cycles = incoming_cycles_arr + self.hit_latency
        
    #     # 3. Process each request line
    #     for i, (cycle, req_line) in enumerate(zip(incoming_cycles_arr, incoming_requests_arr_np)):
    #         # Track first valid request
    #         if not self.first_request_seen:
    #             if any(addr != -1 for addr in req_line):
    #                 self.first_request_rcvd_cycle = cycle
    #                 self.first_request_seen = True
            
    #         # Process each address in request line
    #         for addr in req_line:
    #             if addr == -1:  # Skip empty slots
    #                 continue
                    
    #             # 4. Manage prefetch operations
    #             # stall_cycles = self.manage_prefetches(addr, cycle)
    #                         # Convert address to integer if needed
    #             addr_int = addr.item() if hasattr(addr, 'item') else int(addr)
    #             stall_cycles = self.manage_prefetches(cycle, addr_int)
    #             if stall_cycles > 0:
    #                 out_cycles[i:] += stall_cycles  # Apply stall to current and subsequent requests
        
    #     return out_cycles.reshape(-1, 1)  # Return as column vector

    # def service_reads(self, incoming_cycles_arr: np.ndarray, incoming_requests_arr_np: np.ndarray) -> np.ndarray:
    #     """
    #     Services incoming read requests, managing prefetch operations and maintaining buffer state.
        
    #     Args:
    #         incoming_cycles_arr: Array of cycles when each request arrives
    #         incoming_requests_arr_np: 2D array of requests (each row contains addresses for one cycle)
            
    #     Returns:
    #         Array of output cycles when each request will be serviced
            
    #     Raises:
    #         AssertionError: If parameters aren't set or input arrays don't match
    #     """
    #     # Step 1: Assert parameter configuration
    #     assert self.params_set_flag, "Buffer parameters must be set before servicing reads"
    #     assert len(incoming_cycles_arr) == incoming_requests_arr_np.shape[0], \
    #         "Number of cycles must match number of request entries"
        
    #     # Step 2: Calculate output cycles (base case - just add hit latency)
    #     output_cycles = incoming_cycles_arr + self.hit_latency
        
    #     # Step 3: Process each request
    #     for i, (cycle, requests) in enumerate(zip(incoming_cycles_arr, incoming_requests_arr_np)):
    #         # Track first valid request if not already seen
    #         if not self.first_request_seen:
    #             valid_requests = [addr for addr in requests if addr != -1]
    #             if valid_requests:
    #                 self.first_request_rcvd_cycle = cycle
    #                 self.first_request_seen = True
            
    #         # Process each valid address in the request
    #         for addr in requests:
    #             if addr != -1:  # -1 indicates no request
    #                 self.manage_prefetches(cycle, addr)
        
    #     # Step 4: Return output cycles
    #     return output_cycles

    #
    # def manage_prefetches(self, cycle, addr):

    #     # If this is a new address, otherwise its a hit
    #     if self.check_hit(addr):
    #         return

    #     if addr not in self.current_set:
    #         self.current_set.add(addr)
    #         self.elems_current_set += 1

    #         if self.elems_current_set == self.num_items_per_set:
    #             self.list_of_sets += [self.current_set]
    #             self.current_set = set()
    #             self.elems_current_set = 0
    #             self.current_set_id += 1

    #             if self.current_set_id == self.read_buffer_set_end_id + 1:  # This should be prefetched
    #                 if not self.active_buffer_prefetch_done:
    #                     self.prefetch_bandwidth = self.default_bandwidth
    #                     self.last_prefetch_end_cycle = self.first_request_rcvd_cycle - 1 - self.backing_buffer.get_latency()

    #                     cycles_needed = (self.num_sets_prefetch_buffer * self.num_items_per_set) \
    #                                     / self.prefetch_bandwidth
    #                     cycles_needed = math.ceil(cycles_needed)

    #                     self.last_prefetch_start_cycle = self.last_prefetch_end_cycle - cycles_needed + 1

    #                     self.prefetch()
    #                     self.prefetch_buffer_set_start_id =self.read_buffer_set_end_id + 1
    #                     self.prefetch_buffer_set_end_id = self.prefetch_buffer_set_start_id + \
    #                                                       self.num_sets_prefetch_buffer - 1
    #                     self.active_buffer_prefetch_done = True

    #                 else:
    #                     elems_to_prefetch = self.num_sets_prefetch_buffer * self.num_items_per_set
    #                     cycles_needed = self.last_prefetch_end_cycle - self.last_prefetch_start_cycle + 1
    #                     self.prefetch_bandwidth = math.ceil(elems_to_prefetch / cycles_needed)
    #                     self.prefetch()
    #                     self.prefetch_buffer_set_start_id += self.num_sets_prefetch_buffer
    #                     self.prefetch_buffer_set_end_id += self.num_sets_prefetch_buffer
                    
    #                 #Solving memory leak by discarding sets that are no longer in use
    #                 i = self.read_buffer_set_start_id
    #                 for j in range(self.num_sets_prefetch_buffer):
    #                     self.list_of_sets[i+j] = None


    #                 self.read_buffer_set_start_id += self.num_sets_prefetch_buffer
    #                 self.read_buffer_set_end_id += self.num_sets_prefetch_buffer
    #                 self.last_prefetch_start_cycle = self.last_prefetch_end_cycle +1
    #                 self.last_prefetch_end_cycle = cycle
    # ICL SUCESS
    def manage_prefetches(self, cycle: int, addr: int) -> None:
        """
        Manages prefetch operations including hit/miss handling, set management,
        and prefetch triggering based on buffer state.
        
        Args:
            cycle: Current simulation cycle
            addr: Memory address being accessed
        """
        # --- Hit Check ---
        if self.check_hit(addr):
            return
        
        # --- Miss Handling ---
        if addr not in self.current_set:
            self.current_set.add(addr)
            self.elems_current_set += 1
            
            # --- Set Full Condition ---
            if self.elems_current_set == self.num_items_per_set:
                self.list_of_sets.append(self.current_set)
                self.current_set = set()
                self.elems_current_set = 0
                self.current_set_id += 1
                
                # --- Prefetch Trigger ---
                if self.current_set_id == self.prefetch_buffer_set_end_id + 1:
                    if not self.active_buffer_prefetch_done:
                        # Active buffer prefetch
                        self.prefetch_bandwidth = self.default_bandwidth
                        prefetch_cycles = (self.num_sets_active_buffer * 
                                        self.num_items_per_set) // self.prefetch_bandwidth
                        self.last_prefetch_end_cycle = cycle + prefetch_cycles
                        self.prefetch()
                        self.prefetch_buffer_set_start_id = self.num_sets_active_buffer
                        self.prefetch_buffer_set_end_id = (self.num_sets_active_buffer + 
                                                        self.num_sets_prefetch_buffer - 1)
                        self.active_buffer_prefetch_done = True
                    else:
                        # Prefetch buffer continuation
                        num_elems = (self.prefetch_buffer_set_end_id - 
                                    self.prefetch_buffer_set_start_id + 1) * self.num_items_per_set
                        prefetch_cycles = num_elems // self.prefetch_bandwidth
                        self.prefetch_bandwidth = min(self.prefetch_bandwidth * 2, 
                                                    self.max_bandwidth)
                        self.prefetch()
                        self.prefetch_buffer_set_start_id = self.prefetch_buffer_set_end_id + 1
                        self.prefetch_buffer_set_end_id = (self.prefetch_buffer_set_start_id + 
                                                        self.num_sets_prefetch_buffer - 1)
                    
                    # --- Cleanup Old Sets ---
                    discard_start = max(0, self.read_set_start_id - self.num_sets_prefetch_buffer)
                    for i in range(discard_start, self.read_set_start_id):
                        self.list_of_sets[i] = None
                    
                    # --- Update Read Buffer ---
                    self.read_set_start_id = self.prefetch_buffer_set_start_id - self.num_sets_active_buffer
                    self.read_set_end_id = self.read_set_start_id + self.num_sets_active_buffer - 1
                    
                    # --- Update Cycle Tracking ---
                    self.last_prefetch_start_cycle = cycle
                    self.last_prefetch_end_cycle = cycle + prefetch_cycles

    # def manage_prefetches(self, cycle: int, addr: int) -> int:
    #     """Manage prefetch operations for a given address.
        
    #     Args:
    #         cycle: Current simulation cycle
    #         addr: Memory address to manage
            
    #     Returns:
    #         int: Additional stall cycles required (0 if no stall)
            
    #     Operations:
    #         1. Handles address insertion into current set
    #         2. Manages set transitions
    #         3. Controls prefetch operations
    #         4. Updates buffer state tracking
    #     """
    #     stall_cycles = 0
        
    #     # 1. Convert address to integer if it's a numpy type
    #     if hasattr(addr, 'item'):  # Check if it's a numpy scalar
    #         addr = addr.item()        
    #     # 1. Check for existing hit
    #     if self.check_hit(addr):
    #         return stall_cycles
            
    #     # 2. Add new address to current set
    #     if addr not in self.current_set:
    #         self.current_set.add(addr)
    #         self.elems_current_set += 1
            
    #     # 3. Check set completeness
    #     if self.elems_current_set == self.num_items_per_set:
    #         self.list_of_sets.append(self.current_set)
    #         self.current_set = set()
    #         self.elems_current_set = 0
    #         self.current_set_id += 1
            
    #     # 4. Handle prefetch conditions
    #     if self.current_set_id == self.read_set_end_id:
    #         if not self.active_buffer_prefetch_done:
    #             # Initial prefetch setup
    #             self.prefetch_bandwidth = self.default_bandwidth
    #             prefetch_sets = self.num_sets_prefetch_buffer
    #             cycles_needed = (prefetch_sets * self.num_items_per_set) // self.prefetch_bandwidth
                
    #             self.prefetch(
    #                 start_cycle=max(cycle, self.first_request_cycle),
    #                 num_sets=prefetch_sets
    #             )
                
    #             self.prefetch_set_start_id = self.read_set_end_id
    #             self.prefetch_set_end_id = self.read_set_end_id + prefetch_sets
    #             self.active_buffer_prefetch_done = True
    #             stall_cycles = cycles_needed
                
    #         else:
    #             # Ongoing prefetch operations
    #             elems_to_prefetch = self.num_items_per_set - self.elems_current_set
    #             adjusted_bandwidth = min(
    #                 self.default_bandwidth,
    #                 max(1, elems_to_prefetch // 2)  # Ensure at least 1
    #             )
    #             cycles_needed = elems_to_prefetch // adjusted_bandwidth
                
    #             self.prefetch(
    #                 start_cycle=cycle,
    #                 num_sets=1
    #             )
                
    #             self.prefetch_set_start_id = self.read_set_end_id
    #             self.prefetch_set_end_id = self.read_set_end_id + 1
    #             stall_cycles = cycles_needed
        
    #     # 5. Memory management
    #     if len(self.list_of_sets) > 2 * (self.num_sets_active_buffer + self.num_sets_prefetch_buffer):
    #         # Discard oldest sets beyond buffer capacity
    #         excess = len(self.list_of_sets) - (self.num_sets_active_buffer + self.num_sets_prefetch_buffer)
    #         self.list_of_sets = self.list_of_sets[excess:]
    #         self.read_set_start_id += excess
    #         self.prefetch_set_start_id += excess
        
    #     # 6. Update cycle tracking
    #     self.last_prefetch_start = cycle
    #     self.last_prefetch_end = cycle + stall_cycles
        
    #     return stall_cycles
# ds sucess
    # def manage_prefetches(self, cycle: int, addr: int) -> None:
    #     """
    #     Manages prefetch operations for a given address, updating buffer state and handling prefetch requests.
        
    #     Args:
    #         cycle: Current simulation cycle
    #         addr: Memory address to manage prefetching for
    #     """
    #     # 1. Check for Cache Hit
    #     if self.check_hit(addr):
    #         return

    #     # 2. Manage Set Organization and Prefetch Operations
    #     if addr not in self.current_set:
    #         # Add new address to current set
    #         self.current_set.add(addr)
    #         self.elems_current_set += 1

    #         # Check if current set is full
    #         if self.elems_current_set == self.num_items_per_set:
    #             # Complete current set and start new one
    #             self.list_of_sets.append(self.current_set)
    #             self.current_set = set()
    #             self.elems_current_set = 0
    #             self.current_set_id += 1

    #             # Check prefetch trigger condition
    #             if self.current_set_id == self.read_buffer_set_end_id + 1:
    #                 if not self.active_buffer_prefetch_done:
    #                     # Initial prefetch setup
    #                     self.prefetch_bandwidth = self.default_bandwidth
    #                     prefetch_cycles = max(1, self.num_sets_prefetch_buffer // self.prefetch_bandwidth)
    #                     self.last_prefetch_start_cycle = cycle
    #                     self.last_prefetch_end_cycle = cycle + prefetch_cycles
    #                     # self.prefetch(self.num_sets_prefetch_buffer)
    #                     self.prefetch()
    #                     self.prefetch_set_start_id = self.current_set_id
    #                     self.prefetch_set_end_id = self.current_set_id + self.num_sets_prefetch_buffer - 1
    #                     self.active_buffer_prefetch_done = True
    #                 else:
    #                     # Extended prefetch handling
    #                     remaining_elements = (self.total_size_elems - 
    #                                         len(self.list_of_sets) * self.num_items_per_set)
    #                     elements_to_prefetch = min(self.num_sets_prefetch_buffer, remaining_elements)
                        
    #                     if elements_to_prefetch > 0:
    #                         # Dynamic bandwidth adjustment
    #                         cycles_since_last = cycle - self.last_prefetch_end
    #                         effective_bw = max(1, min(self.default_bandwidth, 
    #                                             elements_to_prefetch // max(1, cycles_since_last)))
    #                         self.prefetch_bandwidth = effective_bw
    #                         # self.prefetch(elements_to_prefetch)
    #                         self.prefetch()
    #                         self.prefetch_set_start_id = self.prefetch_set_end_id + 1
    #                         self.prefetch_set_end_id = self.prefetch_set_start_id + elements_to_prefetch - 1

    #                 # 2.1 Memory Cleanup
    #                 if len(self.list_of_sets) > (self.num_sets_active_buffer + self.num_sets_prefetch_buffer):
    #                     sets_to_keep = self.num_sets_active_buffer + self.num_sets_prefetch_buffer
    #                     first_set_to_keep = len(self.list_of_sets) - sets_to_keep
    #                     for i in range(first_set_to_keep):
    #                         self.list_of_sets[i] = None

    #                 # 2.2 Update Pointers for Next Cycle
    #                 if self.active_buffer_prefetch_done:
    #                     self.read_buffer_set_start_id = self.prefetch_set_start_id - self.num_sets_active_buffer
    #                     self.read_buffer_set_end_id = self.prefetch_set_start_id - 1
    #                     self.last_prefetch_start = cycle
    #                     prefetch_cycles = max(1, (self.prefetch_set_end_id - self.prefetch_set_start_id + 1) // 
    #                                         self.prefetch_bandwidth)
    #                     self.last_prefetch_end = cycle + prefetch_cycles


    #
    # def check_hit(self, addr):
    #     assert self.params_set_flag, 'Parameters are not set yet'

    #     start_set_idx = self.read_buffer_set_start_id
    #     end_set_idx = min(self.current_set_id, self.read_buffer_set_end_id + 1)

    #     if start_set_idx == end_set_idx:
    #         return False

    #     for idx in range(start_set_idx, end_set_idx):
    #         if addr in self.list_of_sets[idx]:
    #             return True

        # return False
    def check_hit(self, addr: int) -> bool:
        """Check if an address exists in the read buffer.
        
        Args:
            addr: Memory address to check
            
        Returns:
            bool: True if address found (hit), False otherwise
            
        Raises:
            AssertionError: If parameters not configured
        """
        # 1. Validate configuration
        assert self.params_set_flag, "Parameters must be configured before checking hits"
        
        # 2. Calculate set range to check
        start_idx = self.read_set_start_id
        end_idx = self.read_set_end_id
        
        # 3. Handle empty buffer case
        if start_idx == end_idx:
            return False
            
        # 4. Check sets in current range
        for set_id in range(start_idx, end_idx):
            if set_id < len(self.list_of_sets) and addr in self.list_of_sets[set_id]:
                return True
                
        # 5. Address not found
        return False

    #
    # def complete_all_prefetches(self):
    #     assert self.params_set_flag, 'Parameters are not set yet'

    #     current_set_elems = list(self.current_set)
    #     if len(current_set_elems) > 0:
    #         self.list_of_sets += [self.current_set]
    #     else:
    #         self.current_set_id -= 1    # If there are no elems in this set, dont consider it

    #     if not self.active_buffer_prefetch_done:
    #         self.prefetch_bandwidth = self.default_bandwidth
    #         # self.last_prefetch_end_cycle = -1 - self.backing_buffer.get_latency()

    #         num_sets_to_prefetch = self.current_set_id + 1
    #         self.num_sets_active_buffer = num_sets_to_prefetch

    #         cycles_needed = (num_sets_to_prefetch * self.num_items_per_set) \
    #                         / self.prefetch_bandwidth
    #         cycles_needed = math.ceil(cycles_needed)

    #         self.last_prefetch_start_cycle = self.last_prefetch_end_cycle - cycles_needed + 1

    #         self.prefetch()
    #         self.active_buffer_prefetch_done = True
    #     else:
    #         num_sets_to_prefetch = self.current_set_id - self.prefetch_buffer_set_start_id + 1
    #         self.prefetch_buffer_set_end_id = self.current_set_id
    #         elems_to_prefetch = num_sets_to_prefetch * self.num_items_per_set
    #         cycles_needed = self.last_prefetch_end_cycle - self.last_prefetch_start_cycle + 1
    #         self.prefetch_bandwidth = math.ceil(elems_to_prefetch / cycles_needed)
    #         self.prefetch()

    # ICL failed1
    # def complete_all_prefetches(self) -> None:
    #     """
    #     Completes all pending prefetch operations based on current buffer state.
    #     Handles both active and prefetch buffer regions.
    #     """
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Handle Current Set ---
    #     current_set_list = list(self.current_set)
    #     if current_set_list:  # Non-empty current set
    #         self.list_of_sets.append(set(current_set_list))
    #     else:  # Empty current set
    #         self.current_set_id = max(0, self.current_set_id - 1)
        
    #     # --- Execute Prefetches ---
    #     if not self.active_buffer_prefetch_done:
    #         # Active buffer prefetch
    #         self.prefetch_bandwidth = self.default_bandwidth
    #         num_sets_to_prefetch = self.current_set_id + 1
    #         self.num_sets_active_buffer = num_sets_to_prefetch
    #         cycles_needed = (num_sets_to_prefetch * self.items_per_set) // self.prefetch_bandwidth
    #         self.last_prefetch_start_cycle = self.current_cycle
    #         self.prefetch()
    #         self.active_buffer_prefetch_done = True
    #     else:
    #         # Prefetch buffer continuation
    #         num_sets_to_prefetch = self.current_set_id - self.prefetch_buffer_set_start_id + 1
    #         self.prefetch_buffer_set_end_id = self.current_set_id
    #         elems_to_prefetch = num_sets_to_prefetch * self.items_per_set
    #         cycles_needed = elems_to_prefetch // self.prefetch_bandwidth
    #         self.prefetch_bandwidth = min(self.prefetch_bandwidth * 2, self.max_bandwidth)
    #         self.prefetch()
        
    #     # --- Update Cycle Tracking ---
    #     self.last_prefetch_end_cycle = self.current_cycle + cycles_needed - 1
    #     self.current_cycle = self.last_prefetch_end_cycle + 1

    def complete_all_prefetches(self) -> None:
        """
        Completes all pending prefetch operations based on current buffer state.
        Handles both active and prefetch buffer regions using the correct attribute names.
        """
        # --- Parameter Validation ---
        assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
        # --- Handle Current Set ---
        current_set_list = list(self.current_set)
        if current_set_list:  # Non-empty current set
            self.list_of_sets.append(set(current_set_list))
        else:  # Empty current set
            self.current_set_id = max(0, self.current_set_id - 1)
        
        # --- Execute Prefetches ---
        if not self.active_buffer_prefetch_done:
            # Active buffer prefetch
            self.prefetch_bandwidth = self.default_bandwidth
            num_sets_to_prefetch = self.current_set_id + 1
            self.num_sets_active_buffer = num_sets_to_prefetch
            cycles_needed = (num_sets_to_prefetch * self.num_items_per_set) // self.prefetch_bandwidth
            self.last_prefetch_start_cycle = self.current_cycle if hasattr(self, 'current_cycle') else 0
            self.prefetch()
            self.active_buffer_prefetch_done = True
        else:
            # Prefetch buffer continuation
            num_sets_to_prefetch = self.current_set_id - self.prefetch_buffer_set_start_id + 1
            self.prefetch_buffer_set_end_id = self.current_set_id
            elems_to_prefetch = num_sets_to_prefetch * self.num_items_per_set
            cycles_needed = elems_to_prefetch // self.prefetch_bandwidth
            self.prefetch_bandwidth = min(self.prefetch_bandwidth * 2, self.max_bandwidth) if hasattr(self, 'max_bandwidth') else self.prefetch_bandwidth
            self.prefetch()
        
        # --- Update Cycle Tracking ---
        if hasattr(self, 'current_cycle'):
            self.last_prefetch_end_cycle = self.current_cycle + cycles_needed - 1
            self.current_cycle = self.last_prefetch_end_cycle + 1

    # def complete_all_prefetches(self) -> None:
    #     """Complete all pending prefetch operations to fully populate the buffer.
        
    #     Operations:
    #         1. Finalizes current partial set
    #         2. Executes remaining prefetches
    #         3. Updates all buffer state tracking
            
    #     Raises:
    #         AssertionError: If parameters not configured
    #     """
    #     # 1. Validate configuration
    #     assert self.params_set_flag, "Parameters must be configured before completing prefetches"
        
    #     # 2. Handle current partial set
    #     if self.current_set:
    #         self.list_of_sets.append(self.current_set)
    #         self.current_set = set()
    #         self.current_set_id += 1
    #     elif self.current_set_id > 0:
    #         self.current_set_id -= 1  # Adjust for empty current set
        
    #     # 3. Calculate remaining prefetches
    #     if not self.active_buffer_prefetch_done:
    #         # Initial prefetch setup
    #         self.prefetch_bandwidth = self.default_bandwidth
    #         sets_to_prefetch = self.num_sets_prefetch_buffer
    #         cycles_needed = (sets_to_prefetch * self.num_items_per_set) // self.prefetch_bandwidth
            
    #         self.prefetch(
    #             # start_cycle=max(self.last_prefetch_end + 1, self.first_request_cycle),
    #             # num_sets=sets_to_prefetch
    #         )
            
    #         self.prefetch_set_start_id = self.read_set_end_id
    #         self.prefetch_set_end_id = self.read_set_end_id + sets_to_prefetch
    #         self.last_prefetch_end_cycle += cycles_needed
    #         self.active_buffer_prefetch_done = True
    #     else:
    #         # Complete remaining prefetches
    #         sets_to_prefetch = self.current_set_id - self.prefetch_set_start_id
    #         if sets_to_prefetch > 0:
    #             elems_to_prefetch = sets_to_prefetch * self.num_items_per_set
    #             adjusted_bandwidth = min(
    #                 self.default_bandwidth,
    #                 max(1, elems_to_prefetch // 2)  # Ensure minimum bandwidth
    #             )
    #             cycles_needed = elems_to_prefetch // adjusted_bandwidth
                
    #             # self.prefetch(
    #             #     start_cycle=self.last_prefetch_end + 1,
    #             #     num_sets=sets_to_prefetch
    #             # )
    #             self.prefetch()
                
    #             self.prefetch_set_end_id = self.current_set_id
    #             self.last_prefetch_end += cycles_needed
        
    #     # 4. Update buffer state
    #     self.read_set_start_id = 0
    #     self.read_set_end_id = self.current_set_id


    #
    # def prefetch(self):
    #     assert self.params_set_flag, 'Parameters are not set yet'

    #     if not self.active_buffer_prefetch_done:
    #         start_set_idx = 0
    #         end_set_idx = self.num_sets_active_buffer - 1
    #     else:
    #         start_set_idx = self.prefetch_buffer_set_start_id
    #         end_set_idx = self.prefetch_buffer_set_end_id

    #     all_addresses = []
    #     for idx in range(start_set_idx, end_set_idx + 1):
    #         this_set = self.list_of_sets[idx]
    #         all_addresses += list(this_set)

    #     self.num_access += len(all_addresses)

    #     cycles_needed = self.last_prefetch_end_cycle - self.last_prefetch_start_cycle + 1
    #     max_prefetch_capacity = cycles_needed * self.prefetch_bandwidth

    #     delta = max_prefetch_capacity - len(all_addresses)

    #     if delta > 0:
    #         for _ in range(delta):
    #             all_addresses += [-1]

    #     prefetch_requests = np.asarray(all_addresses).reshape((cycles_needed, self.prefetch_bandwidth))

    #     cycles_arr = np.zeros((cycles_needed,1))
    #     for i in range(cycles_arr.shape[0]):
    #         cycles_arr[i][0] = self.last_prefetch_start_cycle + i

    #     response_cycles_arr = self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr,
    #                                                             incoming_requests_arr_np=prefetch_requests)

    #     # Create / add elements to the trace matrix
    #     this_prefetch_traces = np.concatenate((response_cycles_arr, prefetch_requests), axis=1)

    #     if not self.trace_valid:
    #         self.trace_matrix = this_prefetch_traces
    #         self.trace_valid = True

    #     else:
    #         del_cols = self.trace_matrix.shape[1] - this_prefetch_traces.shape[1]
    #         if del_cols > 0:
    #             empty_cols = np.ones((this_prefetch_traces.shape[0], del_cols))
    #             this_prefetch_traces = np.concatenate((this_prefetch_traces, empty_cols), axis=1)

    #         elif del_cols < 0:
    #             del_cols = int(-1 * del_cols)
    #             empty_cols = np.ones((self.trace_matrix.shape[0], del_cols))
    #             self.trace_matrix = np.concatenate((self.trace_matrix, empty_cols), axis=1)

    #         self.trace_matrix = np.concatenate((self.trace_matrix, this_prefetch_traces), axis=0)
# ICL failed1
    # def prefetch(self) -> None:
    #     """
    #     Manages the complete prefetching process including:
    #     - Address collection from sets
    #     - Cycle calculation and request shaping
    #     - Prefetch execution and trace maintenance
    #     """
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Determine Prefetch Range ---
    #     if not self.active_buffer_prefetch_done:
    #         set_start = 0
    #         set_end = self.num_sets_active_buffer - 1
    #         self.active_buffer_prefetch_done = True
    #     else:
    #         set_start = self.prefetch_buffer_set_start_id
    #         set_end = self.prefetch_buffer_set_end_id
        
    #     # --- Collect Addresses from Sets ---
    #     all_addresses = []
    #     for set_id in range(set_start, set_end + 1):
    #         if set_id < len(self.list_of_sets) and self.list_of_sets[set_id] is not None:
    #             all_addresses.extend(list(self.list_of_sets[set_id]))
        
    #     if not all_addresses:  # Early exit if nothing to prefetch
    #         return
        
    #     # --- Prepare Prefetch Batch ---
    #     self.num_access += len(all_addresses)
    #     cycles_needed = (len(all_addresses) + self.prefetch_bandwidth - 1) // self.prefetch_bandwidth
    #     max_prefetch = cycles_needed * self.prefetch_bandwidth
    #     delta = max_prefetch - len(all_addresses)
        
    #     if delta > 0:  # Pad if needed
    #         all_addresses.extend([-1] * delta)
        
    #     # --- Shape and Execute Requests ---
    #     prefetch_requests = np.array(all_addresses, dtype=np.int32).reshape(
    #         (cycles_needed, self.prefetch_bandwidth))
        
    #     cycles_arr = np.arange(
    #         self.current_cycle,
    #         self.current_cycle + cycles_needed,
    #         dtype=np.int64
    #     )
    #     response_cycles = self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr)
        
    #     # --- Update Trace Matrix ---
    #     this_prefetch_traces = np.column_stack((response_cycles, prefetch_requests))
        
    #     if not self.trace_valid:
    #         self.trace_matrix = this_prefetch_traces
    #         self.trace_valid = True
    #     else:
    #         # Handle column mismatch
    #         if this_prefetch_traces.shape[1] < self.trace_matrix.shape[1]:
    #             padding = np.full(
    #                 (this_prefetch_traces.shape[0], 
    #                 self.trace_matrix.shape[1] - this_prefetch_traces.shape[1]),
    #                 -1
    #             )
    #             this_prefetch_traces = np.hstack((this_prefetch_traces, padding))
    #         elif this_prefetch_traces.shape[1] > self.trace_matrix.shape[1]:
    #             padding = np.full(
    #                 (self.trace_matrix.shape[0],
    #                 this_prefetch_traces.shape[1] - self.trace_matrix.shape[1]),
    #                 -1
    #             )
    #             self.trace_matrix = np.hstack((self.trace_matrix, padding))
            
    #         self.trace_matrix = np.vstack((self.trace_matrix, this_prefetch_traces))
        
    #     # --- Update Cycle Tracking ---
    #     self.current_cycle += cycles_needed
# ICL sucess2
    def prefetch(self) -> None:
        """
        Manages the complete prefetching process including:
        - Address collection from sets
        - Cycle calculation and request shaping
        - Prefetch execution and trace maintenance
        """
        # --- Parameter Validation ---
        assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
        # Initialize current_cycle if not exists
        if not hasattr(self, 'current_cycle'):
            self.current_cycle = 0
        
        # --- Determine Prefetch Range ---
        if not self.active_buffer_prefetch_done:
            set_start = 0
            set_end = self.num_sets_active_buffer - 1
            self.active_buffer_prefetch_done = True
        else:
            set_start = self.prefetch_buffer_set_start_id
            set_end = self.prefetch_buffer_set_end_id
        
        # --- Collect Addresses from Sets ---
        all_addresses = []
        for set_id in range(set_start, set_end + 1):
            if set_id < len(self.list_of_sets) and self.list_of_sets[set_id] is not None:
                all_addresses.extend(list(self.list_of_sets[set_id]))
        
        if not all_addresses:  # Early exit if nothing to prefetch
            return
        
        # --- Prepare Prefetch Batch ---
        self.num_access += len(all_addresses)
        cycles_needed = (len(all_addresses) + self.prefetch_bandwidth - 1) // self.prefetch_bandwidth
        max_prefetch = cycles_needed * self.prefetch_bandwidth
        delta = max_prefetch - len(all_addresses)
        
        if delta > 0:  # Pad if needed
            all_addresses.extend([-1] * delta)
        
        # --- Shape and Execute Requests ---
        prefetch_requests = np.array(all_addresses, dtype=np.int32).reshape(
            (cycles_needed, self.prefetch_bandwidth))
        
        cycles_arr = np.arange(
            self.current_cycle,
            self.current_cycle + cycles_needed,
            dtype=np.int64
        )
        response_cycles = self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr)
        
        # --- Update Trace Matrix ---
        this_prefetch_traces = np.column_stack((response_cycles, prefetch_requests))
        
        if not self.trace_valid:
            self.trace_matrix = this_prefetch_traces
            self.trace_valid = True
        else:
            # Handle column mismatch
            if this_prefetch_traces.shape[1] < self.trace_matrix.shape[1]:
                padding = np.full(
                    (this_prefetch_traces.shape[0], 
                    self.trace_matrix.shape[1] - this_prefetch_traces.shape[1]),
                    -1
                )
                this_prefetch_traces = np.hstack((this_prefetch_traces, padding))
            elif this_prefetch_traces.shape[1] > self.trace_matrix.shape[1]:
                padding = np.full(
                    (self.trace_matrix.shape[0],
                    this_prefetch_traces.shape[1] - self.trace_matrix.shape[1]),
                    -1
                )
                self.trace_matrix = np.hstack((self.trace_matrix, padding))
            
            self.trace_matrix = np.vstack((self.trace_matrix, this_prefetch_traces))
        
        # --- Update Cycle Tracking ---
        self.current_cycle += cycles_needed

    # def prefetch(self) -> None:
    #     """
    #     Initiates prefetch operations by calculating necessary cycles, generating prefetch requests,
    #     and updating the trace matrix accordingly.
        
    #     Operations:
    #         1. Verifies parameter configuration
    #         2. Determines prefetch conditions and set ranges
    #         3. Collects addresses for prefetching
    #         4. Calculates prefetch cycles and capacity
    #         5. Generates and services prefetch requests
    #         6. Updates trace matrix with prefetch operations
    #     """
    #     # Step 1: Assert parameter configuration
    #     assert self.params_set_flag, "Parameters must be set before prefetching"
        
    #     # Step 2: Determine prefetch conditions
    #     if not self.active_buffer_prefetch_done:
    #         # Initial prefetch - use active buffer sets
    #         start_set_idx = self.read_buffer_set_start_id
    #         end_set_idx = self.read_buffer_set_end_id
    #     else:
    #         # Ongoing prefetch - use prefetch buffer sets
    #         start_set_idx = self.prefetch_set_start_id
    #         end_set_idx = self.prefetch_set_end_id
        
    #     # Step 3: Collect addresses from relevant sets
    #     all_addresses = []
    #     for set_idx in range(start_set_idx, end_set_idx + 1):
    #         if set_idx < len(self.list_of_sets) and self.list_of_sets[set_idx] is not None:
    #             all_addresses.extend(self.list_of_sets[set_idx])
        
    #     # Step 4: Update access count
    #     self.num_access += len(all_addresses)
        
    #     # Step 5: Calculate prefetch cycles
    #     cycles_needed = max(1, (self.last_prefetch_end_cycle - self.last_prefetch_start_cycle) + 1)
    #     max_prefetch_capacity = cycles_needed * self.prefetch_bandwidth
        
    #     # Step 6: Adjust address list to fit prefetch capacity
    #     if len(all_addresses) > max_prefetch_capacity:
    #         # Truncate if exceeds capacity
    #         all_addresses = all_addresses[:max_prefetch_capacity]
    #     elif len(all_addresses) < max_prefetch_capacity:
    #         # Pad with -1 if under capacity
    #         all_addresses.extend([-1] * (max_prefetch_capacity - len(all_addresses)))
        
    #     # Step 7: Generate prefetch requests
    #     prefetch_requests = np.array(all_addresses, dtype=np.int32)
    #     prefetch_requests = prefetch_requests.reshape((cycles_needed, self.prefetch_bandwidth))
        
    #     # Step 8: Calculate response cycles
    #     response_cycles = np.zeros((cycles_needed, 1), dtype=np.int32)
    #     response_cycles.fill(self.last_prefetch_start_cycle)
        
    #     # Step 9: Get service reads from backing buffer
    #     returned_cycles = self.backing_buffer.service_reads(response_cycles, prefetch_requests)
        
    #     # Step 10: Update trace matrix
    #     prefetch_traces = np.concatenate((returned_cycles, prefetch_requests), axis=1)
        
    #     # if not self.trace_valid:
    #     #     self.trace_matrix = prefetch_traces
    #     #     self.trace_valid = True
    #     # else:
    #     #     self.trace_matrix = np.concatenate((self.trace_matrix, prefetch_traces), axis=0)
    #     if not self.trace_valid:
    #         self.trace_matrix = prefetch_traces
    #         self.trace_valid = True
    #     else:
    #         # Ensure dimensions match before concatenation
    #         if self.trace_matrix.shape[1] != prefetch_traces.shape[1]:
    #             # If dimensions don't match, pad the smaller one with zeros
    #             max_cols = max(self.trace_matrix.shape[1], prefetch_traces.shape[1])
                
    #             if self.trace_matrix.shape[1] < max_cols:
    #                 pad_width = max_cols - self.trace_matrix.shape[1]
    #                 self.trace_matrix = np.pad(
    #                     self.trace_matrix, 
    #                     ((0, 0), (0, pad_width)), 
    #                     mode='constant', 
    #                     constant_values=-1
    #                 )
                
    #             if prefetch_traces.shape[1] < max_cols:
    #                 pad_width = max_cols - prefetch_traces.shape[1]
    #                 prefetch_traces = np.pad(
    #                     prefetch_traces,
    #                     ((0, 0), (0, pad_width)),
    #                     mode='constant',
    #                     constant_values=-1
    #                 )
            
    #         self.trace_matrix = np.concatenate((self.trace_matrix, prefetch_traces), axis=0)
# sucess ds
    # def prefetch(self) -> None:
    #     """
    #     Robust prefetch implementation with proper bandwidth calculations and trace handling.
    #     Ensures valid bandwidth values by:
    #     1. Proper cycle accounting
    #     2. Valid request counting
    #     3. Correct trace matrix dimensions
    #     """
    #     # 1. Parameter validation
    #     assert self.params_set_flag, "Buffer parameters not configured"
    #     if self.prefetch_bandwidth <= 0:
    #         raise ValueError("Prefetch bandwidth must be positive")

    #     # 2. Determine prefetch sets
    #     start_idx, end_idx = (self.read_buffer_set_start_id, self.read_buffer_set_end_id) \
    #         if not self.active_buffer_prefetch_done else \
    #         (self.prefetch_set_start_id, self.prefetch_set_end_id)

    #     # 3. Collect valid addresses
    #     valid_addresses = []
    #     for set_idx in range(start_idx, end_idx + 1):
    #         if 0 <= set_idx < len(self.list_of_sets) and self.list_of_sets[set_idx]:
    #             valid_addresses.extend(addr for addr in self.list_of_sets[set_idx] if addr != -1)

    #     # 4. Calculate actual prefetch parameters
    #     num_valid = len(valid_addresses)
    #     if num_valid == 0:
    #         return  # No valid prefetch addresses

    #     cycles_needed = max(1, math.ceil(num_valid / self.prefetch_bandwidth))
    #     actual_bandwidth = math.ceil(num_valid / cycles_needed)

    #     # 5. Generate prefetch requests
    #     prefetch_requests = np.full((cycles_needed, self.prefetch_bandwidth), -1, dtype=np.int32)
    #     for i in range(num_valid):
    #         cycle = i // actual_bandwidth
    #         pos = i % actual_bandwidth
    #         if cycle < cycles_needed and pos < self.prefetch_bandwidth:
    #             prefetch_requests[cycle, pos] = valid_addresses[i]

    #     # 6. Service requests and update traces
    #     response_cycles = np.full((cycles_needed, 1), self.last_prefetch_start_cycle, dtype=np.int32)
    #     returned_cycles = self.backing_buffer.service_reads(response_cycles, prefetch_requests)

    #     # 7. Update trace matrix with dimension checking
    #     new_traces = np.concatenate((returned_cycles, prefetch_requests), axis=1)
        
    #     if not self.trace_valid:
    #         self.trace_matrix = new_traces
    #         self.trace_valid = True
    #     else:
    #         # Align columns by padding with -1 if needed
    #         cols_self = self.trace_matrix.shape[1]
    #         cols_new = new_traces.shape[1]
            
    #         if cols_self < cols_new:
    #             padding = np.full((self.trace_matrix.shape[0], cols_new - cols_self), -1)
    #             self.trace_matrix = np.hstack((self.trace_matrix, padding))
    #         elif cols_new < cols_self:
    #             padding = np.full((new_traces.shape[0], cols_self - cols_new), -1)
    #             new_traces = np.hstack((new_traces, padding))
            
    #         self.trace_matrix = np.vstack((self.trace_matrix, new_traces))

    #     # 8. Update access count and prefetch state
    #     self.num_access += num_valid
    #     self.last_prefetch_end_cycle = self.last_prefetch_start_cycle + cycles_needed
    # def prefetch(self, start_cycle: int, num_sets: int) -> None:
    #     """Execute prefetch operations with cycle-accurate tracking.
        
    #     Args:
    #         start_cycle: The cycle when prefetching begins
    #         num_sets: Number of sets to prefetch
            
    #     Operations:
    #         1. Collects addresses from specified sets
    #         2. Generates properly formatted prefetch requests
    #         3. Services requests through backing buffer
    #         4. Updates trace matrix with prefetch operations
    #     """
    #     # 1. Validate configuration
    #     assert self.params_set_flag, "Parameters must be configured before prefetching"
    #     assert num_sets > 0, "Must prefetch at least one set"
        
    #     # 2. Determine prefetch range
    #     set_start = self.prefetch_set_start_id if self.active_buffer_prefetch_done else self.read_set_end_id
    #     set_end = set_start + num_sets
        
    #     # 3. Collect addresses from sets
    #     addresses: List[int] = []
    #     for set_id in range(set_start, set_end):
    #         if set_id < len(self.list_of_sets):
    #             addresses.extend(self.list_of_sets[set_id])
        
    #     # 4. Update access count
    #     self.num_access += len(addresses)
        
    #     # 5. Calculate prefetch parameters
    #     cycles_needed = (len(addresses) + self.prefetch_bandwidth - 1) // self.prefetch_bandwidth
    #     max_capacity = cycles_needed * self.prefetch_bandwidth
        
    #     # 6. Prepare prefetch requests
    #     if len(addresses) < max_capacity:
    #         addresses.extend([-1] * (max_capacity - len(addresses)))
        
    #     prefetch_requests = np.array(addresses, dtype=np.int32).reshape(
    #         cycles_needed, self.prefetch_bandwidth
    #     )
        
    #     # 7. Generate cycle timing
    #     cycles = np.arange(
    #         start_cycle,
    #         start_cycle + cycles_needed,
    #         dtype=np.int32
    #     ).reshape(-1, 1)
        
    #     # 8. Service requests through backing buffer
    #     response_cycles = self.backing_buffer.service_reads(prefetch_requests, cycles)
        
    #     # 9. Update trace matrix
    #     prefetch_trace = np.column_stack((response_cycles, prefetch_requests))
        
    #     if not self.trace_valid:
    #         self.trace_matrix = prefetch_trace
    #         self.trace_valid = True
    #     else:
    #         # Ensure dimensional compatibility
    #         if self.trace_matrix.shape[1] < prefetch_trace.shape[1]:
    #             # Pad existing trace with -1
    #             padding = -1 * np.ones((
    #                 self.trace_matrix.shape[0],
    #                 prefetch_trace.shape[1] - self.trace_matrix.shape[1]
    #             ), dtype=np.int32)
    #             self.trace_matrix = np.hstack((self.trace_matrix, padding))
    #         elif self.trace_matrix.shape[1] > prefetch_trace.shape[1]:
    #             # Pad new trace with -1
    #             padding = -1 * np.ones((
    #                 prefetch_trace.shape[0],
    #                 self.trace_matrix.shape[1] - prefetch_trace.shape[1]
    #             ), dtype=np.int32)
    #             prefetch_trace = np.hstack((prefetch_trace, padding))
            
    #         self.trace_matrix = np.vstack((self.trace_matrix, prefetch_trace))

        #






# 
# 
    # def get_latency(self):
    #     assert self.params_set_flag, 'Parameters are not valid'
    #     return self.hit_latency

        #
    # def get_trace_matrix(self):
    #     if not self.trace_valid:
    #         print('No trace has been generated yet')
    #         return

    #     return self.trace_matrix

        #
    # def get_hit_latency(self):
    #     return self.hit_latency

        #
    # def get_num_accesses(self):
    #     assert self.trace_valid, 'Traces not ready yet'
    #     return self.num_access

        #
    # def get_external_access_start_stop_cycles(self):
    #     assert self.trace_valid, 'Traces not ready yet'
    #     start_cycle = self.trace_matrix[0][0]
    #     end_cycle = self.trace_matrix[-1][0]

    #     return start_cycle, end_cycle

        #
    # def print_trace(self, filename):
    #     if not self.trace_valid:
    #         print('No trace has been generated yet')
    #         return

    #     np.savetxt(filename, self.trace_matrix, fmt='%s', delimiter=",")


    def get_latency(self) -> int:
        """Retrieve the configured hit latency of the buffer.
        
        Returns:
            int: The current hit latency in cycles
            
        Raises:
            AssertionError: If parameters haven't been configured
        """
        assert self.params_set_flag, "Parameters must be configured before accessing latency"
        return self.hit_latency

    def get_trace_matrix(self) -> Optional[np.ndarray]:
        """Retrieve the generated trace matrix if available.
        
        Returns:
            Optional[np.ndarray]: The trace matrix if valid, None otherwise
        """
        if not self.trace_valid:
            print("Trace not available: No valid trace generated yet")
            return None
        return self.trace_matrix.copy()  # Return copy to prevent modification

    def get_hit_latency(self) -> int:
        """Alias for get_latency() for interface compatibility.
        
        Returns:
            int: The current hit latency in cycles
        """
        return self.get_latency()

    def get_num_accesses(self) -> int:
        """Retrieve the total number of buffer accesses.
        
        Returns:
            int: The access count
            
        Raises:
            AssertionError: If no valid trace exists
        """
        assert self.trace_valid, "Cannot get access count: No valid trace exists"
        return self.num_access

    def get_external_access_start_stop_cycles(self) -> Tuple[int, int]:
        """Get the start and end cycles of external accesses.
        
        Returns:
            Tuple[int, int]: (start_cycle, end_cycle)
            
        Raises:
            AssertionError: If no valid trace exists
        """
        assert self.trace_valid, "Cannot get cycle information: No valid trace exists"
        return int(self.trace_matrix[0, 0]), int(self.trace_matrix[-1, 0])

    def print_trace(self, filename: str) -> None:
        """Save trace matrix to specified file in CSV format.
        
        Args:
            filename: Path to output file
            
        Operations:
            - Validates trace existence
            - Saves in CSV format with header
        """
        if not self.trace_valid:
            print("Trace not saved: No valid trace generated yet")
            return
            
        # Generate dynamic header based on trace shape
        num_columns = self.trace_matrix.shape[1]
        header = "Cycle," + ",".join(f"Address_{i}" for i in range(1, num_columns))
        
        np.savetxt(
            filename,
            self.trace_matrix,
            fmt='%d',
            delimiter=',',
            header=header,
            comments=''
        )

