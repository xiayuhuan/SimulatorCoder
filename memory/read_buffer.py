# Double buffer read memory implementation
# TODO: Verification Pending
import math
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from typing import Optional
from scalesim.memory.read_port import read_port


class read_buffer:
    # def __init__(self):
    #     # Buffer properties: User specified
    #     self.total_size_bytes = 128
    #     self.word_size = 1                      # Bytes
    #     self.active_buf_frac = 0.9
    #     self.hit_latency = 1                    # Cycles after which a request is served if already in the buffer

    #     # Buffer properties: Calculated
    #     self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
    #     self.active_buf_size = int(math.ceil(self.total_size_elems * 0.9))
    #     self.prefetch_buf_size = self.total_size_elems - self.active_buf_size

    #     # Backing interface properties
    #     self.backing_buffer = read_port()
    #     self.req_gen_bandwidth = 100            # words per cycle

    #     # Status of the buffer
    #     self.hashed_buffer = dict()
    #     self.num_lines = 0
    #     self.num_active_buf_lines = 1
    #     self.num_prefetch_buf_lines = 1
    #     self.active_buffer_set_limits = []
    #     self.prefetch_buffer_set_limits = []

    #     # Variables to enable prefetching
    #     self.fetch_matrix = np.ones((1, 1))
    #     self.last_prefect_cycle = -1
    #     self.next_line_prefetch_idx = 0
    #     self.next_col_prefetch_idx = 0

    #     # Access counts
    #     self.num_access = 0

    #     # Trace matrix
    #     self.trace_matrix = np.ones((1, 1))

    #     # Flags
    #     self.active_buf_full_flag = False
    #     self.hashed_buffer_valid = False
    #     self.trace_valid = False

    def __init__(self):
        """Initialize read buffer with default parameters and state."""
        # User-configurable parameters
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9
        self.hit_latency = 1
        
        # Calculate derived buffer properties
        self.total_size_elems = self.total_size_bytes // self.word_size
        self.active_buf_size = int(self.active_buf_frac * self.total_size_elems)
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size
        
        # Backing interface properties
        # from read_port import read_port  # Local import to prevent circular dependencies
        self.backing_buffer = read_port()
        self.req_gen_bandwidth = 100  # words per cycle
        
        # Buffer status tracking
        self.hashed_buffer: Dict[int, int] = {}  # key: hash, value: cycle
        self.num_lines = 0
        self.num_active_buf_lines = 0
        self.num_prefetch_buf_lines = 0
        self.active_buf_limits: List[int] = []
        self.prefetch_buf_limits: List[int] = []
        
        # Prefetching state
        self.fetch_matrix = np.zeros((1, 1), dtype=np.int32)  # Will be resized dynamically
        self.last_prefetch_cycle = -1
        self.next_line_prefetch_idx = 0
        self.next_col_prefetch_idx = 0
        
        # Access tracking
        self.num_access = 0
        self.trace_matrix = np.ones((1, 1), dtype=np.int32)  # Initialized with ones
        
        # State flags
        self.active_buf_full_flag = False
        self.hashed_buffer_valid = False
        self.trace_valid = False
    #
    # def set_params(self, backing_buf_obj,
    #                total_size_bytes=1, word_size=1, active_buf_frac=0.9,
    #                hit_latency=1, backing_buf_bw=1
    #                ):

    #     self.total_size_bytes = total_size_bytes
    #     self.word_size = word_size

    #     assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
    #     self.active_buf_frac = round(active_buf_frac, 2)
    #     self.hit_latency = hit_latency

    #     self.backing_buffer = backing_buf_obj
    #     self.req_gen_bandwidth = backing_buf_bw

    #     # Calculate these based on the values provided
    #     self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
    #     self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
    #     self.prefetch_buf_size = self.total_size_elems - self.active_buf_size
    def set_params(self, 
                backing_buf_obj,
                total_size_bytes: int = 1,
                word_size: int = 1,
                active_buf_frac: float = 0.9,
                hit_latency: int = 1,
                backing_buf_bw: int = 1) -> None:
        """Configure read buffer parameters and recalculate derived properties.
        
        Args:
            backing_buf_obj: Backing buffer interface object
            total_size_bytes: Total buffer capacity in bytes (must be positive)
            word_size: Size of each data word in bytes (must be positive)
            active_buf_frac: Fraction of buffer for active portion [0.5, 1)
            hit_latency: Cycles for buffer hit (must be positive)
            backing_buf_bw: Backing interface bandwidth in words/cycle (must be positive)
            
        Raises:
            ValueError: If any parameter constraints are violated
        """
        # Validate and set basic parameters
        if total_size_bytes <= 0:
            raise ValueError("Buffer size must be positive")
        self.total_size_bytes = total_size_bytes
        
        if word_size <= 0:
            raise ValueError("Word size must be positive")
        self.word_size = word_size
        
        # Validate and set active buffer fraction
        if not 0.5 <= active_buf_frac < 1.0:
            raise ValueError("Active buffer fraction must be in [0.5, 1)")
        self.active_buf_frac = round(active_buf_frac, 2)
        
        # Set timing parameters
        if hit_latency <= 0:
            raise ValueError("Hit latency must be positive")
        self.hit_latency = hit_latency
        
        # Configure backing interface
        if not hasattr(backing_buf_obj, 'service_reads'):
            raise ValueError("Backing buffer must implement service_reads()")
        self.backing_buffer = backing_buf_obj
        
        if backing_buf_bw <= 0:
            raise ValueError("Bandwidth must be positive")
        self.req_gen_bandwidth = backing_buf_bw
        
        # Recalculate derived properties
        self._recalculate_buffer_properties()

    def _recalculate_buffer_properties(self) -> None:
        """Internal method to update all derived buffer properties."""
        self.total_size_elems = self.total_size_bytes // self.word_size
        self.active_buf_size = int(self.active_buf_frac * self.total_size_elems)
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size
        
        # Reset buffer state tracking
        self.num_active_buf_lines = 0
        self.num_prefetch_buf_lines = 0
        self.active_buf_full_flag = False

    #
    # def reset(self): # TODO: check if all resets are working propoerly
    #     # Buffer properties: User specified
    #     self.total_size_bytes = 128
    #     self.word_size = 1  # Bytes
    #     self.active_buf_frac = 0.9
    #     self.hit_latency = 1  # Cycles after which a request is served if already in the buffer

    #     # Buffer properties: Calculated
    #     self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
    #     self.active_buf_size = int(math.ceil(self.total_size_elems * 0.9))
    #     self.prefetch_buf_size = self.total_size_elems - self.active_buf_size

    #     # Backing interface properties
    #     self.backing_buffer = read_port()
    #     self.req_gen_bandwidth = 100  # words per cycle

    #     # Status of the buffer
    #     self.hashed_buffer = dict()
    #     self.active_buffer_set_limits = []
    #     self.prefetch_buffer_set_limits = []

    #     # Variables to enable prefetching
    #     self.fetch_matrix = np.ones((1, 1))
    #     self.last_prefect_cycle = -1
    #     self.next_line_prefetch_idx = 0
    #     self.next_col_prefetch_idx = 0

    #     # Access counts
    #     self.num_access = 0

    #     # Trace matrix
    #     self.trace_matrix = np.ones((1, 1))

    #     # Flags
    #     self.active_buf_full_flag = False
    #     self.hashed_buffer_valid = False
    #     self.trace_valid = False

    def reset(self) -> None:
        """Reset buffer to initial state with exact 90-10% partitioning.
        
        Features:
        - Precise 90% active buffer allocation (rounded up)
        - Guaranteed active + prefetch = total elements
        - Complete state reinitialization
        """
        # 1. Restore default parameters
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9  # 90% active buffer
        self.hit_latency = 1

        # 2. Calculate buffer sizes with precise rounding (关键修改点)
        self.total_size_elems = self.total_size_bytes // self.word_size
        self.active_buf_size = int(math.ceil(self.total_size_elems * 0.9))  # 90% rounded up
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size  # Remaining 10%

        # 3. Reinitialize backing interface
        # from read_port import read_port
        self.backing_buffer = read_port()
        self.req_gen_bandwidth = 100

        # 4. Clear buffer state
        self.hashed_buffer: Dict[int, int] = {}
        self.active_buf_limits: List[int] = []
        self.prefetch_buf_limits: List[int] = []

        # 5. Reset prefetch system
        self.fetch_matrix = np.ones((1, 1), dtype=np.int32)
        self.last_prefetch_cycle = -1
        self.next_line_prefetch_idx = 0
        self.next_col_prefetch_idx = 0

        # 6. Reset counters
        self.num_access = 0
        self.num_lines = 0
        self.num_active_buf_lines = 0
        self.num_prefetch_buf_lines = 0

        # 7. Reset tracing
        self.trace_matrix = np.ones((1, 1), dtype=np.int32)
        self.active_buf_full_flag = False
        self.hashed_buffer_valid = False
        self.trace_valid = False

    #
    # def set_fetch_matrix(self, fetch_matrix_np):
    #     # The operand matrix determines what to pre-fetch into both active and prefetch buffers
    #     # In 'user' mode, this will be set in the set_params

    #     num_elems = fetch_matrix_np.shape[0] * fetch_matrix_np.shape[1]
    #     num_lines = int(math.ceil(num_elems / self.req_gen_bandwidth))
    #     self.fetch_matrix = np.ones((num_lines, self.req_gen_bandwidth)) * -1

    #     # Put stuff into the fetch matrix
    #     # This is done to ensure that there is no shape mismatch
    #     # Not sure if this is the optimal way to do it or not
    #     for i in range(num_elems):
    #         src_row = math.floor(i / fetch_matrix_np.shape[1])
    #         src_col = math.floor(i % fetch_matrix_np.shape[1])

    #         dest_row = math.floor(i / self.req_gen_bandwidth)
    #         dest_col = math.floor(i % self.req_gen_bandwidth)

    #         self.fetch_matrix[dest_row][dest_col] = fetch_matrix_np[src_row][src_col]

    #     # Once the fetch matrices are set, populate the data structure for fast lookups and servicing
    #     self.prepare_hashed_buffer()

    def set_fetch_matrix(self, fetch_matrix_np: np.ndarray) -> None:
        """Configure the prefetch operand matrix.
        
        Args:
            fetch_matrix_np: 2D NumPy array of operand IDs to prefetch
            
        Operations:
            1. Calculates required matrix dimensions
            2. Initializes fetch matrix with proper shape
            3. Performs shape-agnostic data mapping
            4. Prepares hashed buffer for fast lookups
        """
        # 1. Calculate matrix dimensions
        num_elems = fetch_matrix_np.size
        num_lines = (num_elems + self.req_gen_bandwidth - 1) // self.req_gen_bandwidth
        
        # 2. Initialize fetch matrix
        self.fetch_matrix = np.full(
            (num_lines, self.req_gen_bandwidth), 
            -1, 
            dtype=np.int32
        )
        
        # 3. Populate fetch matrix with shape-agnostic mapping
        src_rows, src_cols = fetch_matrix_np.shape
        for elem_idx in range(num_elems):
            # Calculate source position
            src_row = elem_idx // src_cols
            src_col = elem_idx % src_cols
            val = fetch_matrix_np[src_row, src_col]
            
            # Calculate destination position
            dst_row = elem_idx // self.req_gen_bandwidth
            dst_col = elem_idx % self.req_gen_bandwidth
            
            self.fetch_matrix[dst_row, dst_col] = val
        
        # 4. Prepare hashed buffer
        self.prepare_hashed_buffer()

    #
    # def prepare_hashed_buffer(self):
    #     elems_per_set = math.ceil(self.total_size_elems / 100)

    #     prefetch_rows = self.fetch_matrix.shape[0]
    #     prefetch_cols = self.fetch_matrix.shape[1]

    #     line_id = 0
    #     elem_ctr = 0
    #     current_line = set()

    #     for r in range(prefetch_rows):
    #         for c in range(prefetch_cols):
    #             elem = self.fetch_matrix[r][c]

    #             if not elem == -1:
    #                 current_line.add(elem)
    #                 elem_ctr += 1

    #             if not elem_ctr < elems_per_set:    # ie > or =
    #                 self.hashed_buffer[line_id] = current_line
    #                 line_id += 1
    #                 elem_ctr = 0
    #                 current_line = set()        # new set

    #     self.hashed_buffer[line_id] = current_line

    #     max_num_active_buf_lines = int(math.ceil(self.active_buf_size / elems_per_set))
    #     max_num_prefetch_buf_lines = int(math.ceil(self.prefetch_buf_size / elems_per_set))
    #     num_lines = line_id + 1

    #     if num_lines > max_num_active_buf_lines:
    #         self.num_active_buf_lines = max_num_active_buf_lines
    #     else:
    #         self.num_active_buf_lines = num_lines

    #     remaining_lines = num_lines - self.num_active_buf_lines

    #     if remaining_lines > max_num_prefetch_buf_lines:
    #         self.num_prefetch_buf_lines = max_num_prefetch_buf_lines
    #     else:
    #         self.num_prefetch_buf_lines = remaining_lines

    #     self.num_lines = num_lines
    #     self.hashed_buffer_valid = True

    def prepare_hashed_buffer(self) -> None:
        """Organizes prefetch elements into hashed sets and allocates buffer lines.
        
        Operations:
        1. Calculates element set size (ceil(total_size/100))
        2. Groups fetch matrix elements into hashed sets
        3. Computes active/prefetch buffer line allocation
        4. Updates buffer metadata and validity flags
        """
        # 1. Calculate elements per set (rounded up)
        elements_per_set = (self.total_size_elems + 99) // 100  # ceil(total/100)
        rows, cols = self.fetch_matrix.shape
        
        # 2. Initialize tracking variables
        self.hashed_buffer = {}
        current_line_id = 0
        element_count = 0
        current_set = set()

        # 3. Process fetch matrix and build hashed sets
        for i in range(rows):
            for j in range(cols):
                elem = self.fetch_matrix[i,j]
                if elem == -1:
                    continue
                    
                current_set.add(elem)
                element_count += 1
                
                # Store completed set
                if element_count >= elements_per_set:
                    self.hashed_buffer[current_line_id] = current_set
                    current_line_id += 1
                    element_count = 0
                    current_set = set()
        
        # 4. Store final partial set
        if current_set:
            self.hashed_buffer[current_line_id] = current_set
        
        # 5. Calculate maximum possible lines
        total_lines = len(self.hashed_buffer)
        max_active_lines = (self.active_buf_size + elements_per_set - 1) // elements_per_set
        max_prefetch_lines = (self.prefetch_buf_size + elements_per_set - 1) // elements_per_set

        # 6. Allocate active buffer lines (with bounds checking)
        self.num_active_buf_lines = min(total_lines, max_active_lines)
        
        # 7. Allocate prefetch buffer lines (with remaining lines and bounds checking)
        remaining_lines = total_lines - self.num_active_buf_lines
        self.num_prefetch_buf_lines = min(remaining_lines, max_prefetch_lines)

        # 8. Update buffer state
        self.num_lines = total_lines
        self.hashed_buffer_valid = True
        
        # Set buffer limits (exclusive upper bounds)
        self.active_buf_limits = list(range(self.num_active_buf_lines))
        self.prefetch_buf_limits = list(
            range(self.num_active_buf_lines, 
                self.num_active_buf_lines + self.num_prefetch_buf_lines)
        )

    #
    # def active_buffer_hit(self, addr):
    #     assert self.active_buf_full_flag, 'Active buffer is not ready yet'

    #     start_id, end_id = self.active_buffer_set_limits
    #     if start_id < end_id:
    #         for line_id in range(start_id, end_id):
    #             this_set = self.hashed_buffer[line_id]      # O(1) --> accessing hash
    #             if addr in this_set:                        # Checking in a set(), O(1) lookup
    #                 return True

    #     else:
    #         for line_id in range(start_id, self.num_lines):
    #             this_set = self.hashed_buffer[line_id]  # O(1) --> accessing hash
    #             if addr in this_set:  # Checking in a set(), O(1) lookup
    #                 return True

    #         for line_id in range(end_id):
    #             this_set = self.hashed_buffer[line_id]  # O(1) --> accessing hash
    #             if addr in this_set:  # Checking in a set(), O(1) lookup
    #                 return True
    #     # Fixing for ISSUE #14
    #     # return True
    #     return False

    def active_buffer_hit(self, addr: int) -> bool:
        """Check if an address exists in the active buffer.
        
        Args:
            addr: Memory address to check
            
        Returns:
            bool: True if address found in active buffer, False otherwise
            
        Raises:
            AssertionError: If active buffer not ready
        """
        # 1. Verify buffer readiness
        assert self.active_buf_full_flag, "Active buffer not ready for access"
        assert self.hashed_buffer_valid, "Hashed buffer not initialized"
        
        # 2. Get active buffer boundaries
        start_id = self.active_buf_limits[0] if self.active_buf_limits else 0
        end_id = self.active_buf_limits[-1] if self.active_buf_limits else 0
        
        # 3. Check normal case (non-wrapped buffer)
        if start_id <= end_id:
            for line_id in range(start_id, end_id + 1):
                if addr in self.hashed_buffer[line_id]:
                    return True
        else:
            # 4. Handle wrapped buffer case
            # Check from start_id to end of buffer
            for line_id in range(start_id, self.num_lines):
                if addr in self.hashed_buffer[line_id]:
                    return True
            # Check from start of buffer to end_id
            for line_id in range(0, end_id + 1):
                if addr in self.hashed_buffer[line_id]:
                    return True
        
        # 5. Address not found
        return False

    #
    # def service_reads(self, incoming_requests_arr_np,   # 2D array with the requests
    #                         incoming_cycles_arr):       # 1D vector with the cycles at which req arrived
    #     # Service the incoming read requests
    #     # returns a cycles array corresponding to the requests buffer
    #     # Logic: Always check if an addr is in active buffer.
    #     #        If hit, return with hit latency
    #     #        Else, make the contents of prefetch buffer as active and then check
    #     #              finish till an ongoing prefetch is done before reassiging prefetch buffer

    #     if not self.active_buf_full_flag:
    #         start_cycle = incoming_cycles_arr[0][0]
    #         self.prefetch_active_buffer(start_cycle=start_cycle)    # Needs to use the entire operand matrix
    #                                                                 # keeping in mind the tile order and everything

    #     out_cycles_arr = []
    #     offset = self.hit_latency
    #     # for cycle, request_line in tqdm(zip(incoming_cycles_arr, incoming_requests_arr_np)):
    #     for i in tqdm(range(incoming_requests_arr_np.shape[0]), disable=True):
    #         cycle = incoming_cycles_arr[i]
    #         # Fixing for ISSUE #14
    #         # request_line = set(incoming_requests_arr_np[i]) #shaves off a few seconds
    #         request_line = incoming_requests_arr_np[i]

    #         for addr in request_line:
    #             if addr == -1:
    #                 continue

    #             # if addr not in self.active_buffer_contents: #this is super slow!!!
    #             # Fixing for ISSUE #14
    #             # if not self.active_buffer_hit(addr):  # --> While loop ensures multiple prefetches if needed
    #             while not self.active_buffer_hit(addr):
    #                 self.new_prefetch()
    #                 potential_stall_cycles = self.last_prefect_cycle - (cycle + offset)
    #                 offset += potential_stall_cycles        # Offset increments if there were potential stalls

    #         out_cycles = cycle + offset
    #         out_cycles_arr.append(out_cycles)

    #     out_cycles_arr_np = np.asarray(out_cycles_arr).reshape((len(out_cycles_arr), 1))

    #     return out_cycles_arr_np

    def service_reads(self, 
                    incoming_requests_arr_np: np.ndarray,
                    incoming_cycles_arr: np.ndarray) -> np.ndarray:
        """Process read requests and calculate completion cycles.
        
        Args:
            incoming_requests_arr_np: 2D array of read requests (-1 for empty slots)
            incoming_cycles_arr: 1D array of arrival cycles
            
        Returns:
            np.ndarray: 2D array of completion cycles
            
        Operations:
            1. Prefetches active buffer if not ready
            2. Processes requests with hit/miss detection
            3. Calculates stall cycles for misses
            4. Returns completion cycles
        """
        # 1. Prefetch active buffer if empty
        if not self.active_buf_full_flag:
            first_cycle = incoming_cycles_arr[0] if len(incoming_cycles_arr) > 0 else 0
            self.prefetch_active_buf(first_cycle)
        
        # 2. Initialize tracking variables
        out_cycles_arr = []
        offset = self.hit_latency  # Default hit latency
        
        # 3. Process each request line
        for req_line, cycle in zip(incoming_requests_arr_np, incoming_cycles_arr):
            new_prefetch = False
            
            # 4. Check each address in request line
            for addr in req_line:
                if addr == -1:  # Skip empty slots
                    continue
                    
                # 5. Handle cache miss
                if not self.active_buffer_hit(addr):
                    if not new_prefetch:  # Only prefetch once per request line
                        self.prefetch_next_line(cycle + offset)
                        new_prefetch = True
                        offset += self.prefetch_latency  # Add prefetch penalty
            
            # 6. Calculate completion cycle
            completion_cycle = cycle + offset
            out_cycles_arr.append(completion_cycle)
            
            # 7. Reset offset for next request
            offset = self.hit_latency
        
        # 8. Format output
        return np.array(out_cycles_arr).reshape(-1, 1)

    #
    # def prefetch_active_buffer(self, start_cycle):
    #     # Depending on size of the active buffer, calculate the number of lines from op mat to fetch
    #     # Also, calculate the cycles arr for requests

    #     # 1. Preparing the requests:
    #     num_lines = math.ceil(self.active_buf_size / self.req_gen_bandwidth)
    #     if not num_lines < self.fetch_matrix.shape[0]:
    #         num_lines = self.fetch_matrix.shape[0]

    #     requested_data_size = num_lines * self.req_gen_bandwidth
    #     self.num_access += requested_data_size

    #     start_idx = 0
    #     end_idx = num_lines

    #     prefetch_requests = self.fetch_matrix[start_idx:end_idx, :]

    #     # 1.1 See if extra requests are made, if so nullify them
    #     self.next_col_prefetch_idx = 0
    #     if requested_data_size > self.active_buf_size:
    #         valid_cols = int(self.active_buf_size % self.req_gen_bandwidth)
    #         row = end_idx - 1
    #         self.next_col_prefetch_idx = valid_cols
    #         for col in range(valid_cols, self.req_gen_bandwidth):
    #             prefetch_requests[row][col] = -1

    #     # TODO: Tally and check if this agrees with the contents of the hashed buffer

    #     # 2. Preparing the cycles array
    #     #    The start_cycle variable ensures that all the requests have been made before any incoming reads came
    #     cycles_arr = np.zeros((num_lines, 1))
    #     for i in range(cycles_arr.shape[0]):
    #         cycles_arr[i][0] = -1 * (num_lines - start_cycle - (i - self.backing_buffer.get_latency()))

    #     # 3. Send the request and get the response cycles count
    #     response_cycles_arr = self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr,
    #                                                             incoming_requests_arr_np=prefetch_requests)

    #     # 4. Update the variables
    #     self.last_prefect_cycle = int(response_cycles_arr[-1][0])

    #     # Update the trace matrix
    #     self.trace_matrix = np.concatenate((response_cycles_arr, prefetch_requests), axis=1)
    #     self.trace_valid = True

    #     # Set active buffer contents
    #     active_buf_start_line_id = 0
    #     active_buf_end_line_id = self.num_active_buf_lines
    #     self.active_buffer_set_limits = [active_buf_start_line_id, active_buf_end_line_id]

    #     prefetch_buf_start_line_id = active_buf_end_line_id
    #     prefetch_buf_end_line_id = prefetch_buf_start_line_id + self.num_prefetch_buf_lines
    #     self.prefetch_buffer_set_limits = [prefetch_buf_start_line_id, prefetch_buf_end_line_id]

    #     self.active_buf_full_flag = True

    #     # Set the line to be prefetched next
    #     # The module operator is to ensure that the indices wrap around
    #     if requested_data_size > self.active_buf_size:  # Some elements in the current idx is left out in this case
    #         self.next_line_prefetch_idx = num_lines % self.fetch_matrix.shape[0]
    #     else:
    #         self.next_line_prefetch_idx = (num_lines + 1) % self.fetch_matrix.shape[0]

    def prefetch_active_buffer(self, start_cycle: int) -> None:
        """Prefetch data into active buffer and update all related states.
        
        Args:
            start_cycle: The cycle when prefetching begins
            
        Operations:
            1. Calculate line count respecting buffer capacity
            2. Prepare prefetch requests from fetch matrix
            3. Generate cycle-accurate request timing
            4. Service requests through backing buffer
            5. Update all buffer states and tracking variables
        """
        # 1. Calculate lines to fetch (bounded by active buffer capacity)
        lines_to_fetch = min(
            (self.active_buf_size + self.req_gen_bandwidth - 1) // self.req_gen_bandwidth,
            self.fetch_matrix.shape[0] - self.next_line_prefetch_idx
        )
        
        # 2. Prepare prefetch requests
        req_data_size = lines_to_fetch * self.req_gen_bandwidth
        self.num_access += req_data_size
        
        # Extract requests with bounds checking
        end_idx = self.next_line_prefetch_idx + lines_to_fetch
        prefetch_requests = self.fetch_matrix[self.next_line_prefetch_idx:end_idx, :]
        
        # Nullify excess requests in last row if needed
        if req_data_size > self.active_buf_size:
            excess = req_data_size - self.active_buf_size
            prefetch_requests[-1, -excess:] = -1
        
        # 3. Prepare cycles array
        cycles = np.arange(
            start_cycle,
            start_cycle + lines_to_fetch,
            dtype=np.int32
        )
        
        # 4. Service requests through backing buffer
        response_cycles = self.backing_buffer.service_reads(
            prefetch_requests,
            cycles
        )
        
        # 5. Update tracking variables
        self.last_prefetch_cycle = response_cycles[-1][0]
        
        # Update trace matrix
        if not self.trace_valid:
            self.trace_matrix = np.column_stack((response_cycles, prefetch_requests))
            self.trace_valid = True
        else:
            new_entries = np.column_stack((response_cycles, prefetch_requests))
            self.trace_matrix = np.vstack((self.trace_matrix, new_entries))
        
        # 6. Set buffer limits
        self.active_buf_limits = list(range(lines_to_fetch))
        self.prefetch_buf_limits = list(
            range(lines_to_fetch, 
                lines_to_fetch + self.num_prefetch_buf_lines)
        )
        
        # 7. Update prefetch state
        self.active_buf_full_flag = True
        self.next_line_prefetch_idx = (self.next_line_prefetch_idx + lines_to_fetch) % self.fetch_matrix.shape[0]
        
        # Adjust for partial lines if we wrapped around
        if self.next_line_prefetch_idx + self.req_gen_bandwidth > self.fetch_matrix.shape[0]:
            self.next_col_prefetch_idx = (self.next_line_prefetch_idx + self.req_gen_bandwidth) % self.fetch_matrix.shape[1]

        #
        # def new_prefetch(self):
        #     # In a new prefetch, some portion of the original data needs to be deleted to accomodate the prefetched data
        #     # In this case we overwrite some data in the active buffer with the prefetched data
        #     # And then create a new prefetch request
        #     # Also return when the prefetched data was made available

        #     # 1. Rewrite the active buffer
        #     assert self.active_buf_full_flag, 'Active buffer is empty'
        #     active_start, active_end = self.active_buffer_set_limits

        #     active_start = int((active_start + self.num_prefetch_buf_lines) % self.num_lines)
        #     active_end = int((active_start + self.num_active_buf_lines) % self.num_lines)
        #     prefetch_start = active_end
        #     prefetch_end = int((prefetch_start + self.num_prefetch_buf_lines) % self.num_lines)

        #     self.active_buffer_set_limits = [active_start, active_end]
        #     self.prefetch_buffer_set_limits = [prefetch_start, prefetch_end]

        #     # 2. Create the request
        #     start_idx = self.next_line_prefetch_idx
        #     num_lines = math.ceil(self.prefetch_buf_size / self.req_gen_bandwidth)
        #     end_idx = start_idx + num_lines
        #     requested_data_size = num_lines * self.req_gen_bandwidth
        #     self.num_access += requested_data_size

        #     # In case we need to circle back
        #     if end_idx > self.fetch_matrix.shape[0]:
        #         last_idx = self.fetch_matrix.shape[0]
        #         prefetch_requests = self.fetch_matrix[start_idx:,:]

        #         new_end_idx = min(end_idx - last_idx, start_idx)    # In case the entire array is engulfed
        #         prefetch_requests = np.concatenate((prefetch_requests, self.fetch_matrix[:new_end_idx,:]))
        #     else:
        #         prefetch_requests = self.fetch_matrix[start_idx:end_idx, :]

        #     # Modify the prefetch request to drop unwanted addresses
        #     # a. Chomp the elements in the first line included in previous fetches
        #     for i in range(0, self.next_col_prefetch_idx):
        #         prefetch_requests[0][i] = -1

        #     # b. Chomp the excess elements in the last line
        #     if requested_data_size > self.active_buf_size:
        #         valid_cols = int(self.active_buf_size % self.req_gen_bandwidth)
        #         row = prefetch_requests.shape[0] - 1
        #         for col in range(valid_cols, self.req_gen_bandwidth):
        #             prefetch_requests[row][col] = -1

        #     # 3. Create the request cycles
        #     cycles_arr = np.zeros((num_lines, 1))
        #     for i in range(cycles_arr.shape[0]):
        #         # Fixing ISSUE #14
        #         # cycles_arr[i][0] = self.last_prefect_cycle + i
        #         cycles_arr[i][0] = self.last_prefect_cycle + i + 1

        #     # 4. Send the request
        #     response_cycles_arr = self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr,
        #                                                             incoming_requests_arr_np=prefetch_requests)

        #     # 5. Update the variables
        #     self.last_prefect_cycle = response_cycles_arr[-1][0]

        #     assert response_cycles_arr.shape == cycles_arr.shape, 'The request and response cycles dims do not match'

        #     this_prefetch_trace = np.concatenate((response_cycles_arr, prefetch_requests), axis=1)
        #     self.trace_matrix = np.concatenate((self.trace_matrix, this_prefetch_trace), axis=0)

        #     # Set the line to be prefetched next
        #     if requested_data_size > self.active_buf_size:
        #         self.next_line_prefetch_idx = num_lines % self.fetch_matrix.shape[0]
        #     else:
        #         self.next_line_prefetch_idx = (num_lines + 1) % self.fetch_matrix.shape[1]          

    def new_prefetch(self, start_cycle: int) -> None:
        """Execute a new prefetch operation with cycle-accurate tracking.
        
        Args:
            start_cycle: The cycle when prefetching begins
            
        Operations:
            1. Reorganizes active/prefetch buffer boundaries
            2. Generates properly bounded prefetch requests
            3. Services requests through backing buffer
            4. Updates all tracking states and indices
        """
        # 1. Verify buffer state and initialize
        assert self.active_buf_full_flag, "Active buffer must be full for new prefetch"
        num_lines = (self.prefetch_buf_size + self.req_gen_bandwidth - 1) // self.req_gen_bandwidth
        
        # 2. Calculate request boundaries with wrap-around handling
        remaining_lines = self.fetch_matrix.shape[0] - self.next_line_prefetch_idx
        if remaining_lines >= num_lines:
            prefetch_requests = self.fetch_matrix[
                self.next_line_prefetch_idx:self.next_line_prefetch_idx + num_lines,
                :
            ]
        else:
            # Handle wrap-around by concatenating matrix segments
            first_segment = self.fetch_matrix[self.next_line_prefetch_idx:, :]
            second_segment = self.fetch_matrix[:num_lines - remaining_lines, :]
            prefetch_requests = np.vstack((first_segment, second_segment))
        
        # 3. Nullify unwanted addresses
        # First line: skip previously prefetched columns
        if self.next_col_prefetch_idx > 0:
            prefetch_requests[0, :self.next_col_prefetch_idx] = -1
        
        # Last line: nullify excess elements
        total_elems = num_lines * self.req_gen_bandwidth
        if total_elems > self.prefetch_buf_size:
            excess = total_elems - self.prefetch_buf_size
            prefetch_requests[-1, -excess:] = -1
        
        # 4. Generate cycle-accurate timing
        cycles = np.arange(
            start_cycle,
            start_cycle + num_lines,
            dtype=np.int32
        ).reshape(-1, 1)
        
        # 5. Service requests through backing buffer
        response_cycles = self.backing_buffer.service_reads(prefetch_requests, cycles)
        self.last_prefetch_cycle = response_cycles[-1][0]
        
        # 6. Update trace matrix
        new_entries = np.column_stack((response_cycles, prefetch_requests))
        if not self.trace_valid:
            self.trace_matrix = new_entries
            self.trace_valid = True
        else:
            self.trace_matrix = np.vstack((self.trace_matrix, new_entries))
        
        # 7. Update buffer limits (rotate buffers)
        self.active_buf_limits = list(
            range(self.num_active_buf_lines, 
                self.num_active_buf_lines + num_lines)
        )
        self.prefetch_buf_limits = list(
            range(self.num_active_buf_lines + num_lines,
                self.num_active_buf_lines + num_lines + self.num_prefetch_buf_lines)
        )
        
        # 8. Update prefetch indices
        self.next_line_prefetch_idx = (self.next_line_prefetch_idx + num_lines) % self.fetch_matrix.shape[0]
        self.next_col_prefetch_idx = (self.next_col_prefetch_idx + total_elems) % self.req_gen_bandwidth

            # This does not need to return anything

    #

    def get_trace_matrix(self):
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        return self.trace_matrix

    #
    def get_hit_latency(self):
        return self.hit_latency

    #
    def get_latency(self):
        return self.hit_latency

    #
    def get_num_accesses(self):
        assert self.trace_valid, 'Traces not ready yet'
        return self.num_access

    #
    def get_external_access_start_stop_cycles(self):
        assert self.trace_valid, 'Traces not ready yet'
        start_cycle = self.trace_matrix[0][0]
        end_cycle = self.trace_matrix[-1][0]

        return start_cycle, end_cycle

    #
    def print_trace(self, filename):
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        np.savetxt(filename, self.trace_matrix, fmt='%s', delimiter=",")
