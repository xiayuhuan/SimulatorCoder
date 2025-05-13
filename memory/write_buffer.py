# Buffer to stage the data to be written
# TODO: Verification Pending
import time
import math
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from scalesim.memory.write_port import write_port
from typing import Optional, Tuple
# from write_port import write_port  # Import locally to avoid circular dependencies

class write_buffer:
    # def __init__(self):
    #     # Buffer properties: User specified
    #     self.total_size_bytes = 128
    #     self.word_size = 1
    #     self.active_buf_frac = 0.9

    #     # Buffer properties: Calculated
    #     self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
    #     self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
    #     self.drain_buf_size = self.total_size_elems - self.active_buf_size

    #     # Backing interface properties
    #     self.backing_buffer = write_port()
    #     print("self.backing_buffer: ")
    #     print(self.backing_buffer)
    #     self.req_gen_bandwidth = 100

    #     # Status of the buffer
    #     self.free_space = self.total_size_elems
    #     self.drain_buf_start_line_id = 0
    #     self.drain_buf_end_line_id = 0

    #     # Helper data structures for faster execution
    #     self.line_idx = 0
    #     self.current_line = np.ones((1, 1)) * -1
    #     self.max_cache_lines = 2 ** 10              # TODO: This is arbitrary, check if this can be tuned
    #     self.trace_matrix_cache = np.zeros((1, 1))

    #     # Access counts
    #     self.num_access = 0

    #     # Trace matrix
    #     self.trace_matrix = np.zeros((1, 1))
    #     self.cycles_vec = np.zeros((1, 1))

    #     # Flags
    #     # This variable determines where the new requests should be buffered
    #     # 0: Directly in the drain buffer
    #     # 1: In the active buffer, while the drain buffer is flushed
    #     self.state = 0
    #     self.drain_end_cycle = 0

    #     self.trace_valid = False
    #     # Fixing ISSUE #10
    #     self.trace_matrix_cache_empty = True
    #     self.trace_matrix_empty = True

    # ds
    def __init__(self) -> None:
        """Initialize the write buffer with default configuration."""
        # User-specified buffer properties
        self.total_size_bytes = 128       # Total buffer size in bytes
        self.word_size = 1                # Size of each element in bytes
        self.active_buf_frac = 0.9        # Fraction of buffer for active portion

        # Calculated buffer properties
        self.total_size_elems = self.total_size_bytes // self.word_size
        self.active_buf_size = int(self.active_buf_frac * self.total_size_elems)
        self.drain_buf_size = self.total_size_elems - self.active_buf_size

        # Backing interface properties
        
        self.backing_buffer = write_port()
        # print(f"Initialized backing buffer: {self.backing_buffer}")
        self.req_gen_bandwidth = 100      # Request generation bandwidth

        # Buffer status tracking
        self.free_space = self.total_size_elems
        self.drain_buf_start_line_id = 0
        self.drain_buf_end_line_id = 0

        # Helper data structures
        self.line_idx = 0
        # self.current_line = np.full(self.word_size, -1, dtype=np.int32)
        self.current_line = np.full((1, self.word_size), -1, dtype=np.int32)
        self.max_cache_lines = 2**10  # 1024 lines
        self.trace_matrix_cache = np.zeros(
            (self.max_cache_lines, self.word_size), dtype=np.int32)

        # Access counting
        self.num_access = 0

        # Trace matrices
        self.trace_matrix = np.zeros(
            (self.max_cache_lines, self.word_size), dtype=np.int32)
        self.cycles_vec = np.zeros(self.max_cache_lines, dtype=np.int32)

        # State flags
        self.state = 0  # 0: buffer directly in drain buffer
        self.drain_end_cycle = 0
        self.trace_valid = False
        self.trace_matrix_cache_empty = True
        self.trace_matrix_empty = True

    # ICL151111358076error 
    # def __init__(self):
    #     """Initialize the write buffer with default settings."""
    #     # User-specified buffer properties
    #     self.total_size_bytes = 128
    #     self.word_size = 1  # bytes
    #     self.active_buf_frac = 0.9
        
    #     # Calculated buffer properties
    #     self.total_size_elems = self.total_size_bytes // self.word_size
    #     self.active_buf_size = int(self.total_size_elems * self.active_buf_frac)
    #     self.drain_buf_size = self.total_size_elems - self.active_buf_size
        
    #     # Backing interface properties
    #     self.backing_buffer = write_port()
    #     print(f"Backing buffer: {self.backing_buffer}")
    #     self.req_gen_bandwidth = 100
        
    #     # Buffer status
    #     self.free_space = self.total_size_elems
    #     self.drain_buf_start_line_id = 0
    #     self.drain_buf_end_line_id = 0
        
    #     # Helper data structures
    #     self.line_idx = 0
    #     self.current_line = np.full(self.total_size_elems, -1)
    #     self.max_cache_lines = 2**10  # 1024 lines
    #     self.trace_matrix_cache = np.zeros((self.max_cache_lines, self.total_size_elems))
        
    #     # Access counts
    #     self.num_access = 0
        
    #     # Trace matrix
    #     self.trace_matrix = np.zeros((self.max_cache_lines, self.total_size_elems))
    #     self.cycles_vec = np.zeros(self.max_cache_lines)
        
    #     # Flags
    #     self.state = 0  # 0: new reqs go to drain buffer directly
    #     self.drain_end_cycle = 0
    #     self.trace_valid = False
    #     self.trace_matrix_cache_empty = True
    #     self.trace_matrix_empty = True
    #

    def __init__(self):
        """Initialize the write buffer with default settings."""
        # User-specified buffer properties
        self.total_size_bytes = 128
        self.word_size = 1  # bytes
        self.active_buf_frac = 0.9
        
        # Calculated buffer properties
        self.total_size_elems = self.total_size_bytes // self.word_size
        self.active_buf_size = int(self.total_size_elems * self.active_buf_frac)
        self.drain_buf_size = self.total_size_elems - self.active_buf_size
        
        # Backing interface properties
        self.backing_buffer = write_port()
        # print(f"Backing buffer: {self.backing_buffer}")
        self.req_gen_bandwidth = 100
        
        # Buffer status
        self.free_space = self.total_size_elems
        self.drain_buf_start_line_id = 0
        self.drain_buf_end_line_id = 0
        
        # Helper data structures
        self.line_idx = 0
        # Changed to 2D array with shape (1, total_size_elems) to match usage
        self.current_line = np.full((1, self.total_size_elems), -1)
        self.max_cache_lines = 2**10  # 1024 lines
        # self.trace_matrix_cache = np.zeros((self.max_cache_lines, self.total_size_elems))
        self.trace_matrix_cache = np.full((0, self.req_gen_bandwidth), -1, dtype=np.int32)
        
        # Access counts
        self.num_access = 0
        
        # Trace matrix
        self.trace_matrix = np.zeros((self.max_cache_lines, self.total_size_elems))
        self.cycles_vec = np.zeros(self.max_cache_lines)
        
        # Flags
        self.state = 0  # 0: new reqs go to drain buffer directly
        self.drain_end_cycle = 0
        self.trace_valid = False
        self.trace_matrix_cache_empty = True
        self.trace_matrix_empty = True
    
    # ICL 1550 error
    # def __init__(self):
    #     """Initialize write buffer with default configuration."""
    #     # 1. User-specified buffer properties
    #     self.total_size_bytes = 128       # Total buffer size in bytes
    #     self.word_size = 1                # Size of each element in bytes
    #     self.active_buf_frac = 0.9        # Fraction of buffer for active portion

    #     # 2. Calculated buffer properties
    #     self.total_size_elems = self.total_size_bytes // self.word_size
    #     self.active_buf_size = int(self.active_buf_frac * self.total_size_elems)
    #     self.drain_buf_size = self.total_size_elems - self.active_buf_size

    #     # 3. Backing interface properties
    #     # from write_port import write_port  # Local import to prevent circular dependencies
    #     self.backing_buffer = write_port()
    #     print(f"Initialized backing buffer: {self.backing_buffer}")
    #     self.req_gen_bandwidth = 100      # Request generation bandwidth

    #     # 4. Buffer status
    #     self.free_space = self.total_size_elems
    #     self.drain_buf_start_line_id = 0
    #     self.drain_buf_end_line_id = 0

    #     # 5. Helper data structures
    #     self.line_idx = 0
    #     self.current_line = np.full((1, self.req_gen_bandwidth), -1, dtype=np.int32)
    #     self.max_cache_lines = 2**10      # 1024 lines
    #     self.trace_matrix_cache = np.zeros(
    #         (self.max_cache_lines, self.req_gen_bandwidth), dtype=np.int32)

    #     # 6. Access counts
    #     self.num_access = 0

    #     # 7. Trace matrices
    #     self.trace_matrix = np.zeros(
    #         (self.max_cache_lines, self.req_gen_bandwidth), dtype=np.int32)
    #     self.cycles_vec = np.zeros(self.max_cache_lines, dtype=np.int32)

    #     # 8. Flags
    #     self.state = 0                    # 0: buffer directly in drain buffer
    #     self.drain_end_cycle = 0
    #     self.trace_valid = False
    #     self.trace_matrix_cache_empty = True
    #     self.trace_matrix_empty = True

    def set_params(self, backing_buf_obj,
                   total_size_bytes=128, word_size=1, active_buf_frac=0.9,
                   backing_buf_bw=100
                   ):
        self.total_size_bytes = total_size_bytes
        self.word_size = word_size

        assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
        self.active_buf_frac = active_buf_frac

        self.backing_buffer = backing_buf_obj
        self.req_gen_bandwidth = backing_buf_bw

        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.drain_buf_size = self.total_size_elems - self.active_buf_size
        self.free_space = self.total_size_elems

    #
    # def reset(self):
    #     self.total_size_bytes = 128
    #     self.word_size = 1
    #     self.active_buf_frac = 0.9

    #     self.backing_buffer = write_buffer()
    #     self.req_gen_bandwidth = 100

    #     self.free_space = self.total_size_elems
    #     self.active_buf_contents = []
    #     self.drain_buf_contents = []
    #     self.drain_end_cycle = 0

    #     self.trace_matrix = np.zeros((1, 1))

    #     self.num_access = 0
    #     self.state = 0

    #     self.trace_valid = False
    #     # Fixing ISSUE #10
    #     self.trace_matrix_cache_empty = True
    #     self.trace_matrix_empty = True
    def _recalculate_buffer_properties(self):
        """Recalculate derived buffer properties."""
        self.total_size_elems = self.total_size_bytes // self.word_size
        self.active_buf_size = int(self.active_buf_frac * self.total_size_elems)
        self.drain_buf_size = self.total_size_elems - self.active_buf_size
    def reset(self) -> None:
        """Reset the write buffer to its initial default state.
        
        This method:
        - Restores all configurable parameters to defaults
        - Clears all data structures and state tracking
        - Reinitializes the backing buffer interface
        - Resets all tracking variables and flags
        """
        # Restore default buffer properties
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9
        
        # Recalculate derived buffer properties
        self._recalculate_buffer_properties()
        
        # Reinitialize backing buffer interface
        # from write_port import write_port  # Local import to prevent circular dependencies
        self.backing_buffer = write_port()
        self.req_gen_bandwidth = 100
        
        # Reset buffer status
        self.free_space = self.total_size_elems
        self.drain_buf_start_line_id = 0
        self.drain_buf_end_line_id = 0
        self.drain_end_cycle = 0
        
        # Reset data structures
        self.line_idx = 0
        self.current_line = np.full((1, self.req_gen_bandwidth), -1, dtype=np.int32)
        self.trace_matrix_cache = np.zeros((0, self.req_gen_bandwidth), dtype=np.int32)
        
        # Reset access tracking
        self.num_access = 0
        self.trace_matrix = np.zeros((0, self.req_gen_bandwidth), dtype=np.int32)
        self.cycles_vec = np.zeros(0, dtype=np.int32)
        
        # Reset state flags
        self.state = 0  # Default to buffering directly in drain buffer
        self.trace_valid = False
        self.trace_matrix_cache_empty = True
        self.trace_matrix_empty = True

    #
    # def store_to_trace_mat_cache(self, elem):
    #     if elem == -1:
    #         return

    #     if self.current_line.shape == (1,1):    # This line is empty
    #         self.current_line = np.ones((1, self.req_gen_bandwidth)) * -1

    #     self.current_line[0, self.line_idx] = elem
    #     self.line_idx += 1
    #     self.free_space -= 1

    #     if not self.line_idx < self.req_gen_bandwidth:
    #         # Store to the cache matrix
    #         # Fixing ISSUE #10
    #         # if self.trace_matrix_cache.shape == (1,1):
    #         if self.trace_matrix_cache_empty:
    #             self.trace_matrix_cache = self.current_line
    #             self.trace_matrix_cache_empty = False
    #         else:
    #             self.trace_matrix_cache = np.concatenate((self.trace_matrix_cache, self.current_line), axis=0)

    #         self.current_line = np.ones((1,1)) * -1
    #         self.line_idx = 0

    #         if not self.trace_matrix_cache.shape[0] < self.max_cache_lines:
    #             self.append_to_trace_mat()

    # def store_to_trace_mat_cache(self, elem: int):
    #     """Store element in trace matrix cache.
        
    #     Args:
    #         elem: Element to store (-1 indicates empty)
    #     """
    #     if elem == -1:
    #         return
            
    #     if self.current_line.shape == (1,1):
    #         self.current_line = np.full((1, self.req_gen_bandwidth), -1, dtype=np.int32)
            
    #     self.current_line[0, self.line_idx] = elem
    #     self.line_idx += 1
    #     self.free_space -= 1
        
    #     if self.line_idx == self.req_gen_bandwidth:
    #         if self.trace_matrix_cache_empty:
    #             self.trace_matrix_cache = self.current_line
    #             self.trace_matrix_cache_empty = False
    #         else:
    #             self.trace_matrix_cache = np.vstack((self.trace_matrix_cache, self.current_line))
            
    #         self.current_line = np.full((1, self.req_gen_bandwidth), -1, dtype=np.int32)
    #         self.line_idx = 0
            
    #         if len(self.trace_matrix_cache) == self.max_cache_lines:
    #             self.append_to_trace_mat()


    def store_to_trace_mat_cache(self, elem: int):
        """Store element in trace matrix cache.
        
        Args:
            elem: Element to store (-1 indicates empty)
        """
        if elem == -1:
            return
            
        # Initialize current_line if empty or wrong size
        if (not hasattr(self, 'current_line')) or \
        self.current_line.shape != (1, self.req_gen_bandwidth):
            self.current_line = np.full((1, self.req_gen_bandwidth), -1, dtype=np.int32)
            self.line_idx = 0
            
        # Store element
        self.current_line[0, self.line_idx] = elem
        self.line_idx += 1
        self.free_space -= 1
        
        # Check if current line is full
        if self.line_idx == self.req_gen_bandwidth:
            # Initialize trace_matrix_cache if empty
            if self.trace_matrix_cache_empty:
                self.trace_matrix_cache = np.full((0, self.req_gen_bandwidth), -1, dtype=np.int32)
                self.trace_matrix_cache_empty = False
                
            # Ensure dimensions match before vstack
            if self.trace_matrix_cache.shape[1] == self.current_line.shape[1]:
                self.trace_matrix_cache = np.vstack((self.trace_matrix_cache, self.current_line))
            else:
                # Handle dimension mismatch by resizing
                resized_line = np.full((1, self.trace_matrix_cache.shape[1]), -1)
                copy_len = min(self.current_line.shape[1], resized_line.shape[1])
                resized_line[0, :copy_len] = self.current_line[0, :copy_len]
                self.trace_matrix_cache = np.vstack((self.trace_matrix_cache, resized_line))
            
            # Reset current line
            self.current_line = np.full((1, self.req_gen_bandwidth), -1, dtype=np.int32)
            self.line_idx = 0
            
            # Check if cache is full
            if len(self.trace_matrix_cache) == self.max_cache_lines:
                self.append_to_trace_mat()

    #
    # def append_to_trace_mat(self, force=False):
    #     if force:   # This forces the contents for self.current_line and self.trace_matrix cache to be dumped
    #         if not self.line_idx == 0:
    #             #if self.trace_matrix_cache.shape == (1,1):
    #             if self.trace_matrix_cache_empty:
    #                 self.trace_matrix_cache = self.current_line
    #                 self.trace_matrix_cache_empty = False
    #             else:
    #                 self.trace_matrix_cache = np.concatenate((self.trace_matrix_cache, self.current_line), axis=0)

    #             self.current_line = np.ones((1,1)) * -1
    #             self.line_idx = 0
    #     # Fixing ISSUE #10
    #     # if self.trace_matrix_cache.shape == (1,1):
    #     if self.trace_matrix_cache_empty:
    #         return

    #     #if self.trace_matrix.shape == (1,1):
    #     if self.trace_matrix_empty:
    #         self.trace_matrix = self.trace_matrix_cache
    #         self.drain_buf_start_line_id = 0
    #         self.trace_matrix_empty = False
    #     else:
    #         self.trace_matrix = np.concatenate((self.trace_matrix, self.trace_matrix_cache), axis=0)

    #     self.trace_matrix_cache = np.zeros((1,1))
    #     # Fixing ISSUE #10
    #     self.trace_matrix_cache_empty = True
    
    # 用yolo_tiny有问题
    # def append_to_trace_mat(self, force: bool = False):
    #     """Append cache to main trace matrix.
        
    #     Args:
    #         force: Whether to force append current line
    #     """
    #     if force and self.line_idx != 0:
    #         if self.trace_matrix_cache_empty:
    #             self.trace_matrix_cache = self.current_line[:1, :self.line_idx]
    #             self.trace_matrix_cache_empty = False
    #         else:
    #             self.trace_matrix_cache = np.vstack((
    #                 self.trace_matrix_cache, 
    #                 self.current_line[:1, :self.line_idx]
    #             ))
    #         self.current_line = np.full((1, self.req_gen_bandwidth), -1, dtype=np.int32)
    #         self.line_idx = 0
            
    #     if self.trace_matrix_cache_empty:
    #         return
            
    #     if self.trace_matrix_empty:
    #         self.trace_matrix = self.trace_matrix_cache
    #         self.trace_matrix_empty = False
    #     else:
    #         self.trace_matrix = np.vstack((self.trace_matrix, self.trace_matrix_cache))
            
    #     self.trace_matrix_cache = np.zeros((0, self.req_gen_bandwidth), dtype=np.int32)
    #     self.trace_matrix_cache_empty = True
    def append_to_trace_mat(self, force: bool = False) -> None:
        """Append cached trace data to main trace matrix.
        
        Args:
            force: If True, forces append of partially filled current line
            
        Operation:
            - Handles partial line appending when forced
            - Manages cache-to-main matrix transfer
            - Maintains proper array dimensions and state flags
        """
        # Handle forced append of partial line
        if force and self.line_idx != 0:
            # Create properly sized slice of current line
            partial_line = self.current_line[:, :self.line_idx]
            
            if self.trace_matrix_cache_empty:
                self.trace_matrix_cache = partial_line
                self.trace_matrix_cache_empty = False
            else:
                # Ensure dimensional compatibility before concatenation
                if partial_line.shape[1] != self.trace_matrix_cache.shape[1]:
                    # Pad partial line to match cache width if needed
                    padded_line = np.full((1, self.req_gen_bandwidth), -1)
                    padded_line[:, :self.line_idx] = partial_line
                    self.trace_matrix_cache = np.vstack((self.trace_matrix_cache, padded_line))
                else:
                    self.trace_matrix_cache = np.vstack((self.trace_matrix_cache, partial_line))
            
            # Reset current line state
            self.current_line.fill(-1)
            self.line_idx = 0

        # Early return if cache is empty
        if self.trace_matrix_cache_empty:
            return

        # Transfer cache to main matrix
        if self.trace_matrix_empty:
            self.trace_matrix = self.trace_matrix_cache
            self.trace_matrix_empty = False
        else:
            # Verify dimensional compatibility
            if self.trace_matrix_cache.shape[1] != self.trace_matrix.shape[1]:
                # Handle width mismatch by padding shorter rows with -1
                max_width = max(self.trace_matrix_cache.shape[1], self.trace_matrix.shape[1])
                padded_cache = np.full((self.trace_matrix_cache.shape[0], max_width), -1)
                padded_cache[:, :self.trace_matrix_cache.shape[1]] = self.trace_matrix_cache
                
                padded_main = np.full((self.trace_matrix.shape[0], max_width), -1)
                padded_main[:, :self.trace_matrix.shape[1]] = self.trace_matrix
                
                self.trace_matrix = np.vstack((padded_main, padded_cache))
            else:
                self.trace_matrix = np.vstack((self.trace_matrix, self.trace_matrix_cache))

        # Reset cache
        self.trace_matrix_cache = np.zeros((0, self.req_gen_bandwidth), dtype=np.int32)
        self.trace_matrix_cache_empty = True

    #
    # def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
    #     assert incoming_cycles_arr_np.shape[0] == incoming_requests_arr_np.shape[0], 'Cycles and requests do not match'
    #     out_cycles_arr = []
    #     offset = 0

    #     DEBUG_num_drains = 0
    #     DEBUG_append_to_trace_times = []

    #     for i in tqdm(range(incoming_requests_arr_np.shape[0]), disable=True):
    #         row = incoming_requests_arr_np[i]
    #         cycle = incoming_cycles_arr_np[i]
    #         current_cycle = cycle[0] + offset

    #         for elem in row:
    #             # Pay no attention to empty requests
    #             if elem == -1:
    #                 continue

    #             self.store_to_trace_mat_cache(elem)

    #             if current_cycle < self.drain_end_cycle:
    #                 if not self.free_space > 0:
    #                     offset += max(self.drain_end_cycle - current_cycle, 0)
    #                     current_cycle = self.drain_end_cycle

    #             elif self.free_space < (self.total_size_elems - self.drain_buf_size):
    #                 self.append_to_trace_mat(force=True)
    #                 self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=current_cycle)

    #         out_cycles_arr.append(current_cycle)

    #     num_lines = incoming_requests_arr_np.shape[0]
    #     out_cycles_arr_np = np.asarray(out_cycles_arr).reshape((num_lines, 1))

    #     #print('DEBUG: Num Drains = ' + str(DEBUG_num_drains))
    #     #print('DEBUG: Num appeneds = ' + str(len(DEBUG_append_to_trace_times)))
    #     #plt.plot(DEBUG_append_to_trace_times)
    #     #plt.show()

    #     return out_cycles_arr_np
    def service_writes(self, incoming_requests_arr_np: np.ndarray, 
                      incoming_cycles_arr_np: np.ndarray) -> np.ndarray:
        """Process incoming write requests.
        
        Args:
            incoming_requests_arr_np: Array of write requests
            incoming_cycles_arr_np: Array of arrival cycles
            
        Returns:
            Array of completion cycles
        """
        assert len(incoming_requests_arr_np) == len(incoming_cycles_arr_np), "Input size mismatch"
        
        out_cycles_arr = []
        offset = 0
        
        for req_row, cycle in zip(incoming_requests_arr_np, incoming_cycles_arr_np):
            adjusted_cycle = cycle + offset
            
            for elem in req_row:
                if elem == -1:
                    continue
                    
                if adjusted_cycle < self.drain_end_cycle and self.free_space == 0:
                    offset = self.drain_end_cycle - cycle
                    adjusted_cycle = self.drain_end_cycle
                    
                if self.free_space == 0:
                    self.append_to_trace_mat(force=True)
                    self.empty_drain_buf(adjusted_cycle)
                    offset = self.drain_end_cycle - cycle
                    adjusted_cycle = self.drain_end_cycle
                    
                self.store_to_trace_mat_cache(elem)
                
            out_cycles_arr.append(adjusted_cycle)
            
        return np.array(out_cycles_arr).reshape(-1, 1)

    #
    # def empty_drain_buf(self, empty_start_cycle=0):

    #     lines_to_fill_dbuf = int(math.ceil(self.drain_buf_size / self.req_gen_bandwidth))
    #     self.drain_buf_end_line_id = self.drain_buf_start_line_id + lines_to_fill_dbuf
    #     self.drain_buf_end_line_id = min(self.drain_buf_end_line_id, self.trace_matrix.shape[0])

    #     requests_arr_np = self.trace_matrix[self.drain_buf_start_line_id: self.drain_buf_end_line_id, :]
    #     num_lines = requests_arr_np.shape[0]

    #     data_sz_to_drain = num_lines * requests_arr_np.shape[1]
    #     # Adjust for -1
    #     for elem in requests_arr_np[-1,:]:
    #         if elem == -1:
    #             data_sz_to_drain -= 1
    #     self.num_access += data_sz_to_drain

    #     cycles_arr = [x+empty_start_cycle for x in range(num_lines)]
    #     cycles_arr_np = np.asarray(cycles_arr).reshape((num_lines, 1))
    #     serviced_cycles_arr = self.backing_buffer.service_writes(requests_arr_np, cycles_arr_np)

    #     # Assign the cycles vector which will be used to generate the complete trace
    #     if not self.trace_valid:
    #         self.cycles_vec = serviced_cycles_arr
    #         self.trace_valid = True
    #     else:
    #         self.cycles_vec = np.concatenate((self.cycles_vec, serviced_cycles_arr), axis=0)

    #     service_end_cycle = serviced_cycles_arr[-1][0]
    #     self.free_space += data_sz_to_drain

    #     self.drain_buf_start_line_id = self.drain_buf_end_line_id
    #     return service_end_cycle

    # deeepseek1 error
    # def empty_drain_buf(self, empty_start_cycle: int = 0) -> int:
    #     """Empty drain buffer to backing storage.
        
    #     Args:
    #         empty_start_cycle: Cycle to start draining
            
    #     Returns:
    #         Cycle when draining completes
    #     """
    #     num_lines = (self.drain_buf_size + self.req_gen_bandwidth - 1) // self.req_gen_bandwidth
    #     self.drain_buf_end_line_id = min(
    #         self.drain_buf_start_line_id + num_lines,
    #         len(self.trace_matrix)
    #     )
        
    #     if self.drain_buf_start_line_id >= self.drain_buf_end_line_id:
    #         return empty_start_cycle
            
    #     requests = self.trace_matrix[
    #         self.drain_buf_start_line_id:self.drain_buf_end_line_id
    #     ]
        
    #     data_size = np.sum(requests != -1)
    #     self.num_access += data_size
        
    #     cycles = np.arange(
    #         empty_start_cycle,
    #         empty_start_cycle + (self.drain_buf_end_line_id - self.drain_buf_start_line_id)
    #     )
        
    #     service_cycles = self.backing_buffer.service_writes(requests, cycles)
        
    #     if not self.trace_valid:
    #         self.cycles_vec = service_cycles.flatten()
    #         self.trace_valid = True
    #     else:
    #         self.cycles_vec = np.concatenate((
    #             self.cycles_vec, 
    #             service_cycles.flatten()
    #         ))
            
    #     self.free_space += data_size
    #     self.drain_buf_start_line_id = self.drain_buf_end_line_id
    #     self.drain_end_cycle = empty_start_cycle + len(service_cycles)
        
    #     return self.drain_end_cycle

    # deepseek2
    def empty_drain_buf(self, empty_start_cycle: int = 0) -> int:
        """Empty drain buffer to backing storage.
        
        Args:
            empty_start_cycle: Cycle to start draining
            
        Returns:
            Cycle when draining completes
        """
        num_lines = (self.drain_buf_size + self.req_gen_bandwidth - 1) // self.req_gen_bandwidth
        self.drain_buf_end_line_id = min(
            self.drain_buf_start_line_id + num_lines,
            len(self.trace_matrix)
        )
        
        if self.drain_buf_start_line_id >= self.drain_buf_end_line_id:
            return empty_start_cycle
            
        requests = self.trace_matrix[
            self.drain_buf_start_line_id:self.drain_buf_end_line_id
        ]
        
        data_size = np.sum(requests != -1)
        self.num_access += data_size
        
        cycles = np.arange(
            empty_start_cycle,
            empty_start_cycle + (self.drain_buf_end_line_id - self.drain_buf_start_line_id)
        ).reshape(-1, 1)  # Ensure cycles is 2D column vector
        
        service_cycles = self.backing_buffer.service_writes(requests, cycles)
        
        if not self.trace_valid:
            # Initialize as 2D array with single column
            self.cycles_vec = service_cycles
            self.trace_valid = True
        else:
            # Vertically stack to maintain 2D structure
            self.cycles_vec = np.vstack((self.cycles_vec, service_cycles))
            
        self.free_space += data_size
        self.drain_buf_start_line_id = self.drain_buf_end_line_id
        self.drain_end_cycle = empty_start_cycle + len(service_cycles)
        
        return self.drain_end_cycle

    # ICL error
    # def empty_drain_buf(self, empty_start_cycle: int = 0) -> int:
    #     """Empty the drain buffer by writing contents to backing buffer.
        
    #     Args:
    #         empty_start_cycle: Cycle when draining begins (default 0)
        
    #     Returns:
    #         Cycle when draining completes
    #     """
    #     # Calculate lines needed to fill drain buffer
    #     lines_to_fill = int(np.ceil(self.drain_buf_size / self.req_gen_bandwidth))
        
    #     # Set drain buffer end line ID
    #     self.drain_buf_end_line_id = min(
    #         self.drain_buf_start_line_id + lines_to_fill,
    #         self.max_cache_lines
    #     )
        
    #     # Extract requests from trace matrix
    #     requests = self.trace_matrix[
    #         self.drain_buf_start_line_id:self.drain_buf_end_line_id
    #     ]
        
    #     # Calculate actual data size to drain (excluding empty slots)
    #     data_size = np.sum(requests != -1)
        
    #     # Update access count
    #     self.num_access += data_size
        
    #     # Generate cycles array for draining
    #     cycles = np.arange(
    #         empty_start_cycle,
    #         empty_start_cycle + (self.drain_buf_end_line_id - self.drain_buf_start_line_id)
    #     )
        
    #     # Service writes through backing buffer
    #     service_cycles = self.backing_buffer.service_writes(
    #         requests=requests,
    #         cycles=cycles
    #     )
        
    #     # Update trace matrix with completion cycles
    #     if not self.trace_valid:
    #         self.trace_matrix[:len(service_cycles)] = requests
    #         self.cycles_vec[:len(service_cycles)] = service_cycles
    #         self.trace_valid = True
    #         self.trace_matrix_empty = False
    #     else:
    #         start_idx = self.drain_buf_start_line_id
    #         end_idx = self.drain_buf_end_line_id
    #         self.trace_matrix[start_idx:end_idx] = requests
    #         self.cycles_vec[start_idx:end_idx] = service_cycles
        
    #     # Update buffer status
    #     self.free_space += data_size
    #     self.drain_buf_start_line_id = self.drain_buf_end_line_id
        
    #     # Return final service cycle
    #     return empty_start_cycle + len(service_cycles)

    # ICL error same
    # def empty_drain_buf(self, empty_start_cycle: int = 0) -> int:
    #     """Empty the drain buffer by writing contents to backing buffer.
        
    #     Args:
    #         empty_start_cycle: Cycle when draining begins (default 0)
        
    #     Returns:
    #         Cycle when draining completes
    #     """
    #     # Calculate lines needed to fill drain buffer
    #     lines_to_fill = int(np.ceil(self.drain_buf_size / self.req_gen_bandwidth))
        
    #     # Set drain buffer end line ID
    #     self.drain_buf_end_line_id = min(
    #         self.drain_buf_start_line_id + lines_to_fill,
    #         self.max_cache_lines
    #     )
        
    #     # Early return if no lines to process
    #     if self.drain_buf_start_line_id >= self.drain_buf_end_line_id:
    #         return empty_start_cycle
        
    #     # Extract requests from trace matrix
    #     requests = self.trace_matrix[
    #         self.drain_buf_start_line_id:self.drain_buf_end_line_id
    #     ]
        
    #     # Calculate actual data size to drain (excluding empty slots)
    #     data_size = np.sum(requests != -1)
        
    #     # Update access count
    #     self.num_access += data_size
        
    #     # Generate cycles array for draining
    #     num_lines = self.drain_buf_end_line_id - self.drain_buf_start_line_id
    #     cycles = np.arange(
    #         empty_start_cycle,
    #         empty_start_cycle + num_lines
    #     )
        
    #     # Service writes through backing buffer
    #     service_cycles = self.backing_buffer.service_writes(
    #         incoming_requests_arr_np=requests,
    #         incoming_cycles_arr_np=cycles
    #     ).flatten()  # Convert 2D result to 1D
        
    #     # Update trace matrix with completion cycles
    #     if not self.trace_valid:
    #         self.trace_matrix[:num_lines] = requests
    #         self.cycles_vec[:num_lines] = service_cycles
    #         self.trace_valid = True
    #         self.trace_matrix_empty = False
    #     else:
    #         start_idx = self.drain_buf_start_line_id
    #         end_idx = self.drain_buf_end_line_id
    #         self.trace_matrix[start_idx:end_idx] = requests
    #         self.cycles_vec[start_idx:end_idx] = service_cycles
        
    #     # Update buffer status
    #     self.free_space += data_size
    #     self.drain_buf_start_line_id = self.drain_buf_end_line_id
        
    #     # Return final service cycle
    #     return empty_start_cycle + num_lines
    #



    def empty_all_buffers(self, cycle):
        self.append_to_trace_mat(force=True)

        if self.trace_matrix_empty:
           return

        while self.drain_buf_start_line_id < self.trace_matrix.shape[0]:
            self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=cycle)
            cycle = self.drain_end_cycle + 1

    #
    # def get_trace_matrix(self):
    #     if not self.trace_valid:
    #         print('No trace has been generated yet')
    #         return

    #     trace_matrix = np.concatenate((self.cycles_vec, self.trace_matrix), axis=1)

    #     return trace_matrix

    # deepseek
    def get_trace_matrix(self) -> Optional[np.ndarray]:
        if not self.trace_valid:
            print("No trace generated yet")
            return None
        return np.column_stack((self.cycles_vec, self.trace_matrix))

    #
    def get_free_space(self):
        return self.free_space

    #
    def get_num_accesses(self):
        assert self.trace_valid, 'Traces not ready yet'
        return self.num_access

    #
    def get_external_access_start_stop_cycles(self):
        assert self.trace_valid, 'Traces not ready yet'
        start_cycle = self.cycles_vec[0][0]
        end_cycle = self.cycles_vec[-1][0]

        return start_cycle, end_cycle
    
    #  deepseek
    # def get_external_access_start_stop_cycles(self) -> Tuple[int, int]:
    #     """Get the start and end cycles of external accesses.
        
    #     Returns:
    #         Tuple of (start_cycle, end_cycle)
        
    #     Raises:
    #         AssertionError: If traces are not yet valid
    #     """
    #     assert self.trace_valid, 'Traces not ready yet'
        
    #     # cycles_vec is 1D array, so we don't need [0][0] indexing
    #     start_cycle = self.cycles_vec[0]  # First element
    #     end_cycle = self.cycles_vec[-1]   # Last element

    #     return start_cycle, end_cycle

    #
    def print_trace(self, filename):
        if not self.trace_valid:
            print('No trace has been generated yet')
            return
        trace_matrix = self.get_trace_matrix()
        np.savetxt(filename, trace_matrix, fmt='%s', delimiter=",")
