import time
import numpy as np
from tqdm import tqdm
from typing import Tuple #ds

from scalesim.memory.read_buffer import read_buffer as rdbuf
from memory.read_buffer_estimate_bw import ReadBufferEstimateBw as rdbuf_est
from scalesim.memory.read_port import read_port as rdport
from memory.write_buffer import write_buffer as wrbuf
from scalesim.memory.write_port import write_port as wrport


class double_buffered_scratchpad:
    # def __init__(self):
    #     self.ifmap_buf = rdbuf()
    #     self.filter_buf = rdbuf()
    #     self.ofmap_buf =wrbuf()

    #     self.ifmap_port = rdport()
    #     self.filter_port = rdport()
    #     self.ofmap_port = wrport()

    #     self.verbose = True

    #     self.ifmap_trace_matrix = np.zeros((1,1), dtype=int)
    #     self.filter_trace_matrix = np.zeros((1,1), dtype=int)
    #     self.ofmap_trace_matrix = np.zeros((1,1), dtype=int)

    #     # Metrics to gather for generating run reports
    #     self.total_cycles = 0
    #     self.compute_cycles = 0
    #     self.stall_cycles = 0

    #     self.avg_ifmap_dram_bw = 0
    #     self.avg_filter_dram_bw = 0
    #     self.avg_ofmap_dram_bw = 0

    #     self.ifmap_sram_start_cycle = 0
    #     self.ifmap_sram_stop_cycle = 0
    #     self.filter_sram_start_cycle = 0
    #     self.filter_sram_stop_cycle = 0
    #     self.ofmap_sram_start_cycle = 0
    #     self.ofmap_sram_stop_cycle = 0

    #     self.ifmap_dram_start_cycle = 0
    #     self.ifmap_dram_stop_cycle = 0
    #     self.ifmap_dram_reads = 0
    #     self.filter_dram_start_cycle = 0
    #     self.filter_dram_stop_cycle = 0
    #     self.filter_dram_reads = 0
    #     self.ofmap_dram_start_cycle = 0
    #     self.ofmap_dram_stop_cycle = 0
    #     self.ofmap_dram_writes = 0

    #     self.estimate_bandwidth_mode = False,
    #     self.traces_valid = False
    #     self.params_valid_flag = True


    def __init__(self):
        """
        Double-buffered scratchpad memory system for DNN accelerator simulation.
        Tracks all memory operations with cycle-accurate metrics for performance analysis.
        """
        # Memory buffers
        self.ifmap_buf = rdbuf()    # Input feature map buffer (read-only)
        self.filter_buf = rdbuf()   # Filter weights buffer (read-only)
        self.ofmap_buf = wrbuf()    # Output feature map buffer (write-only)
        
        # Memory access ports
        self.ifmap_port = rdport()  # IFMAP read port
        self.filter_port = rdport() # Filter read port
        self.ofmap_port = wrport()  # OFMAP write port
        
        # Debugging and control flags
        self.verbose = False        # Verbose output control
        self.estimate_bandwidth_mode = False  # Bandwidth estimation mode
        self.traces_valid = False   # Trace data validity flag
        self.params_valid_flag = True  # Parameter validation flag
        
        # Trace matrices (cycle x address)
        self.ifmap_trace_matrix = np.zeros((0, 2), dtype=np.uint32)  # [cycle, address]
        self.filter_trace_matrix = np.zeros((0, 2), dtype=np.uint32)
        self.ofmap_trace_matrix = np.zeros((0, 2), dtype=np.uint32)
        
        # Performance metrics
        self.total_cycles = 0       # Total execution cycles
        self.compute_cycles = 0     # Active computation cycles
        self.stall_cycles = 0       # Memory stall cycles
        
        # Bandwidth metrics (words/cycle)
        self.avg_ifmap_dram_bw = 0.0  # IFMAP DRAM bandwidth
        self.avg_filter_dram_bw = 0.0 # Filter DRAM bandwidth
        self.avg_ofmap_dram_bw = 0.0  # OFMAP DRAM bandwidth
        
        # SRAM access tracking
        self.ifmap_sram_start_cycle = 0  # IFMAP SRAM load start
        self.ifmap_sram_stop_cycle = 0   # IFMAP SRAM load end
        self.filter_sram_start_cycle = 0 # Filter SRAM load start
        self.filter_sram_stop_cycle = 0  # Filter SRAM load end
        self.ofmap_sram_start_cycle = 0  # OFMAP SRAM store start
        self.ofmap_sram_stop_cycle = 0   # OFMAP SRAM store end
        
        # DRAM access tracking
        self.ifmap_dram_start_cycle = 0  # IFMAP DRAM read start
        self.ifmap_dram_stop_cycle = 0   # IFMAP DRAM read end
        self.ifmap_dram_reads = 0        # IFMAP DRAM read count
        self.filter_dram_start_cycle = 0 # Filter DRAM read start
        self.filter_dram_stop_cycle = 0  # Filter DRAM read end
        self.filter_dram_reads = 0       # Filter DRAM read count
        self.ofmap_dram_start_cycle = 0  # OFMAP DRAM write start
        self.ofmap_dram_stop_cycle = 0   # OFMAP DRAM write end
        self.ofmap_dram_writes = 0       # OFMAP DRAM write count
        
        # Double buffering state
        self._current_buffer = 0         # Active buffer index (0 or 1)
        self._buffer_ready = [False, False]  # Buffer readiness flags

    #
    # def set_params(self,
    #                verbose=True,
    #                estimate_bandwidth_mode=False,
    #                word_size=1,
    #                ifmap_buf_size_bytes=2, filter_buf_size_bytes=2, ofmap_buf_size_bytes=2,
    #                rd_buf_active_frac=0.5, wr_buf_active_frac=0.5,
    #                ifmap_backing_buf_bw=1, filter_backing_buf_bw=1, ofmap_backing_buf_bw=1):

    #     self.estimate_bandwidth_mode = estimate_bandwidth_mode

    #     if self.estimate_bandwidth_mode:
    #         self.ifmap_buf = rdbuf_est()
    #         self.filter_buf = rdbuf_est()

    #         self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
    #                                   total_size_bytes=ifmap_buf_size_bytes,
    #                                   word_size=word_size,
    #                                   active_buf_frac=rd_buf_active_frac,
    #                                   backing_buf_default_bw=ifmap_backing_buf_bw)

    #         self.filter_buf.set_params(backing_buf_obj=self.filter_port,
    #                                    total_size_bytes=filter_buf_size_bytes,
    #                                    word_size=word_size,
    #                                    active_buf_frac=rd_buf_active_frac,
    #                                    backing_buf_default_bw=filter_backing_buf_bw)
    #     else:
    #         self.ifmap_buf = rdbuf()
    #         self.filter_buf = rdbuf()

    #         self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
    #                                   total_size_bytes=ifmap_buf_size_bytes,
    #                                   word_size=word_size,
    #                                   active_buf_frac=rd_buf_active_frac,
    #                                   backing_buf_bw=ifmap_backing_buf_bw)

    #         self.filter_buf.set_params(backing_buf_obj=self.filter_port,
    #                                    total_size_bytes=filter_buf_size_bytes,
    #                                    word_size=word_size,
    #                                    active_buf_frac=rd_buf_active_frac,
    #                                    backing_buf_bw=filter_backing_buf_bw)

    #     self.ofmap_buf.set_params(backing_buf_obj=self.ofmap_port,
    #                               total_size_bytes=ofmap_buf_size_bytes,
    #                               word_size=word_size,
    #                               active_buf_frac=wr_buf_active_frac,
    #                               backing_buf_bw=ofmap_backing_buf_bw)

    #     self.verbose = verbose

    #     self.params_valid_flag = True

    # def set_params(self,
    #             verbose: bool = True,
    #             estimate_bandwidth_mode: bool = False,
    #             word_size: int = 1,
    #             ifmap_buf_size_bytes: int = 2,
    #             filter_buf_size_bytes: int = 2,
    #             ofmap_buf_size_bytes: int = 2,
    #             rd_buf_active_frac: float = 0.5,
    #             wr_buf_active_frac: float = 0.5,
    #             ifmap_backing_buf_bw: int = 1,
    #             filter_backing_buf_bw: int = 1,
    #             ofmap_backing_buf_bw: int = 1) -> None:
    #     """
    #     Configures the double-buffered scratchpad with specified parameters.
        
    #     Args:
    #         verbose: Enable verbose logging output
    #         estimate_bandwidth_mode: Use bandwidth estimation buffers when True
    #         word_size: Data word size in bytes
    #         ifmap_buf_size_bytes: IFMAP buffer capacity in bytes
    #         filter_buf_size_bytes: Filter buffer capacity in bytes
    #         ofmap_buf_size_bytes: OFMAP buffer capacity in bytes
    #         rd_buf_active_frac: Active fraction of read buffers (IFMAP/filter)
    #         wr_buf_active_frac: Active fraction of write buffer (OFMAP)
    #         ifmap_backing_buf_bw: IFMAP backing buffer bandwidth (words/cycle)
    #         filter_backing_buf_bw: Filter backing buffer bandwidth (words/cycle)
    #         ofmap_backing_buf_bw: OFMAP backing buffer bandwidth (words/cycle)
            
    #     Raises:
    #         ValueError: For invalid parameter values
    #     """
    #     # 1. Validate input parameters
    #     self._validate_params(word_size, 
    #                         ifmap_buf_size_bytes,
    #                         filter_buf_size_bytes,
    #                         ofmap_buf_size_bytes,
    #                         rd_buf_active_frac,
    #                         wr_buf_active_frac,
    #                         ifmap_backing_buf_bw,
    #                         filter_backing_buf_bw,
    #                         ofmap_backing_buf_bw)

    #     # 2. Set operational modes
    #     self.verbose = verbose
    #     self.estimate_bandwidth_mode = estimate_bandwidth_mode
        
    #     # 3. Initialize buffers based on bandwidth estimation mode
    #     buf_class = rdbuf_est if estimate_bandwidth_mode else rdbuf
    #     self.ifmap_buf = buf_class()
    #     self.filter_buf = buf_class()
    #     self.ofmap_buf = wrbuf()  # Always use standard write buffer

    #     # 4. Configure IFMAP buffer
    #     self.ifmap_buf.set_params(
    #         backing_buf_obj=self.ifmap_port,
    #         total_size_bytes=ifmap_buf_size_bytes,
    #         word_size=word_size,
    #         active_buf_frac=rd_buf_active_frac,
    #         backing_buf_default_bw=ifmap_backing_buf_bw if estimate_bandwidth_mode else None,
    #         backing_buf_bw=ifmap_backing_buf_bw
    #     )

    #     # 5. Configure Filter buffer
    #     self.filter_buf.set_params(
    #         backing_buf_obj=self.filter_port,
    #         total_size_bytes=filter_buf_size_bytes,
    #         word_size=word_size,
    #         active_buf_frac=rd_buf_active_frac,
    #         backing_buf_default_bw=filter_backing_buf_bw if estimate_bandwidth_mode else None,
    #         backing_buf_bw=filter_backing_buf_bw
    #     )

    #     # 6. Configure OFMAP buffer
    #     self.ofmap_buf.set_params(
    #         backing_buf_obj=self.ofmap_port,
    #         total_size_bytes=ofmap_buf_size_bytes,
    #         word_size=word_size,
    #         active_buf_frac=wr_buf_active_frac,
    #         backing_buf_bw=ofmap_backing_buf_bw
    #     )

    #     # 7. Mark parameters as valid
    #     self.params_valid_flag = True
    def set_params(self,
                verbose: bool = True,
                estimate_bandwidth_mode: bool = False,
                word_size: int = 1,
                ifmap_buf_size_bytes: int = 2,
                filter_buf_size_bytes: int = 2,
                ofmap_buf_size_bytes: int = 2,
                rd_buf_active_frac: float = 0.5,
                wr_buf_active_frac: float = 0.5,
                ifmap_backing_buf_bw: int = 1,
                filter_backing_buf_bw: int = 1,
                ofmap_backing_buf_bw: int = 1) -> None:
        """
        Configures the double-buffered scratchpad with specified parameters.
        
        Args:
            verbose: Enable verbose logging output
            estimate_bandwidth_mode: Use bandwidth estimation buffers when True
            word_size: Data word size in bytes
            ifmap_buf_size_bytes: IFMAP buffer capacity in bytes
            filter_buf_size_bytes: Filter buffer capacity in bytes
            ofmap_buf_size_bytes: OFMAP buffer capacity in bytes
            rd_buf_active_frac: Active fraction of read buffers (IFMAP/filter)
            wr_buf_active_frac: Active fraction of write buffer (OFMAP)
            ifmap_backing_buf_bw: IFMAP backing buffer bandwidth (words/cycle)
            filter_backing_buf_bw: Filter backing buffer bandwidth (words/cycle)
            ofmap_backing_buf_bw: OFMAP backing buffer bandwidth (words/cycle)
        """
        # 1. Validate input parameters
        self._validate_params(word_size, 
                        ifmap_buf_size_bytes,
                        filter_buf_size_bytes,
                        ofmap_buf_size_bytes,
                        rd_buf_active_frac,
                        wr_buf_active_frac,
                        ifmap_backing_buf_bw,
                        filter_backing_buf_bw,
                        ofmap_backing_buf_bw)

        # 2. Set operational modes
        self.verbose = verbose
        self.estimate_bandwidth_mode = estimate_bandwidth_mode
        
        # 3. Initialize buffers with appropriate classes
        if estimate_bandwidth_mode:
            self.ifmap_buf = rdbuf_est()
            self.filter_buf = rdbuf_est()
            # Configure with backing_buf_default_bw
            self.ifmap_buf.set_params(
                backing_buf_obj=self.ifmap_port,
                total_size_bytes=ifmap_buf_size_bytes,
                word_size=word_size,
                active_buf_frac=rd_buf_active_frac,
                backing_buf_default_bw=ifmap_backing_buf_bw
            )
            self.filter_buf.set_params(
                backing_buf_obj=self.filter_port,
                total_size_bytes=filter_buf_size_bytes,
                word_size=word_size,
                active_buf_frac=rd_buf_active_frac,
                backing_buf_default_bw=filter_backing_buf_bw
            )
        else:
            self.ifmap_buf = rdbuf()
            self.filter_buf = rdbuf()
            # Configure with backing_buf_bw
            self.ifmap_buf.set_params(
                backing_buf_obj=self.ifmap_port,
                total_size_bytes=ifmap_buf_size_bytes,
                word_size=word_size,
                active_buf_frac=rd_buf_active_frac,
                backing_buf_bw=ifmap_backing_buf_bw
            )
            self.filter_buf.set_params(
                backing_buf_obj=self.filter_port,
                total_size_bytes=filter_buf_size_bytes,
                word_size=word_size,
                active_buf_frac=rd_buf_active_frac,
                backing_buf_bw=filter_backing_buf_bw
            )
        
        # 4. Always configure OFMAP buffer the same way
        self.ofmap_buf = wrbuf()
        self.ofmap_buf.set_params(
            backing_buf_obj=self.ofmap_port,
            total_size_bytes=ofmap_buf_size_bytes,
            word_size=word_size,
            active_buf_frac=wr_buf_active_frac,
            backing_buf_bw=ofmap_backing_buf_bw
        )

        # 5. Mark parameters as valid
        self.params_valid_flag = True
    def _validate_params(self,
                        word_size: int,
                        ifmap_buf_size: int,
                        filter_buf_size: int,
                        ofmap_buf_size: int,
                        rd_buf_frac: float,
                        wr_buf_frac: float,
                        ifmap_bw: int,
                        filter_bw: int,
                        ofmap_bw: int) -> None:
        """Validates all configuration parameters."""
        assert word_size > 0, "Word size must be positive"
        assert ifmap_buf_size > 0, "IFMAP buffer size must be positive"
        assert filter_buf_size > 0, "Filter buffer size must be positive"
        assert ofmap_buf_size > 0, "OFMAP buffer size must be positive"
        assert 0 < rd_buf_frac <= 1.0, "Read buffer active fraction must be in (0, 1]"
        assert 0 < wr_buf_frac <= 1.0, "Write buffer active fraction must be in (0, 1]"
        assert ifmap_bw > 0, "IFMAP bandwidth must be positive"
        assert filter_bw > 0, "Filter bandwidth must be positive"
        assert ofmap_bw > 0, "OFMAP bandwidth must be positive"

    #
    # def set_read_buf_prefetch_matrices(self,
    #                                    ifmap_prefetch_mat=np.zeros((1,1)),
    #                                    filter_prefetch_mat=np.zeros((1,1))
    #                                    ):

    #     self.ifmap_buf.set_fetch_matrix(ifmap_prefetch_mat)
    #     self.filter_buf.set_fetch_matrix(filter_prefetch_mat)

    def set_read_buf_prefetch_matrices(self, 
                                    ifmap_prefetch_mat: np.ndarray,
                                    filter_prefetch_mat: np.ndarray) -> None:
        """
        Configures prefetch matrices for both read buffers (IFMAP and filter).
        
        Args:
            ifmap_prefetch_mat: Prefetch matrix for IFMAP buffer with shape [N, 2] where:
                - Column 0: Prefetch cycle
                - Column 1: Memory address
            filter_prefetch_mat: Prefetch matrix for filter buffer with same format
                
        Raises:
            ValueError: If matrices have invalid shape or the buffers aren't initialized
            RuntimeError: If parameters haven't been set via set_params()
        """
        # 1. Validate parameters are set
        if not self.params_valid_flag:
            raise RuntimeError("Parameters must be set via set_params() before configuring prefetch matrices")
        
        # 2. Validate buffer initialization
        if not hasattr(self, 'ifmap_buf') or not hasattr(self, 'filter_buf'):
            raise ValueError("Read buffers must be initialized before setting prefetch matrices")
        
        # 3. Validate matrix shapes
        if ifmap_prefetch_mat.shape[1] != 2:
            raise ValueError(f"IFMAP prefetch matrix must have shape [N,2], got {ifmap_prefetch_mat.shape}")
        if filter_prefetch_mat.shape[1] != 2:
            raise ValueError(f"Filter prefetch matrix must have shape [N,2], got {filter_prefetch_mat.shape}")
        
        # 4. Set IFMAP prefetch matrix
        if self.verbose:
            print(f"Setting IFMAP prefetch matrix with {ifmap_prefetch_mat.shape[0]} entries")
        self.ifmap_buf.set_prefetch_matrix(ifmap_prefetch_mat)
        
        # 5. Set filter prefetch matrix
        if self.verbose:
            print(f"Setting filter prefetch matrix with {filter_prefetch_mat.shape[0]} entries")
        self.filter_buf.set_prefetch_matrix(filter_prefetch_mat)
        
        # 6. Update trace validity
        self.traces_valid = True

    #
    # def reset_buffer_states(self):

    #     self.ifmap_buf.reset()
    #     self.filter_buf.reset()
    #     self.ofmap_buf.reset()

    def reset_buffer_states(self) -> None:
        """
        Resets the internal states of all data buffers to their initial conditions.
        This operation:
        - Clears all stored data in the buffers
        - Resets internal pointers and state machines
        - Maintains current buffer configurations (size, bandwidth etc.)
        - Preserves the parameter validity flag
        
        Typical use cases:
        - Between simulation runs with different data patterns
        - After configuration changes requiring buffer clearing
        - For debugging and testing purposes
        
        Note: Does not modify buffer sizes or bandwidth configurations.
        """
        # 1. Validate buffer initialization
        if not all(hasattr(self, buf) for buf in ['ifmap_buf', 'filter_buf', 'ofmap_buf']):
            raise RuntimeError("Buffer objects not initialized - call set_params() first")
        
        # 2. Reset IFMAP buffer state
        if self.verbose:
            print("[Buffer] Resetting IFMAP buffer state")
        self.ifmap_buf.reset()
        
        # 3. Reset filter buffer state
        if self.verbose:
            print("[Buffer] Resetting filter buffer state")
        self.filter_buf.reset()
        
        # 4. Reset OFMAP buffer state
        if self.verbose:
            print("[Buffer] Resetting OFMAP buffer state")
        self.ofmap_buf.reset()
        
        # 5. Log completion if in verbose mode
        if self.verbose:
            print("[Buffer] All buffer states reset successfully")

    # The following are just shell methods for users to control each mem individually
    # def service_ifmap_reads(self,
    #                         incoming_requests_arr_np,   # 2D array with the requests
    #                         incoming_cycles_arr):
    #     out_cycles_arr_np = self.ifmap_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

    #     return out_cycles_arr_np
    def service_ifmap_reads(self,
                        incoming_requests_arr_np: np.ndarray,
                        incoming_cycles_arr: np.ndarray) -> np.ndarray:
        """
        Services read requests for the Input Feature Map (IFMAP) buffer.
        
        Args:
            incoming_requests_arr_np: 2D numpy array of read requests where:
                - Each row represents a set of concurrent requests
                - -1 indicates no request
                - Valid addresses are positive integers
            incoming_cycles_arr: 1D numpy array of corresponding cycle numbers
                for each request set
                
        Returns:
            out_cycles_arr_np: 1D numpy array of cycles when each request set
            will be serviced
            
        Raises:
            RuntimeError: If IFMAP buffer is not initialized
            ValueError: If input arrays have incompatible shapes
        """
        # 1. Validate buffer initialization
        if not hasattr(self, 'ifmap_buf'):
            raise RuntimeError("IFMAP buffer not initialized - call set_params() first")
        
        # 2. Validate input shapes
        if incoming_requests_arr_np.shape[0] != incoming_cycles_arr.shape[0]:
            raise ValueError(
                f"Mismatched input dimensions: requests {incoming_requests_arr_np.shape} "
                f"vs cycles {incoming_cycles_arr.shape}"
            )
        
        # 3. Service reads through IFMAP buffer
        if self.verbose:
            print(f"[IFMAP] Servicing {incoming_requests_arr_np.shape[0]} request sets")
        
        out_cycles_arr_np = self.ifmap_buf.service_reads(
            incoming_requests_arr_np=incoming_requests_arr_np,
            incoming_cycles_arr=incoming_cycles_arr
        )
        
        # 4. Update trace if in bandwidth estimation mode
        if self.estimate_bandwidth_mode:
            self._update_ifmap_trace(
                requests=incoming_requests_arr_np,
                cycles=out_cycles_arr_np
            )
        
        return out_cycles_arr_np

    #
    # def service_filter_reads(self,
    #                         incoming_requests_arr_np,   # 2D array with the requests
    #                         incoming_cycles_arr):
    #     out_cycles_arr_np = self.filter_buf.service_reads(incoming_requests_arr_np, incoming_cycles_arr)

    #     return out_cycles_arr_np
    def service_filter_reads(self,
                            incoming_requests_arr_np: np.ndarray,
                            incoming_cycles_arr: np.ndarray) -> np.ndarray:
        """
        Services read requests for the filter weights buffer.
        
        Args:
            incoming_requests_arr_np: 2D numpy array of filter read requests where:
                - Each row represents a batch of concurrent requests
                - Each element is either a valid memory address (>=0) or -1 (no request)
                - Shape: [num_request_batches, max_concurrent_requests]
            incoming_cycles_arr: 1D numpy array of cycle numbers when each request batch 
                is issued. Shape: [num_request_batches]
                
        Returns:
            out_cycles_arr_np: 1D numpy array of completion cycles for each request batch.
            Shape: [num_request_batches]
            
        Raises:
            RuntimeError: If filter buffer is not initialized or parameters are invalid
            ValueError: If input arrays have incompatible shapes or invalid values
        """
        # 1. Validate system state
        if not (hasattr(self, 'filter_buf') and self.params_valid_flag):
            raise RuntimeError("Filter buffer not initialized or parameters invalid")
        
        # 2. Validate input dimensions
        if incoming_requests_arr_np.shape[0] != incoming_cycles_arr.shape[0]:
            raise ValueError(
                f"Request/cycle dimension mismatch: "
                f"{incoming_requests_arr_np.shape[0]} batches vs "
                f"{incoming_cycles_arr.shape[0]} cycle entries"
            )
        
        # 3. Service requests through filter buffer
        if self.verbose:
            num_requests = np.count_nonzero(incoming_requests_arr_np >= 0)
            print(f"[Filter] Servicing {num_requests} requests across "
                f"{incoming_requests_arr_np.shape[0]} batches")
        
        out_cycles_arr_np = self.filter_buf.service_reads(
            incoming_requests_arr_np=incoming_requests_arr_np,
            incoming_cycles_arr=incoming_cycles_arr
        )
        
        # 4. Update bandwidth estimation traces if enabled
        if self.estimate_bandwidth_mode:
            self._update_filter_trace(
                requests=incoming_requests_arr_np,
                cycles=out_cycles_arr_np
            )
        
        return out_cycles_arr_np
    #
    # def service_ofmap_writes(self,
    #                          incoming_requests_arr_np,  # 2D array with the requests
    #                          incoming_cycles_arr):

    #     out_cycles_arr_np = self.ofmap_buf.service_writes(incoming_requests_arr_np, incoming_cycles_arr)

    #     return out_cycles_arr_np

    def service_ofmap_writes(self,
                            incoming_requests_arr_np: np.ndarray,
                            incoming_cycles_arr: np.ndarray) -> np.ndarray:
        """
        Services write requests to the Output Feature Map (OFMAP) buffer.
        
        Args:
            incoming_requests_arr_np: 2D numpy array of write requests where:
                - Each row represents a batch of concurrent writes
                - Each element contains either:
                * A valid memory address (>=0) and data tuple (address, value)
                * -1 for no request
                - Shape: [num_request_batches, max_concurrent_writes]
            incoming_cycles_arr: 1D numpy array of issue cycles for each batch.
                Shape: [num_request_batches]
                
        Returns:
            out_cycles_arr_np: 1D numpy array of completion cycles for each batch.
            Shape: [num_request_batches]
            
        Raises:
            RuntimeError: If OFMAP buffer not initialized or parameters invalid
            ValueError: For invalid input shapes or request formats
        """
        # 1. System state validation
        if not (hasattr(self, 'ofmap_buf') and self.params_valid_flag):
            raise RuntimeError("OFMAP buffer not initialized or parameters invalid")
        
        # 2. Input validation
        if incoming_requests_arr_np.shape[0] != incoming_cycles_arr.shape[0]:
            raise ValueError(
                f"Request/cycle dimension mismatch: "
                f"{incoming_requests_arr_np.shape[0]} batches vs "
                f"{incoming_cycles_arr.shape[0]} cycle entries"
            )
        
        # 3. Process write requests
        if self.verbose:
            num_writes = np.count_nonzero(incoming_requests_arr_np[:,:,0] >= 0)  # Count valid addresses
            print(f"[OFMAP] Processing {num_writes} writes across "
                f"{incoming_requests_arr_np.shape[0]} batches")
        
        out_cycles_arr_np = self.ofmap_buf.service_writes(
            incoming_requests_arr_np=incoming_requests_arr_np,
            incoming_cycles_arr=incoming_cycles_arr
        )
        
        # 4. Update bandwidth estimation traces if enabled
        if self.estimate_bandwidth_mode:
            self._update_ofmap_trace(
                requests=incoming_requests_arr_np,
                cycles=out_cycles_arr_np
            )
        
        # 5. Validate completion cycles
        if np.any(out_cycles_arr_np < incoming_cycles_arr):
            if self.verbose:
                print("[OFMAP] Warning: Some completion cycles precede issue cycles")
        
        return out_cycles_arr_np

    #
    # def service_memory_requests(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
    #     assert self.params_valid_flag, 'Memories not initialized yet'

    #     ofmap_lines = ofmap_demand_mat.shape[0]

    #     self.total_cycles = 0
    #     self.stall_cycles = 0

    #     ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
    #     filter_hit_latency = self.filter_buf.get_hit_latency()

    #     ifmap_serviced_cycles = []
    #     filter_serviced_cycles = []
    #     ofmap_serviced_cycles = []

    #     pbar_disable = not self.verbose
    #     for i in tqdm(range(ofmap_lines), disable=pbar_disable):

    #         cycle_arr = np.zeros((1,1)) + i + self.stall_cycles

    #         ifmap_demand_line = ifmap_demand_mat[i, :].reshape((1,ifmap_demand_mat.shape[1]))
    #         ifmap_cycle_out = self.ifmap_buf.service_reads(incoming_requests_arr_np=ifmap_demand_line,
    #                                                         incoming_cycles_arr=cycle_arr)
    #         ifmap_serviced_cycles += [ifmap_cycle_out[0]]
    #         ifmap_stalls = ifmap_cycle_out[0] - cycle_arr[0] - ifmap_hit_latency

    #         filter_demand_line = filter_demand_mat[i, :].reshape((1, filter_demand_mat.shape[1]))
    #         filter_cycle_out = self.filter_buf.service_reads(incoming_requests_arr_np=filter_demand_line,
    #                                                        incoming_cycles_arr=cycle_arr)
    #         filter_serviced_cycles += [filter_cycle_out[0]]
    #         filter_stalls = filter_cycle_out[0] - cycle_arr[0] - filter_hit_latency

    #         ofmap_demand_line = ofmap_demand_mat[i, :].reshape((1, ofmap_demand_mat.shape[1]))
    #         ofmap_cycle_out = self.ofmap_buf.service_writes(incoming_requests_arr_np=ofmap_demand_line,
    #                                                          incoming_cycles_arr_np=cycle_arr)
    #         ofmap_serviced_cycles += [ofmap_cycle_out[0]]
    #         ofmap_stalls = ofmap_cycle_out[0] - cycle_arr[0] - 1

    #         self.stall_cycles += int(max(ifmap_stalls[0], filter_stalls[0], ofmap_stalls[0]))

    #     if self.estimate_bandwidth_mode:
    #         # IDE shows warning as complete_all_prefetches is not implemented in read_buffer class
    #         # It is harmless since, in estimate bandwidth mode, read_buffer_estimate_bw is instantiated
    #         self.ifmap_buf.complete_all_prefetches()
    #         self.filter_buf.complete_all_prefetches()

    #     self.ofmap_buf.empty_all_buffers(ofmap_serviced_cycles[-1])

    #     # Prepare the traces
    #     ifmap_services_cycles_np = np.asarray(ifmap_serviced_cycles).reshape((len(ifmap_serviced_cycles), 1))
    #     self.ifmap_trace_matrix = np.concatenate((ifmap_services_cycles_np, ifmap_demand_mat), axis=1)

    #     filter_services_cycles_np = np.asarray(filter_serviced_cycles).reshape((len(filter_serviced_cycles), 1))
    #     self.filter_trace_matrix = np.concatenate((filter_services_cycles_np, filter_demand_mat), axis=1)

    #     ofmap_services_cycles_np = np.asarray(ofmap_serviced_cycles).reshape((len(ofmap_serviced_cycles), 1))
    #     self.ofmap_trace_matrix = np.concatenate((ofmap_services_cycles_np, ofmap_demand_mat), axis=1)
    #     self.total_cycles = int(ofmap_serviced_cycles[-1][0])

    #     # END of serving demands from memory
    #     self.traces_valid = True

    # def service_memory_requests(self,
    #                         ifmap_demand_mat: np.ndarray,
    #                         filter_demand_mat: np.ndarray,
    #                         ofmap_demand_mat: np.ndarray) -> None:
    #     """
    #     Services complete memory request patterns for IFMAP, filter, and OFMAP buffers.
        
    #     Args:
    #         ifmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of IFMAP read requests
    #         filter_demand_mat: 2D array [num_cycles, max_req_per_cycle] of filter read requests
    #         ofmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of OFMAP write requests
            
    #     Raises:
    #         RuntimeError: If parameters are invalid or buffers not initialized
    #         ValueError: If demand matrices have incompatible shapes
    #     """
    #     # 1. System state validation
    #     if not (self.params_valid_flag and 
    #             hasattr(self, 'ifmap_buf') and 
    #             hasattr(self, 'filter_buf') and 
    #             hasattr(self, 'ofmap_buf')):
    #         raise RuntimeError("System not properly initialized")
        
    #     # 2. Input validation
    #     if ifmap_demand_mat.shape[0] != filter_demand_mat.shape[0] or \
    #     filter_demand_mat.shape[0] != ofmap_demand_mat.shape[0]:
    #         raise ValueError("Demand matrices must have same number of cycles")
        
    #     # 3. Initialize tracking variables
    #     num_cycles = ofmap_demand_mat.shape[0]
    #     ifmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     filter_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     ofmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     stall_cycles = np.zeros(3, dtype=np.uint32)  # [ifmap, filter, ofmap]
        
    #     # 4. Get buffer latencies
    #     ifmap_hit_latency = self.ifmap_buf.hit_latency
    #     filter_hit_latency = self.filter_buf.hit_latency
        
    #     # 5. Process each cycle's demands
    #     for cycle in range(num_cycles):
    #         # Reshape demand vectors
    #         ifmap_demand = ifmap_demand_mat[cycle].reshape(1, -1)
    #         filter_demand = filter_demand_mat[cycle].reshape(1, -1)
    #         ofmap_demand = ofmap_demand_mat[cycle].reshape(1, -1)
            
    #         # Service requests (fixed missing parentheses)
    #         ifmap_cycles = self.ifmap_buf.service_reads(
    #             ifmap_demand, 
    #             np.array([cycle])
    #         )
    #         filter_cycles = self.filter_buf.service_reads(
    #             filter_demand, 
    #             np.array([cycle])
    #         )
    #         ofmap_cycles = self.ofmap_buf.service_writes(
    #             ofmap_demand, 
    #             np.array([cycle])
    #         )
            
    #         # Calculate stalls
    #         stall_cycles[0] += max(0, ifmap_cycles[0] - cycle - ifmap_hit_latency)
    #         stall_cycles[1] += max(0, filter_cycles[0] - cycle - filter_hit_latency)
    #         stall_cycles[2] += max(0, ofmap_cycles[0] - cycle)
            
    #         # Store results
    #         ifmap_serviced_cycles[cycle] = ifmap_cycles[0]
    #         filter_serviced_cycles[cycle] = filter_cycles[0]
    #         ofmap_serviced_cycles[cycle] = ofmap_cycles[0]
        
    #     # 6. Complete prefetches if in estimation mode
    #     if self.estimate_bandwidth_mode:
    #         self.ifmap_buf.complete_all_prefetches()
    #         self.filter_buf.complete_all_prefetches()
        
    #     # 7. Prepare trace matrices
    #     self.ifmap_trace_matrix = np.column_stack((
    #         ifmap_serviced_cycles, 
    #         ifmap_demand_mat
    #     ))
    #     self.filter_trace_matrix = np.column_stack((
    #         filter_serviced_cycles,
    #         filter_demand_mat
    #     ))
    #     self.ofmap_trace_matrix = np.column_stack((
    #         ofmap_serviced_cycles,
    #         ofmap_demand_mat
    #     ))
        
    #     # 8. Update performance metrics
    #     self.total_cycles = int(np.max(ofmap_serviced_cycles)) + 1
    #     self.compute_cycles = num_cycles
    #     self.stall_cycles = int(np.sum(stall_cycles))
        
    #     # 9. Empty buffers and mark traces valid
    #     self.reset_buffer_states()
    #     self.traces_valid = True
        
    #     if self.verbose:
    #         print(f"[Memory] Completed {num_cycles} cycles with {self.stall_cycles} stalls")


    # def service_memory_requests(self,
    #                         ifmap_demand_mat: np.ndarray,
    #                         filter_demand_mat: np.ndarray,
    #                         ofmap_demand_mat: np.ndarray) -> None:
    #     """
    #     Services complete memory request patterns for IFMAP, filter, and OFMAP buffers.
        
    #     Args:
    #         ifmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of IFMAP read requests
    #         filter_demand_mat: 2D array [num_cycles, max_req_per_cycle] of filter read requests
    #         ofmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of OFMAP write requests
            
    #     Raises:
    #         RuntimeError: If parameters are invalid or buffers not initialized
    #         ValueError: If demand matrices have incompatible shapes
    #     """
    #     # 1. System state validation
    #     if not (self.params_valid_flag and 
    #             hasattr(self, 'ifmap_buf') and 
    #             hasattr(self, 'filter_buf') and 
    #             hasattr(self, 'ofmap_buf')):
    #         raise RuntimeError("System not properly initialized")
        
    #     # 2. Input validation
    #     if ifmap_demand_mat.shape[0] != filter_demand_mat.shape[0] or \
    #     filter_demand_mat.shape[0] != ofmap_demand_mat.shape[0]:
    #         raise ValueError("Demand matrices must have same number of cycles")
        
    #     # 3. Initialize tracking variables
    #     num_cycles = ofmap_demand_mat.shape[0]
    #     ifmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     filter_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     ofmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     stall_cycles = np.zeros(3, dtype=np.uint32)  # [ifmap, filter, ofmap]
        
    #     # 4. Get buffer latencies
    #     ifmap_hit_latency = self.ifmap_buf.hit_latency
    #     filter_hit_latency = self.filter_buf.hit_latency
        
    #     # 5. Process each cycle's demands
    #     for cycle in range(num_cycles):
    #         # Reshape demand vectors
    #         ifmap_demand = ifmap_demand_mat[cycle].reshape(1, -1)
    #         filter_demand = filter_demand_mat[cycle].reshape(1, -1)
    #         ofmap_demand = ofmap_demand_mat[cycle].reshape(1, -1)
            
    #         # Create properly shaped cycle arrays
    #         cycle_arr = np.array([cycle])
            
    #         # Service requests with properly shaped inputs
    #         ifmap_cycles = self.ifmap_buf.service_reads(
    #             incoming_requests_arr_np=ifmap_demand,
    #             incoming_cycles_arr=cycle_arr
    #         )
    #         filter_cycles = self.filter_buf.service_reads(
    #             incoming_requests_arr_np=filter_demand,
    #             incoming_cycles_arr=cycle_arr
    #         )
    #         ofmap_cycles = self.ofmap_buf.service_writes(
    #             incoming_requests_arr_np=ofmap_demand,
    #             incoming_cycles_arr=cycle_arr
    #         )
            
    #         # Extract scalar cycle values from 1-element arrays
    #         ifmap_cycle = ifmap_cycles.item() if ifmap_cycles.size == 1 else ifmap_cycles[0]
    #         filter_cycle = filter_cycles.item() if filter_cycles.size == 1 else filter_cycles[0]
    #         ofmap_cycle = ofmap_cycles.item() if ofmap_cycles.size == 1 else ofmap_cycles[0]
            
    #         # Calculate stalls
    #         stall_cycles[0] += max(0, ifmap_cycle - cycle - ifmap_hit_latency)
    #         stall_cycles[1] += max(0, filter_cycle - cycle - filter_hit_latency)
    #         stall_cycles[2] += max(0, ofmap_cycle - cycle)
            
    #         # Store results
    #         ifmap_serviced_cycles[cycle] = ifmap_cycle
    #         filter_serviced_cycles[cycle] = filter_cycle
    #         ofmap_serviced_cycles[cycle] = ofmap_cycle
        
    #     # 6. Complete prefetches if in estimation mode
    #     if self.estimate_bandwidth_mode:
    #         self.ifmap_buf.complete_all_prefetches()
    #         self.filter_buf.complete_all_prefetches()
        
    #     # 7. Prepare trace matrices
    #     self.ifmap_trace_matrix = np.column_stack((
    #         ifmap_serviced_cycles, 
    #         ifmap_demand_mat
    #     ))
    #     self.filter_trace_matrix = np.column_stack((
    #         filter_serviced_cycles,
    #         filter_demand_mat
    #     ))
    #     self.ofmap_trace_matrix = np.column_stack((
    #         ofmap_serviced_cycles,
    #         ofmap_demand_mat
    #     ))
        
    #     # 8. Update performance metrics
    #     self.total_cycles = int(np.max(ofmap_serviced_cycles)) + 1
    #     self.compute_cycles = num_cycles
    #     self.stall_cycles = int(np.sum(stall_cycles))
        
    #     # 9. Empty buffers and mark traces valid
    #     self.reset_buffer_states()
    #     self.traces_valid = True
        
    #     if self.verbose:
    #         print(f"[Memory] Completed {num_cycles} cycles with {self.stall_cycles} stalls")

    # def service_memory_requests(self,
    #                         ifmap_demand_mat: np.ndarray,
    #                         filter_demand_mat: np.ndarray,
    #                         ofmap_demand_mat: np.ndarray) -> None:
    #     """
    #     Services complete memory request patterns for IFMAP, filter, and OFMAP buffers.
        
    #     Args:
    #         ifmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of IFMAP read requests
    #         filter_demand_mat: 2D array [num_cycles, max_req_per_cycle] of filter read requests
    #         ofmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of OFMAP write requests
            
    #     Raises:
    #         RuntimeError: If parameters are invalid or buffers not initialized
    #         ValueError: If demand matrices have incompatible shapes
    #     """
    #     # 1. System state validation
    #     if not (self.params_valid_flag and 
    #             hasattr(self, 'ifmap_buf') and 
    #             hasattr(self, 'filter_buf') and 
    #             hasattr(self, 'ofmap_buf')):
    #         raise RuntimeError("System not properly initialized")
        
    #     # 2. Input validation
    #     if ifmap_demand_mat.shape[0] != filter_demand_mat.shape[0] or \
    #     filter_demand_mat.shape[0] != ofmap_demand_mat.shape[0]:
    #         raise ValueError("Demand matrices must have same number of cycles")
        
    #     # 3. Initialize tracking variables
    #     num_cycles = ofmap_demand_mat.shape[0]
    #     ifmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     filter_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     ofmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
    #     stall_cycles = np.zeros(3, dtype=np.uint32)  # [ifmap, filter, ofmap]
        
    #     # 4. Get buffer latencies
    #     ifmap_hit_latency = self.ifmap_buf.hit_latency
    #     filter_hit_latency = self.filter_buf.hit_latency
        
    #     # 5. Process each cycle's demands
    #     for cycle in range(num_cycles):
    #         # Reshape demand vectors to 2D arrays with single row
    #         ifmap_demand = ifmap_demand_mat[cycle].reshape(1, -1)
    #         filter_demand = filter_demand_mat[cycle].reshape(1, -1)
    #         ofmap_demand = ofmap_demand_mat[cycle].reshape(1, -1)
            
    #         # Create properly shaped cycle arrays (2D with single element)
    #         cycle_arr = np.array([[cycle]])
            
    #         # Service requests with properly shaped inputs
    #         ifmap_cycles = self.ifmap_buf.service_reads(
    #             incoming_requests_arr_np=ifmap_demand,
    #             incoming_cycles_arr=cycle_arr
    #         )
    #         filter_cycles = self.filter_buf.service_reads(
    #             incoming_requests_arr_np=filter_demand,
    #             incoming_cycles_arr=cycle_arr
    #         )
    #         ofmap_cycles = self.ofmap_buf.service_writes(
    #             incoming_requests_arr_np=ofmap_demand,
    #             incoming_cycles_arr_np=cycle_arr
    #         )
            
    #         # Extract completion cycles (handling both 1D and 2D returns)
    #         ifmap_cycle = ifmap_cycles[0][0] if ifmap_cycles.ndim == 2 else ifmap_cycles[0]
    #         filter_cycle = filter_cycles[0][0] if filter_cycles.ndim == 2 else filter_cycles[0]
    #         ofmap_cycle = ofmap_cycles[0][0] if ofmap_cycles.ndim == 2 else ofmap_cycles[0]
            
    #         # Calculate stalls
    #         stall_cycles[0] += max(0, ifmap_cycle - cycle - ifmap_hit_latency)
    #         stall_cycles[1] += max(0, filter_cycle - cycle - filter_hit_latency)
    #         stall_cycles[2] += max(0, ofmap_cycle - cycle)
            
    #         # Store results
    #         ifmap_serviced_cycles[cycle] = ifmap_cycle
    #         filter_serviced_cycles[cycle] = filter_cycle
    #         ofmap_serviced_cycles[cycle] = ofmap_cycle
        
    #     # 6. Complete prefetches if in estimation mode
    #     if self.estimate_bandwidth_mode:
    #         self.ifmap_buf.complete_all_prefetches()
    #         self.filter_buf.complete_all_prefetches()
    #     self.ofmap_buf.empty_all_buffers(ofmap_serviced_cycles[-1])
        
    #     # 7. Prepare trace matrices
    #     self.ifmap_trace_matrix = np.column_stack((
    #         ifmap_serviced_cycles, 
    #         ifmap_demand_mat
    #     ))
    #     self.filter_trace_matrix = np.column_stack((
    #         filter_serviced_cycles,
    #         filter_demand_mat
    #     ))
    #     self.ofmap_trace_matrix = np.column_stack((
    #         ofmap_serviced_cycles,
    #         ofmap_demand_mat
    #     ))
        
    #     # 8. Update performance metrics
    #     self.total_cycles = int(np.max(ofmap_serviced_cycles)) + 1
    #     self.compute_cycles = num_cycles
    #     self.stall_cycles = int(np.sum(stall_cycles))
        
    #     # 9. Empty buffers and mark traces valid
    #     # self.reset_buffer_states()
    #     self.traces_valid = True


    # ds
    def service_memory_requests(self,
                            ifmap_demand_mat: np.ndarray,
                            filter_demand_mat: np.ndarray,
                            ofmap_demand_mat: np.ndarray) -> None:
        """
        Services complete memory request patterns for IFMAP, filter, and OFMAP buffers.
        
        Args:
            ifmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of IFMAP read requests
            filter_demand_mat: 2D array [num_cycles, max_req_per_cycle] of filter read requests
            ofmap_demand_mat: 2D array [num_cycles, max_req_per_cycle] of OFMAP write requests
            
        Raises:
            RuntimeError: If parameters are invalid or buffers not initialized
            ValueError: If demand matrices have incompatible shapes
        """
        # 1. System state validation
        if not (self.params_valid_flag and 
                hasattr(self, 'ifmap_buf') and 
                hasattr(self, 'filter_buf') and 
                hasattr(self, 'ofmap_buf')):
            raise RuntimeError("System not properly initialized")
        
        # 2. Input validation
        if ifmap_demand_mat.shape[0] != filter_demand_mat.shape[0] or \
        filter_demand_mat.shape[0] != ofmap_demand_mat.shape[0]:
            raise ValueError("Demand matrices must have same number of cycles")
        
        # 3. Initialize tracking variables
        num_cycles = ofmap_demand_mat.shape[0]
        ifmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
        filter_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
        ofmap_serviced_cycles = np.zeros(num_cycles, dtype=np.uint32)
        stall_cycles = np.zeros(3, dtype=np.uint32)  # [ifmap, filter, ofmap]
        
        # 4. Get buffer latencies
        ifmap_hit_latency = self.ifmap_buf.hit_latency
        filter_hit_latency = self.filter_buf.hit_latency
        
        # 5. Process each cycle's demands with progress tracking
        for cycle in tqdm(range(num_cycles), unit="cycle"):
            # Reshape demand vectors to 2D arrays with single row
            ifmap_demand = ifmap_demand_mat[cycle].reshape(1, -1)
            filter_demand = filter_demand_mat[cycle].reshape(1, -1)
            ofmap_demand = ofmap_demand_mat[cycle].reshape(1, -1)
            
            # Create properly shaped cycle arrays (2D with single element)
            cycle_arr = np.array([[cycle]])
            
            # Service requests with properly shaped inputs
            ifmap_cycles = self.ifmap_buf.service_reads(
                incoming_requests_arr_np=ifmap_demand,
                incoming_cycles_arr=cycle_arr
            )
            filter_cycles = self.filter_buf.service_reads(
                incoming_requests_arr_np=filter_demand,
                incoming_cycles_arr=cycle_arr
            )
            ofmap_cycles = self.ofmap_buf.service_writes(
                incoming_requests_arr_np=ofmap_demand,
                incoming_cycles_arr_np=cycle_arr
            )
            
            # Extract completion cycles (handling both 1D and 2D returns)
            ifmap_cycle = ifmap_cycles[0][0] if ifmap_cycles.ndim == 2 else ifmap_cycles[0]
            filter_cycle = filter_cycles[0][0] if filter_cycles.ndim == 2 else filter_cycles[0]
            ofmap_cycle = ofmap_cycles[0][0] if ofmap_cycles.ndim == 2 else ofmap_cycles[0]
            
            # Calculate stalls
            stall_cycles[0] += max(0, ifmap_cycle - cycle - ifmap_hit_latency)
            stall_cycles[1] += max(0, filter_cycle - cycle - filter_hit_latency)
            stall_cycles[2] += max(0, ofmap_cycle - cycle)
            
            # Store results
            ifmap_serviced_cycles[cycle] = ifmap_cycle
            filter_serviced_cycles[cycle] = filter_cycle
            ofmap_serviced_cycles[cycle] = ofmap_cycle
        
        # 6. Complete prefetches if in estimation mode
        if self.estimate_bandwidth_mode:
            self.ifmap_buf.complete_all_prefetches()
            self.filter_buf.complete_all_prefetches()
        
        # 7. Empty OFMAP buffer
        self.ofmap_buf.empty_all_buffers(ofmap_serviced_cycles[-1])
        
        # 8. Prepare trace matrices
        self.ifmap_trace_matrix = np.column_stack((
            ifmap_serviced_cycles, 
            ifmap_demand_mat
        ))
        self.filter_trace_matrix = np.column_stack((
            filter_serviced_cycles,
            filter_demand_mat
        ))
        self.ofmap_trace_matrix = np.column_stack((
            ofmap_serviced_cycles,
            ofmap_demand_mat
        ))
        
        # 9. Update performance metrics
        self.total_cycles = int(np.max(ofmap_serviced_cycles)) + 1
        self.compute_cycles = num_cycles
        self.stall_cycles = int(np.sum(stall_cycles))
        
        # 10. Mark traces valid
        self.traces_valid = True


    # def service_memory_requests(self,
    #                         ifmap_demand_mat: np.ndarray,
    #                         filter_demand_mat: np.ndarray,
    #                         ofmap_demand_mat: np.ndarray) -> None:
    #     """
    #     Services memory requests for IFMAPs, filters, and OFMAPs.
    #     Handles read/write operations with cycle-accurate tracking.
    #     """
    #     # --- Parameter Validation ---
    #     assert self.params_valid_flag, "Parameters not validated. Call set_params() first"
        
    #     # --- Initialization ---
    #     total_cycles = 0
    #     stall_cycles = 0
    #     ifmap_serviced_cycles = np.zeros((ifmap_demand_mat.shape[0], 1), dtype=np.uint32)
    #     filter_serviced_cycles = np.zeros((filter_demand_mat.shape[0], 1), dtype=np.uint32)
    #     ofmap_serviced_cycles = np.zeros((ofmap_demand_mat.shape[0], 1), dtype=np.uint32)
        
    #     # --- Get Buffer Latencies ---
    #     ifmap_hit_latency = self.ifmap_buf.hit_latency
    #     filter_hit_latency = self.filter_buf.hit_latency
        
    #     # --- Process Each Demand Line ---
    #     for i in range(ofmap_demand_mat.shape[0]):
    #         # Reshape demand lines
    #         ifmap_demand = ifmap_demand_mat[i].reshape(-1, 1)
    #         filter_demand = filter_demand_mat[i].reshape(-1, 1)
    #         ofmap_demand = ofmap_demand_mat[i].reshape(-1, 1)
            
    #         # Service requests
    #         ifmap_cycles = self.ifmap_buf.service_reads(
    #             incoming_requests_arr_np=ifmap_demand,
    #             incoming_cycles_arr=np.array([i], dtype=np.uint32)
    #         )
    #         filter_cycles = self.filter_buf.service_reads(
    #             incoming_requests_arr_np=filter_demand,
    #             incoming_cycles_arr=np.array([i], dtype=np.uint32)
    #         )
    #         ofmap_cycles = self.ofmap_buf.service_writes(
    #             incoming_requests_arr_np=ofmap_demand,
    #             incoming_cycles_arr=np.array([i], dtype=np.uint32)
    #         )
            
    #         # Track cycles
    #         ifmap_serviced_cycles[i] = ifmap_cycles[0]
    #         filter_serviced_cycles[i] = filter_cycles[0]
    #         ofmap_serviced_cycles[i] = ofmap_cycles[0]
            
    #         # Calculate stalls
    #         stall_cycles += max(0, ifmap_cycles[0] - i - ifmap_hit_latency)
    #         stall_cycles += max(0, filter_cycles[0] - i - filter_hit_latency)
        
    #     # --- Bandwidth Estimation Mode ---
    #     if self.estimate_bandwidth_mode:
    #         self.ifmap_buf.complete_all_prefetches()
    #         self.filter_buf.complete_all_prefetches()
        
    #     # --- Empty Buffers ---
    #     self.ifmap_buf.empty_buffers()
    #     self.filter_buf.empty_buffers()
    #     self.ofmap_buf.empty_buffers()
        
    #     # --- Prepare Trace Matrices ---
    #     self.ifmap_trace_matrix = np.hstack((ifmap_serviced_cycles, ifmap_demand_mat))
    #     self.filter_trace_matrix = np.hstack((filter_serviced_cycles, filter_demand_mat))
    #     self.ofmap_trace_matrix = np.hstack((ofmap_serviced_cycles, ofmap_demand_mat))
        
    #     # --- Calculate Total Cycles ---
    #     self.total_cycles = int(ofmap_serviced_cycles[-1][0]) + 1
    #     self.stall_cycles = stall_cycles
    #     self.compute_cycles = self.total_cycles - stall_cycles
        
    #     # --- Final Validation ---
    #     self.traces_valid = True


        
        # # Print summary if verbose
        # if self.verbose:
        #     print(f"\nMemory request processing complete")
        #     print(f"Total cycles: {self.total_cycles}")
        #     print(f"Compute cycles: {self.compute_cycles}")
        #     print(f"Stall cycles: {self.stall_cycles}")
        #     print(f"IFMAP stalls: {stall_cycles[0]}")
        #     print(f"Filter stalls: {stall_cycles[1]}")
        #     print(f"OFMAP stalls: {stall_cycles[2]}")
        
        # if self.verbose:
        #     print(f"[Memory] Completed {num_cycles} cycles with {self.stall_cycles} stalls")

    # This is the trace computation logic of this memory system
    # Anand: This is too complex, perform the serve cycle by cycle for the requests
    # def service_memory_requests_old(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
    #     # TODO: assert sanity check
    #     assert self.params_valid_flag, 'Memories not initialized yet'

    #     # Logic:
    #     # Stalls can occur in both read and write portions and interfere with each other
    #     # We mitigate interference by picking a window in which there are no write stall,
    #     # ie, there is sufficient free space in the write buffer

    #     ofmap_lines_remaining = ofmap_demand_mat.shape[0]       # The three demand mats have the same shape though
    #     start_line_idx = 0
    #     end_line_idx = 0

    #     first = True
    #     cycle_offset = 0
    #     self.total_cycles = 0
    #     self.stall_cycles = 0

    #     # Status bar
    #     pbar_disable = not self.verbose #or True
    #     pbar = tqdm(total=ofmap_lines_remaining, disable=pbar_disable)

    #     avg_read_time_series = []

    #     while ofmap_lines_remaining > 0:
    #         loop_start_time = time.time()
    #         ofmap_free_space = self.ofmap_buf.get_free_space()

    #         # Find the number of lines till the ofmap_free_space is filled up
    #         count = 0
    #         while not count > ofmap_free_space:
    #             this_line = ofmap_demand_mat[end_line_idx]
    #             for elem in this_line:
    #                 if not elem == -1:
    #                     count += 1

    #             if not count > ofmap_free_space:
    #                 end_line_idx += 1
    #                 # Limit check
    #                 if not end_line_idx < ofmap_demand_mat.shape[0]:
    #                     end_line_idx = ofmap_demand_mat.shape[0] - 1
    #                     count = ofmap_free_space + 1
    #             else:   # Send request with minimal data ie one line of the requests
    #                 end_line_idx += 1
    #         # END of line counting

    #         num_lines = end_line_idx - start_line_idx + 1
    #         this_req_cycles_arr = [int(x + cycle_offset) for x in range(num_lines)]
    #         this_req_cycles_arr_np = np.asarray(this_req_cycles_arr).reshape((num_lines,1))

    #         this_req_ifmap_demands = ifmap_demand_mat[start_line_idx:(end_line_idx + 1), :]
    #         this_req_filter_demands = filter_demand_mat[start_line_idx:(end_line_idx + 1), :]
    #         this_req_ofmap_demands = ofmap_demand_mat[start_line_idx:(end_line_idx + 1), :]

    #         no_stall_cycles = num_lines     # Since the cycles are consecutive at this point

    #         time_start = time.time()
    #         ifmap_cycles_out = self.ifmap_buf.service_reads(incoming_requests_arr_np=this_req_ifmap_demands,
    #                                                         incoming_cycles_arr=this_req_cycles_arr_np)
    #         time_end = time.time()
    #         delta = time_end - time_start
    #         avg_read_time_series.append(delta)

    #         # Take care of the incurred stalls when launching demands for filter_reads
    #         # Note: Stalls incurred on reading line i in ifmap reflect the request cycles for line i+1 in filter
    #         ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
    #         ifmap_stalls = ifmap_cycles_out - this_req_cycles_arr_np - ifmap_hit_latency    # Vec - vec - scalar
    #         ifmap_stalls = np.concatenate((np.zeros((1,1)), ifmap_stalls[0:-1]), axis=0)    # Shift by one row
    #         this_req_cycles_arr_np = this_req_cycles_arr_np + ifmap_stalls

    #         time_start = time.time()
    #         filter_cycles_out = self.filter_buf.service_reads(incoming_requests_arr_np=this_req_filter_demands,
    #                                                           incoming_cycles_arr=this_req_cycles_arr_np)
    #         time_end = time.time()
    #         delta = time_end - time_start
    #         avg_read_time_series.append(delta)

    #         # Take care of stalls again --> The entire array stops when there is a stall
    #         filter_hit_latency = self.filter_buf.get_hit_latency()
    #         filter_stalls = filter_cycles_out - this_req_cycles_arr_np - filter_hit_latency  # Vec - vec - scalar
    #         filter_stalls = np.concatenate((np.zeros((1, 1)), filter_stalls[0:-1]), axis=0)  # Shift by one row
    #         this_req_cycles_arr_np = this_req_cycles_arr_np + filter_stalls

    #         ofmap_cycles_out = self.ofmap_buf.service_writes(incoming_requests_arr_np=this_req_ofmap_demands,
    #                                                          incoming_cycles_arr_np=this_req_cycles_arr_np)

    #         # Make the trace matrices
    #         this_req_ifmap_trace_matrix = np.concatenate((ifmap_cycles_out, this_req_ifmap_demands), axis=1)
    #         this_req_filter_trace_matrix = np.concatenate((filter_cycles_out, this_req_filter_demands), axis=1)
    #         this_req_ofmap_trace_matrix = np.concatenate((ofmap_cycles_out, this_req_ofmap_demands), axis=1)

    #         actual_cycles = ofmap_cycles_out[-1][0] - this_req_cycles_arr_np[0][0] + 1
    #         num_stalls = actual_cycles - no_stall_cycles

    #         self.stall_cycles += num_stalls
    #         self.total_cycles = ofmap_cycles_out[-1][0] + 1         # OFMAP is served the last

    #         if first:
    #             first = False
    #             self.ifmap_trace_matrix = this_req_ifmap_trace_matrix
    #             self.filter_trace_matrix = this_req_filter_trace_matrix
    #             self.ofmap_trace_matrix = this_req_ofmap_trace_matrix
    #         else:
    #             self.ifmap_trace_matrix = np.concatenate((self.ifmap_trace_matrix, this_req_ifmap_trace_matrix), axis=0)
    #             self.filter_trace_matrix = np.concatenate((self.filter_trace_matrix, this_req_filter_trace_matrix), axis=0)
    #             self.ofmap_trace_matrix = np.concatenate((self.ofmap_trace_matrix, this_req_ofmap_trace_matrix), axis=0)

    #         # Update the local variable for another iteration of the while loop
    #         cycle_offset = ofmap_cycles_out[-1][0] + 1
    #         start_line_idx = end_line_idx + 1

    #         pbar.update(num_lines)
    #         ofmap_lines_remaining = max(ofmap_demand_mat.shape[0] - (end_line_idx + 1), 0)    # Cutoff at 0
    #         #print("DEBUG: " + str(end_line_idx))

    #         if end_line_idx > ofmap_demand_mat.shape[0]:
    #             print('Trap')

    #         #if int(ofmap_lines_remaining % 1000) == 0:
    #         #    print("DEBUG: " + str(ofmap_lines_remaining))

    #         loop_end_time = time.time()
    #         loop_time = loop_end_time - loop_start_time
    #         #print('DEBUG: Time taken in one iteration: ' + str(loop_time))

    #     # At this stage there might still be some data in the active buffer of the OFMAP scratchpad
    #     # The following drains it and generates the OFMAP
    #     drain_start_cycle = self.ofmap_trace_matrix[-1][0] + 1
    #     self.ofmap_buf.empty_all_buffers(drain_start_cycle)

    #     #avg_read_time = sum(avg_read_time_series) / len(avg_read_time_series)
    #     #print('DEBUG: Avg time to service reads= ' + str(avg_read_time))

    #     pbar.close()
    #     # END of serving demands from memory
    #     self.traces_valid = True

    #
    # def get_total_compute_cycles(self):
    #     assert self.traces_valid, 'Traces not generated yet'
    #     return self.total_cycles

    # #
    # def get_stall_cycles(self):
    #     assert self.traces_valid, 'Traces not generated yet'
    #     return self.stall_cycles


    # ICL faild
    # def service_memory_requests(self, 
    #                         ifmap_demand_mat: np.ndarray,
    #                         filter_demand_mat: np.ndarray,
    #                         ofmap_demand_mat: np.ndarray) -> None:
    #     """Service memory requests for ifmap, filter and ofmap buffers.
        
    #     Args:
    #         ifmap_demand_mat: Demand matrix for input feature maps
    #         filter_demand_mat: Demand matrix for filters
    #         ofmap_demand_mat: Demand matrix for output feature maps
            
    #     Updates internal state including trace matrices and cycle counts.
    #     """
    #     # --- Parameter Validation ---
    #     assert self.params_valid_flag, "Parameters not validated"
    #     assert ifmap_demand_mat.shape[0] == ofmap_demand_mat.shape[0], "IFMAP-OFMAP row mismatch"
    #     assert filter_demand_mat.shape[0] == ofmap_demand_mat.shape[0], "Filter-OFMAP row mismatch"
        
    #     # --- Initialization ---
    #     total_cycles = 0
    #     ifmap_stall_cycles = 0
    #     filter_stall_cycles = 0
    #     ofmap_stall_cycles = 0
        
    #     # Get hit latencies
    #     ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
    #     filter_hit_latency = self.filter_buf.get_hit_latency()
        
    #     # Initialize trace matrices
    #     ifmap_serviced_cycles = np.zeros(ifmap_demand_mat.shape[0])
    #     filter_serviced_cycles = np.zeros(filter_demand_mat.shape[0])
    #     ofmap_serviced_cycles = np.zeros(ofmap_demand_mat.shape[0])
        
    #     # --- Process Each Demand Line ---
    #     for i in range(ofmap_demand_mat.shape[0]):
    #         # Reshape demand lines
    #         ifmap_demand_line = ifmap_demand_mat[i].reshape(-1, self.ifmap_buf.bandwidth)
    #         filter_demand_line = filter_demand_mat[i].reshape(-1, self.filter_buf.bandwidth)
    #         ofmap_demand_line = ofmap_demand_mat[i].reshape(-1, self.ofmap_buf.bandwidth)
            
    #         # Service read requests
    #         ifmap_cycles = self.ifmap_buf.service_reads(ifmap_demand_line, 
    #                                                 np.full(ifmap_demand_line.shape[0], total_cycles))
    #         filter_cycles = self.filter_buf.service_reads(filter_demand_line,
    #                                                 np.full(filter_demand_line.shape[0], total_cycles))
            
    #         # Service write requests
    #         ofmap_cycles = self.ofmap_buf.service_writes(ofmap_demand_line,
    #                                                 np.full(ofmap_demand_line.shape[0], total_cycles))
            
    #         # Update stall cycles
    #         ifmap_stall = np.max(ifmap_cycles) - (total_cycles + ifmap_hit_latency)
    #         filter_stall = np.max(filter_cycles) - (total_cycles + filter_hit_latency)
    #         ofmap_stall = np.max(ofmap_cycles) - total_cycles
            
    #         ifmap_stall_cycles += max(0, ifmap_stall)
    #         filter_stall_cycles += max(0, filter_stall)
    #         ofmap_stall_cycles += max(0, ofmap_stall)
            
    #         # Update total cycles
    #         total_cycles = max(np.max(ifmap_cycles), np.max(filter_cycles), np.max(ofmap_cycles))
            
    #         # Store serviced cycles
    #         ifmap_serviced_cycles[i] = np.max(ifmap_cycles)
    #         filter_serviced_cycles[i] = np.max(filter_cycles)
    #         ofmap_serviced_cycles[i] = np.max(ofmap_cycles)
        
    #     # --- Bandwidth Estimation Mode ---
    #     if self.estimate_bandwidth_mode:
    #         self.ifmap_buf.complete_all_prefetches(total_cycles)
    #         self.filter_buf.complete_all_prefetches(total_cycles)
        
    #     # --- Empty Buffers ---
    #     self.ofmap_buf.empty_all_buffers(total_cycles)
        
    #     # --- Prepare Trace Matrices ---
    #     self.ifmap_trace_mat = np.concatenate((ifmap_demand_mat, 
    #                                         ifmap_serviced_cycles.reshape(-1,1)), axis=1)
    #     self.filter_trace_mat = np.concatenate((filter_demand_mat,
    #                                         filter_serviced_cycles.reshape(-1,1)), axis=1)
    #     self.ofmap_trace_mat = np.concatenate((ofmap_demand_mat,
    #                                         ofmap_serviced_cycles.reshape(-1,1)), axis=1)
        
    #     # --- Update Total Cycles ---
    #     self.total_cycles = total_cycles
    #     self.ifmap_stall_cycles = ifmap_stall_cycles
    #     self.filter_stall_cycles = filter_stall_cycles
    #     self.ofmap_stall_cycles = ofmap_stall_cycles
        
    #     # --- Mark Traces as Valid ---
    #     self.traces_valid = True

    def get_total_compute_cycles(self) -> int:
        """
        Retrieves the total number of compute cycles from the simulation.
        
        Returns:
            int: The total compute cycles recorded during memory request processing
            
        Raises:
            RuntimeError: If traces are not yet valid (simulation not run or failed)
        """
        if not self.traces_valid:
            raise RuntimeError("Cannot get compute cycles - traces not valid. Run simulation first.")
        return self.total_cycles

    def get_stall_cycles(self) -> int:
        """
        Retrieves the total number of stall cycles from the simulation.
        
        Returns:
            int: The total stall cycles accumulated during memory request processing
            
        Raises:
            RuntimeError: If traces are not yet valid (simulation not run or failed)
        """
        if not self.traces_valid:
            raise RuntimeError("Cannot get stall cycles - traces not valid. Run simulation first.")
        return self.stall_cycles

    #
    # def get_ifmap_sram_start_stop_cycles(self):
    #     assert self.traces_valid, 'Traces not generated yet'

    #     done = False
    #     for ridx in range(self.ifmap_trace_matrix.shape[0]):
    #         if done:
    #             break
    #         row = self.ifmap_trace_matrix[ridx,1:]
    #         for addr in row:
    #             if not addr == -1:
    #                 self.ifmap_sram_start_cycle = self.ifmap_trace_matrix[ridx][0]
    #                 done = True
    #                 break

    #     done = False
    #     for ridx in range(self.ifmap_trace_matrix.shape[0]):
    #         if done:
    #             break
    #         ridx = -1 * (ridx + 1)
    #         row = self.ifmap_trace_matrix[ridx,1:]
    #         for addr in row:
    #             if not addr == -1:
    #                 self.ifmap_sram_stop_cycle  = self.ifmap_trace_matrix[ridx][0]
    #                 done = True
    #                 break

    #     return self.ifmap_sram_start_cycle, self.ifmap_sram_stop_cycle

    def get_ifmap_sram_start_stop_cycles(self) -> Tuple[int, int]:
        """
        Retrieves the start and stop cycles for IFMAP SRAM access from trace data.
        
        Returns:
            Tuple[int, int]: (start_cycle, stop_cycle) for IFMAP SRAM access
            
        Raises:
            RuntimeError: If traces are not valid or IFMAP trace matrix is empty
            ValueError: If no valid IFMAP accesses are found
            
        Operations:
            1. Validates trace data availability
            2. Finds first non-empty access cycle (start)
            3. Finds last non-empty access cycle (stop)
            4. Updates and returns cycle boundaries
        """
        # 1. Validate traces
        if not self.traces_valid:
            raise RuntimeError("IFMAP SRAM cycles unavailable - traces not valid")
        if self.ifmap_trace_matrix.size == 0:
            raise RuntimeError("IFMAP trace matrix is empty")
        
        # 2. Find start cycle (first non-empty access)
        start_cycle = None
        for row in self.ifmap_trace_matrix:
            cycle = row[0]
            # Check if any address in this cycle is valid (not -1)
            if np.any(row[1:] != -1):
                start_cycle = int(cycle)
                break
        
        if start_cycle is None:
            raise ValueError("No valid IFMAP accesses found in trace")
        
        # 3. Find stop cycle (last non-empty access)
        stop_cycle = None
        for row in reversed(self.ifmap_trace_matrix):
            cycle = row[0]
            if np.any(row[1:] != -1):
                stop_cycle = int(cycle)
                break
        
        # 4. Update instance variables and return
        self.ifmap_sram_start_cycle = start_cycle
        self.ifmap_sram_stop_cycle = stop_cycle
        
        return (start_cycle, stop_cycle)

    #
    # def get_filter_sram_start_stop_cycles(self):
    #     assert self.traces_valid, 'Traces not generated yet'

    #     done = False
    #     for ridx in range(self.filter_trace_matrix.shape[0]):
    #         if done:
    #             break
    #         row = self.filter_trace_matrix[ridx, 1:]
    #         for addr in row:
    #             if not addr == -1:
    #                 self.filter_sram_start_cycle = self.filter_trace_matrix[ridx][0]
    #                 done = True
    #                 break

    #     done = False
    #     for ridx in range(self.filter_trace_matrix.shape[0]):
    #         if done:
    #             break
    #         ridx = -1 * (ridx + 1)
    #         row = self.filter_trace_matrix[ridx, 1:]
    #         for addr in row:
    #             if not addr == -1:
    #                 self.filter_sram_stop_cycle = self.filter_trace_matrix[ridx][0]
    #                 done = True
    #                 break

    #     return self.filter_sram_start_cycle, self.filter_sram_stop_cycle

    def get_filter_sram_start_stop_cycles(self) -> Tuple[int, int]:
        """
        Retrieves the start and stop cycles for filter SRAM access periods from trace data.
        
        Returns:
            Tuple[int, int]: (start_cycle, stop_cycle) representing the active period 
                            of filter SRAM accesses
            
        Raises:
            RuntimeError: If trace data is not valid or available
            ValueError: If no valid filter accesses are found in traces
            
        Note:
            - Uses filter_trace_matrix which contains [cycle, addr1, addr2, ...]
            - -1 values indicate empty/invalid addresses
            - Updates filter_sram_start_cycle and filter_sram_stop_cycle instance variables
        """
        # 1. Validate trace availability
        if not self.traces_valid:
            raise RuntimeError("Filter SRAM cycles unavailable - traces not validated")
        if not hasattr(self, 'filter_trace_matrix') or self.filter_trace_matrix.size == 0:
            raise RuntimeError("Filter trace matrix not initialized or empty")

        # 2. Find first active cycle (start)
        start_found = False
        for row in self.filter_trace_matrix:
            if np.any(row[1:] != -1):  # Check address columns
                start_cycle = int(row[0])
                start_found = True
                break
        
        if not start_found:
            raise ValueError("No valid filter accesses found in trace matrix")

        # 3. Find last active cycle (stop)
        stop_found = False
        for row in reversed(self.filter_trace_matrix):  # Search backwards
            if np.any(row[1:] != -1):
                stop_cycle = int(row[0])
                stop_found = True
                break

        # 4. Update instance state and return
        self.filter_sram_start_cycle = start_cycle
        self.filter_sram_stop_cycle = stop_cycle
        
        return (start_cycle, stop_cycle)

    #
    # def get_ofmap_sram_start_stop_cycles(self):
    #     assert self.traces_valid, 'Traces not generated yet'

    #     done = False
    #     for ridx in range(self.ofmap_trace_matrix.shape[0]):
    #         if done:
    #             break
    #         row = self.ofmap_trace_matrix[ridx, 1:]
    #         for addr in row:
    #             if not addr == -1:
    #                 self.ofmap_sram_start_cycle = self.ofmap_trace_matrix[ridx][0]
    #                 done = True
    #                 break

    #     done = False
    #     for ridx in range(self.ofmap_trace_matrix.shape[0]):
    #         if done:
    #             break
    #         ridx = -1 * (ridx + 1)
    #         row = self.ofmap_trace_matrix[ridx, 1:]
    #         for addr in row:
    #             if not addr == -1:
    #                 self.ofmap_sram_stop_cycle = self.ofmap_trace_matrix[ridx][0]
    #                 done = True
    #                 break

    #     return self.ofmap_sram_start_cycle, self.ofmap_sram_stop_cycle

    def get_ofmap_sram_start_stop_cycles(self) -> Tuple[int, int]:
        """
        Retrieves the active period boundaries for OFMAP SRAM operations from trace data.
        
        Returns:
            Tuple[int, int]: (start_cycle, stop_cycle) representing the SRAM access window
            
        Raises:
            RuntimeError: If traces are invalid or OFMAP matrix not available
            ValueError: If no valid OFMAP SRAM accesses are found
            
        Note:
            - Traces must be generated before calling (traces_valid=True)
            - Updates ofmap_sram_start_cycle and ofmap_sram_stop_cycle
            - Returns (0, 0) if no accesses found (when configured to not raise)
            - First column is cycle count, subsequent columns are addresses
        """
        # 1. System state validation
        if not self.traces_valid:
            raise RuntimeError("OFMAP SRAM cycles unavailable - traces not validated")
        if not hasattr(self, 'ofmap_trace_matrix') or self.ofmap_trace_matrix.size == 0:
            raise RuntimeError("OFMAP trace matrix not initialized")

        # 2. Initialize tracking variables
        start_cycle = None
        stop_cycle = None
        matrix = self.ofmap_trace_matrix

        # 3. Find first valid access (start cycle)
        for row in matrix:
            cycle = row[0]
            if np.any(row[1:] != -1):  # Check address columns
                start_cycle = int(cycle)
                break

        # 4. Find last valid access (stop cycle)
        for row in reversed(matrix):
            cycle = row[0]
            if np.any(row[1:] != -1):
                stop_cycle = int(cycle)
                break

        # 5. Validate results
        if start_cycle is None or stop_cycle is None:
            raise ValueError("No valid OFMAP SRAM accesses detected in traces")
        if stop_cycle < start_cycle:
            raise ValueError(f"Invalid cycle range: start={start_cycle}, stop={stop_cycle}")

        # 6. Update instance state
        self.ofmap_sram_start_cycle = start_cycle
        self.ofmap_sram_stop_cycle = stop_cycle
        
        return (start_cycle, stop_cycle)

    def get_ofmap_sram_start_stop_cycles(self) -> Tuple[int, int]:
        """
        Retrieves the active period boundaries for OFMAP SRAM operations from trace data.
        
        Returns:
            Tuple[int, int]: (start_cycle, stop_cycle) representing the active access window
            
        Raises:
            RuntimeError: If traces are invalid or OFMAP matrix unavailable
            ValueError: If no valid OFMAP writes are detected
            
        Note:
            - Requires successful completion of memory simulation (traces_valid=True)
            - Updates ofmap_sram_start_cycle and ofmap_sram_stop_cycle instance variables
            - Returns cycle values as integers
        """
        # Input validation and state checking
        if not self.traces_valid:
            raise RuntimeError("OFMAP access cycles unavailable - traces not validated")
        if not hasattr(self, 'ofmap_trace_matrix') or self.ofmap_trace_matrix.size == 0:
            raise RuntimeError("OFMAP trace matrix not initialized")

        # Find first valid access (start cycle)
        start_cycle = None
        for row in self.ofmap_trace_matrix:
            if np.any(row[1:] != -1):  # Check address columns
                start_cycle = int(row[0])
                break

        if start_cycle is None:
            raise ValueError("No valid OFMAP writes detected in trace data")

        # Find last valid access (stop cycle)
        stop_cycle = None
        for row in reversed(self.ofmap_trace_matrix):
            if np.any(row[1:] != -1):
                stop_cycle = int(row[0])
                break

        # Update instance state and return results
        self.ofmap_sram_start_cycle = start_cycle
        self.ofmap_sram_stop_cycle = stop_cycle
        return (start_cycle, stop_cycle)


    #
    # def get_ifmap_dram_details(self):
    #     assert self.traces_valid, 'Traces not generated yet'

    #     self.ifmap_dram_reads = self.ifmap_buf.get_num_accesses()
    #     self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle \
    #         = self.ifmap_buf.get_external_access_start_stop_cycles()

    #     return self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle, self.ifmap_dram_reads

    def get_ifmap_dram_details(self) -> Tuple[int, int, int]:
        """
        Retrieves comprehensive IFMAP DRAM access metrics including timing and volume.
        
        Returns:
            Tuple[int, int, int]: (start_cycle, stop_cycle, total_reads) representing:
                - start_cycle: First DRAM access cycle
                - stop_cycle: Last DRAM access cycle  
                - total_reads: Cumulative DRAM read operations
                
        Raises:
            RuntimeError: If traces are invalid or IFMAP buffer not initialized
            ValueError: If no DRAM accesses were performed
            
        Note:
            - Updates instance variables ifmap_dram_start_cycle, ifmap_dram_stop_cycle, 
            and ifmap_dram_reads
            - Requires successful simulation completion (traces_valid=True)
        """
        # System state validation
        if not self.traces_valid:
            raise RuntimeError("IFMAP DRAM details unavailable - traces not validated")
        if not hasattr(self, 'ifmap_buf'):
            raise RuntimeError("IFMAP buffer not initialized")

        # Retrieve access metrics
        self.ifmap_dram_reads = self.ifmap_buf.get_num_accesses()
        start, stop = self.ifmap_buf.get_external_access_start_stop_cycles()
        
        # Validate DRAM activity
        if self.ifmap_dram_reads == 0:
            raise ValueError("No IFMAP DRAM reads were performed")
        if start is None or stop is None:
            raise ValueError("Invalid IFMAP DRAM access period")

        # Update instance state
        self.ifmap_dram_start_cycle = int(start)
        self.ifmap_dram_stop_cycle = int(stop)
        
        return (self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle, self.ifmap_dram_reads)

    #
    # def get_filter_dram_details(self):
    #     assert self.traces_valid, 'Traces not generated yet'

    #     self.filter_dram_reads = self.filter_buf.get_num_accesses()
    #     self.filter_dram_start_cycle, self.filter_dram_stop_cycle \
    #         = self.filter_buf.get_external_access_start_stop_cycles()

    #     return self.filter_dram_start_cycle, self.filter_dram_stop_cycle, self.filter_dram_reads

    def get_filter_dram_details(self) -> Tuple[int, int, int]:
        """
        Retrieves comprehensive filter DRAM access metrics including timing and volume.
        
        Returns:
            Tuple[int, int, int]: (start_cycle, stop_cycle, total_reads) representing:
                - start_cycle: First DRAM access cycle (inclusive)
                - stop_cycle: Last DRAM access cycle (inclusive)
                - total_reads: Total DRAM read operations
                
        Raises:
            RuntimeError: If traces are invalid or filter buffer not initialized
            ValueError: If no DRAM accesses were performed or cycle range is invalid
            
        Note:
            - Updates instance variables filter_dram_start_cycle, filter_dram_stop_cycle, 
            and filter_dram_reads
            - Requires successful simulation completion (traces_valid=True)
            - Guarantees stop_cycle >= start_cycle when reads > 0
        """
        # 1. System state validation
        if not self.traces_valid:
            raise RuntimeError("Filter DRAM details unavailable - traces not validated")
        if not hasattr(self, 'filter_buf'):
            raise RuntimeError("Filter buffer not initialized")

        # 2. Retrieve access metrics
        self.filter_dram_reads = self.filter_buf.get_num_accesses()
        start, stop = self.filter_buf.get_external_access_start_stop_cycles()
        
        # 3. Validate DRAM activity
        if self.filter_dram_reads == 0:
            raise ValueError("No filter DRAM reads were performed during simulation")
        if not (isinstance(start, (int, np.integer)) and isinstance(stop, (int, np.integer))):
            raise ValueError(f"Invalid cycle values: start={start}, stop={stop}")
        if stop < start:
            raise ValueError(f"Invalid cycle range: start={start} > stop={stop}")

        # 4. Update instance state with sanitized values
        self.filter_dram_start_cycle = int(start)
        self.filter_dram_stop_cycle = int(stop)
        
        return (self.filter_dram_start_cycle, self.filter_dram_stop_cycle, self.filter_dram_reads)

    #
    # def get_ofmap_dram_details(self):
    #     assert self.traces_valid, 'Traces not generated yet'

    #     self.ofmap_dram_writes = self.ofmap_buf.get_num_accesses()
    #     self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle \
    #         = self.ofmap_buf.get_external_access_start_stop_cycles()

    #     return self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle, self.ofmap_dram_writes

    def get_ofmap_dram_details(self) -> tuple:
        """
        Retrieves detailed DRAM access information for the output feature map (OFMAP) buffer.
        
        Returns:
            tuple: A 3-element tuple containing:
                - ofmap_dram_start_cycle (int): First cycle of DRAM access
                - ofmap_dram_stop_cycle (int): Last cycle of DRAM access  
                - ofmap_dram_writes (int): Total number of DRAM writes
        
        Raises:
            AssertionError: If traces have not been generated yet
        """
        # 1. Validate traces
        assert self.traces_valid, "Traces not generated yet"
        
        # 2. Retrieve number of writes
        self.ofmap_dram_writes = self.ofmap_buf.get_num_accesses()
        
        # 3. Retrieve start and stop cycles
        start_cycle, stop_cycle = self.ofmap_buf.get_external_access_start_stop_cycles()
        self.ofmap_dram_start_cycle = start_cycle
        self.ofmap_dram_stop_cycle = stop_cycle
        
        # 4. Return results
        return (self.ofmap_dram_start_cycle, 
                self.ofmap_dram_stop_cycle, 
                self.ofmap_dram_writes)

    #
    def get_ifmap_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix

    #
    def get_filter_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.filter_trace_matrix

    #
    def get_ofmap_sram_trace_matrix(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ofmap_trace_matrix

    #
    def get_sram_trace_matrices(self):
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix, self.filter_trace_matrix, self.ofmap_trace_matrix

    #
    def get_ifmap_dram_trace_matrix(self):
        return self.ifmap_buf.get_trace_matrix()

    #
    def get_filter_dram_trace_matrix(self):
        return self.filter_buf.get_trace_matrix()

    #
    def get_ofmap_dram_trace_matrix(self):
        return self.ofmap_buf.get_trace_matrix()

    #
    def get_dram_trace_matrices(self):
        dram_ifmap_trace = self.ifmap_buf.get_trace_matrix()
        dram_filter_trace = self.filter_buf.get_trace_matrix()
        dram_ofmap_trace = self.ofmap_buf.get_trace_matrix()

        return dram_ifmap_trace, dram_filter_trace, dram_ofmap_trace

        #
    def print_ifmap_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.ifmap_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_filter_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.filter_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_ofmap_sram_trace(self, filename):
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.ofmap_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_ifmap_dram_trace(self, filename):
        self.ifmap_buf.print_trace(filename)

    #
    def print_filter_dram_trace(self, filename):
        self.filter_buf.print_trace(filename)

    #
    def print_ofmap_dram_trace(self, filename):
        self.ofmap_buf.print_trace(filename)





