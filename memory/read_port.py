# Dummy memory like interface to service the requests of the last level memory

# class read_port:
#     def __init__(self):
#         self.latency = 1

#     def set_params(self, latency):
#         self.latency = latency

#     def get_latency(self):
#         return self.latency

#     # The incoming read requests will be needed when the capability of port is expanded
#     # At the moment its kept for compatibility
#     def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
#         out_cycles_arr = incoming_cycles_arr + self.latency
#         return out_cycles_arr

import numpy as np
from typing import Union, Optional, Any

class read_port:
    """A read port module for DNN accelerators with configurable latency.
    
    This module handles the timing characteristics of read operations in a memory system,
    calculating completion cycles based on arrival times and port latency.
    """

    def __init__(self) -> None:
        """Initialize the read port with default latency of 1 cycle."""
        self.latency = 1  # Default single-cycle latency

    def set_params(self, latency: int) -> None:
        """Configure the read port's timing parameters.
        
        Args:
            latency: The new latency value in cycles (must be positive integer)
            
        Raises:
            ValueError: If latency is not a positive integer
        """
        if not isinstance(latency, int) or latency <= 0:
            raise ValueError("Latency must be a positive integer")
        self.latency = latency

    def get_latency(self) -> int:
        """Query the current latency setting.
        
        Returns:
            The current latency value in cycles
        """
        return self.latency

    def service_reads(self,
                     incoming_requests_arr_np: Optional[np.ndarray] = None,
                     incoming_cycles_arr: Union[np.ndarray, list] = None) -> np.ndarray:
        """Process read requests and calculate completion cycles.
        
        Args:
            incoming_requests_arr_np: Unused parameter kept for interface compatibility
            incoming_cycles_arr: Array/list of arrival cycles for each read request
            
        Returns:
            NumPy array of completion cycles for each request
            
        Raises:
            ValueError: If input cycles array is None or invalid
        """
        if incoming_cycles_arr is None:
            raise ValueError("Input cycles array cannot be None")
            
        # Convert to numpy array if needed
        cycles = np.asarray(incoming_cycles_arr, dtype=np.int64)
        
        # Calculate completion cycles
        completion_cycles = cycles + self.latency
        
        return completion_cycles