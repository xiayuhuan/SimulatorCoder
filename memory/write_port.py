# This is shell module to ensure continuity
import numpy as np


class write_port:
    def __init__(self):
        self.latency = 0

    # def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
    #     out_cycles_arr_np = incoming_cycles_arr_np + self.latency
    #     out_cycles_arr_np = out_cycles_arr_np.reshape((out_cycles_arr_np.shape[0], 1))
    #     return out_cycles_arr_np

    def service_writes(self,
                     incoming_requests_arr_np: np.ndarray,
                     incoming_cycles_arr_np: np.ndarray) -> np.ndarray:
        """Process incoming write requests and calculate completion cycles.
        
        Args:
            incoming_requests_arr_np: Array of incoming write requests (not used)
            incoming_cycles_arr_np: Array of arrival cycles for each request
            
        Returns:
            2D numpy array where each row contains the completion cycle for a request
            
        Note:
            The incoming_requests_arr_np parameter is not used but kept for interface consistency
        """
        # Calculate completion cycles by adding latency to arrival cycles
        completion_cycles = incoming_cycles_arr_np + self.latency
        
        # Reshape to 2D array (n_requests x 1)
        return completion_cycles.reshape(-1, 1)
