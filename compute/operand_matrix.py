import math
import numpy as np
from tqdm import tqdm

from scalesim.topology_utils import topologies as topoutil
from scalesim.scale_config import scale_config as cfg


# This class defines data types for operand matrices
class operand_matrix(object):
    # def __init__(self):
    #     # Objects from outer container classes
    #     self.config = cfg()
    #     self.topoutil = topoutil()

    #     # Layer hyper parameters
    #     self.layer_id = 0
    #     self.ifmap_rows, self.ifmap_cols = 1, 1
    #     self.filter_rows, self.filter_cols = 1, 1
    #     self.num_input_channels, self.num_filters = 1, 1
    #     self.row_stride, self.col_stride = 1, 1
    #     self.batch_size = 1

    #     #  Derived hyper parameters
    #     self.ofmap_px_per_filt, self.conv_window_size = 1, 1
    #     self.ofmap_rows, self.ofmap_cols = 1, 1

    #     # Offsets
    #     self.ifmap_offset, self.filter_offset, self.ofmap_offset = 0, 10000000, 20000000
    #     self.matrix_offset_arr = [0, 10000000, 20000000]

    #     # Address matrices
    #     self.ifmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.conv_window_size), dtype=int)
    #     self.filter_addr_matrix = np.ones((self.conv_window_size, self.num_filters), dtype=int)
    #     self.ofmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.num_filters), dtype=int)

    #     # Flags
    #     self.params_set_flag = False
    #     self.matrices_ready_flag = False

    def __init__(self):
        """
        Initializes layer configuration with default values and necessary objects.
        
        Initializes:
        - Outer container class instances
        - Layer hyperparameters with default values
        - Derived hyperparameters
        - Memory offsets
        - Address matrices
        - Status flags
        """
        # 1. Initialize outer container instances
        self.config = cfg()
        self.topoutil = topoutil()

        # 2. Set default layer hyperparameters
        # Identification and dimensions
        self.layer_id = 0
        self.ifmap_rows = 1
        self.ifmap_cols = 1
        self.filter_rows = 1
        self.filter_cols = 1
        self.num_input_channels = 1
        self.num_filters = 1
        
        # Strides and batch
        self.row_stride = 1
        self.col_stride = 1
        self.batch_size = 1

        # 3. Initialize derived hyperparameters
        self.ofmap_px_per_filt = 1
        self.conv_window_size = 1
        self.ofmap_rows = 1
        self.ofmap_cols = 1

        # 4. Set memory offsets
        self.ifmap_offset = 0
        self.filter_offset = 10000000
        self.ofmap_offset = 20000000
        self.matrix_offset_arr = [0, 10000000, 20000000]

        # 5. Initialize address matrices
        # Using numpy for matrix operations
        import numpy as np
        self.ifmap_addr_matrix = np.ones((1, 1), dtype=int)
        self.filter_addr_matrix = np.ones((1, 1), dtype=int)
        self.ofmap_addr_matrix = np.ones((1, 1), dtype=int)

        # 6. Set status flags
        self.params_set_flag = False
        self.matrices_ready_flag = False

    #
    def set_params(self,
                   config_obj,
                   topoutil_obj,
                   layer_id=0,
                   ):

        self.config = config_obj
        self.topoutil = topoutil_obj
        self.layer_id = layer_id

        # TODO: Marked for cleanup
        #my_name = 'operand_matrix.set_params(): '
        #err_prefix = 'Error: ' + my_name
        #
        #if (not len(layer_hyper_param_arr) == 7 and not len(layer_hyper_param_arr) == 8
        #        and not len(layer_hyper_param_arr) == 9) or (not len(layer_calc_hyper_param_arr) == 4) \
        #        or (not len(self.matrix_offset_arr) == 3):
        #    message = err_prefix + 'Invalid arguments. Exiting.'
        #    print(message)
        #    return -1

        self.ifmap_rows, self.ifmap_cols = self.topoutil.get_layer_ifmap_dims(self.layer_id)
        self.filter_rows, self.filter_cols = self.topoutil.get_layer_filter_dims(self.layer_id)
        self.num_input_channels = self.topoutil.get_layer_num_channels(self.layer_id)
        self.num_filters = self.topoutil.get_layer_num_filters(self.layer_id)
        self.row_stride, self.col_stride = self.topoutil.get_layer_strides(self.layer_id)
        # TODO: Marked for cleanup
        #self.row_stride = layer_hyper_param_arr[6]
        #if len(layer_hyper_param_arr) == 8:
        #    self.col_stride = layer_hyper_param_arr[7]

        # TODO: Anand
        # TODO: Next release
        # TODO: Add an option for batching
        self.batch_size = 1

        # TODO: Marked for cleanup
        #if len(layer_hyper_param_arr) == 9:
        #    self.batch_size = layer_hyper_param_arr[8]

        # Assign the calculated hyper parameters
        self.ofmap_rows, self.ofmap_cols = self.topoutil.get_layer_ofmap_dims(self.layer_id)
        self.ofmap_rows = int(self.ofmap_rows)
        self.ofmap_cols = int(self.ofmap_cols)
        self.ofmap_px_per_filt = int(self.ofmap_rows * self.ofmap_cols)
        self.conv_window_size = int(self.topoutil.get_layer_window_size(self.layer_id))

        # Assign the offsets
        self.ifmap_offset, self.filter_offset, self.ofmap_offset \
            = self.config.get_offsets()

        # Address matrices: This is needed to take into account the updated dimensions
        self.ifmap_addr_matrix = np.ones((self.ofmap_px_per_filt * self.batch_size, self.conv_window_size), dtype='>i4')
        self.filter_addr_matrix = np.ones((self.conv_window_size, self.num_filters), dtype='>i4')
        self.ofmap_addr_matrix = np.ones((self.ofmap_px_per_filt, self.num_filters), dtype='>i4')
        self.params_set_flag = True

        # TODO: This should be called from top level
        # TODO: Implement get() function for getting the matrix
        # TODO: Marked for cleanup
        # Return 0 if operand matrix generation is successful
        #self.create_operand_matrices()
        #if self.matrices_ready_flag:
        #    return True, self.ifmap_addr_matrix, self.filter_addr_matrix, self.ofmap_addr_matrix
        #else:
        #    message = err_prefix + 'Address Matrices not created. Exiting!'
        #    print(message)
        #    return False, None, None, None

    # top level function to create the operand matrices
    # def create_operand_matrices(self):
    #     my_name = 'operand_matrix.create_operand_matrices(): '
    #     err_prefix = 'Error: ' + my_name

    #     if not self.params_set_flag:
    #         message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
    #         print(message)
    #         return -1

    #     retcode_1 = self.create_ifmap_matrix()
    #     retcode_2 = self.create_filter_matrix()
    #     retcode_3 = self.create_ofmap_matrix()

    #     retcode = retcode_1 + retcode_2 + retcode_3
    #     if retcode == 0:
    #         self.matrices_ready_flag = True

    #     return retcode

    def create_operand_matrices(self):
        """
        Creates the operand matrices (IFMAP, filter, and OFMAP) required for the layer configuration.

        This method coordinates the creation of all three operand matrices and checks their success status.
        If any matrix creation fails, the method returns an aggregated error code. On full success,
        it sets the matrices_ready_flag to True.

        Returns:
            int: -1 if parameters are not set or if any matrix creation fails, 0 on full success.
        """
        # Check if parameters are set
        if not self.params_set_flag:
            error_msg = (
                "Error: Parameters not set. "
                "Ensure layer parameters are configured before creating operand matrices."
            )
            print(error_msg)
            return -1

        # Initialize return codes for each matrix creation
        retcode_1 = self.create_ifmap_matrix()  # IFMAP matrix
        retcode_2 = self.create_filter_matrix()  # Filter matrix
        retcode_3 = self.create_ofmap_matrix()  # OFMAP matrix

        # Aggregate return codes (assumes 0 = success, -1 = failure)
        retcode = retcode_1 + retcode_2 + retcode_3

        # If all matrices were created successfully (retcode == 0), set flag
        if retcode == 0:
            self.matrices_ready_flag = True

        return retcode

    # creates the ifmap operand
    # def create_ifmap_matrix(self):
    #     my_name = 'operand_matrix.create_ifmap_matrix(): '
    #     err_prefix = 'Error: ' + my_name

    #     if not self.params_set_flag:
    #         message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
    #         print(message)
    #         return -1

    #     row_indices = np.arange(self.batch_size * self.ofmap_px_per_filt)
    #     col_indices = np.arange(self.conv_window_size)
    #     # Create 2D index arrays using meshgrid
    #     i, j = np.meshgrid(row_indices, col_indices, indexing='ij')

    #     # Call calc_ifmap_elem_addr_numpy with 2D index arrays
    #     self.ifmap_addr_matrix = self.calc_ifmap_elem_addr(i, j)
    #     return 0


# error
    # def create_ifmap_matrix(self) -> int:
    #     """
    #     Creates the IFMAP address matrix based on layer configuration.
        
    #     Returns:
    #         int: 0 on success, -1 if parameters not set
            
    #     Processing Flow:
    #         1. Check if parameters are set
    #         2. Generate OFMAP position indices
    #         3. Compute IFMAP addresses
    #         4. Store resulting matrix
    #     """
    #     # ===== BRANCH STRUCTURE =====
    #     if not self.params_set_flag:
    #         print("Error: Layer parameters not set. Call set_params() first.")
    #         return -1
        
    #     # ===== SEQUENTIAL STRUCTURE =====
    #     # Generate OFMAP position indices
    #     ofmap_rows = (self.ifmap_rows - self.filter_rows) // self.row_stride + 1
    #     ofmap_cols = (self.ifmap_cols - self.filter_cols) // self.col_stride + 1
        
    #     i = np.arange(ofmap_rows)
    #     j = np.arange(ofmap_cols)
        
    #     # Create 2D index grid
    #     i_grid, j_grid = np.meshgrid(i, j, indexing='ij')
        
    #     # ===== SEQUENTIAL STRUCTURE =====
    #     # Compute IFMAP addresses
    #     self.ifmap_addr_matrix = self.calc_ifmap_elem_addr(i_grid.ravel(), j_grid.ravel())
        
    #     # Reshape to match OFMAP dimensions
    #     self.ifmap_addr_matrix = self.ifmap_addr_matrix.reshape(
    #         ofmap_rows, ofmap_cols, 
    #         self.filter_rows * self.filter_cols * self.num_input_channels
    #     )
        
    #     return 0

    # ds
    def create_ifmap_matrix(self):
        """
        Creates the input feature map (IFMAP) address matrix based on the layer configuration.

        The method generates row and column indices for the IFMAP matrix, computes the
        corresponding addresses using `calc_ifmap_elem_addr`, and stores the result in
        `self.ifmap_addr_matrix`.

        Returns:
            int: Returns -1 if the parameters are not set (self.params_set_flag is False),
                otherwise returns None after successful matrix creation.
        """
        # Check if parameters are set
        if not self.params_set_flag:
            error_msg = (
                "Error: Parameters not set. "
                "Ensure layer parameters are configured before creating IFMAP matrix."
            )
            print(error_msg)
            return -1

        # Generate row indices (0 to batch_size * ofmap_px_per_filt - 1)
        rows = np.arange(self.batch_size * self.ofmap_px_per_filt)
        
        # Generate column indices (0 to conv_window_size - 1)
        cols = np.arange(self.conv_window_size)
        
        # Create 2D index arrays using meshgrid
        i, j = np.meshgrid(rows, cols, indexing='ij')
        
        # Compute IFMAP addresses using the 2D indices
        self.ifmap_addr_matrix = self.calc_ifmap_elem_addr(i, j)
        
        # return None errror
        return 0

    # ICL sucess
    def create_ifmap_matrix(self):
        """
        Generates the IFMAP address matrix for convolution operations.
        
        Returns:
            int: 0 on success, -1 if parameters aren't configured
        """
        # 1. Parameter validation (Branch Structure)
        if not self.params_set_flag:
            print("Error: Layer parameters not configured")
            return -1
        
        # 2. Calculate matrix dimensions (Sequential Structure)
        ofmap_size = self.ofmap_rows * self.ofmap_cols
        window_size = self.filter_rows * self.filter_cols * self.num_input_channels
        
        # 3. Create index arrays
        i = np.arange(ofmap_size)  # Output feature map indices
        j = np.arange(window_size) # Filter window indices
        
        # 4. Generate 2D index grid
        i_grid, j_grid = np.meshgrid(i, j, indexing='ij')
        
        # 5. Compute address matrix
        addr_matrix = self.calc_ifmap_elem_addr(i_grid.flatten(), j_grid.flatten())
        self.ifmap_addr_matrix = addr_matrix.reshape((ofmap_size, window_size))
        
        return 0


    # logic to translate ifmap into matrix fed into systolic array MACs
    # def calc_ifmap_elem_addr(self, i, j):
    #     offset = self.ifmap_offset
    #     ifmap_rows = self.ifmap_rows
    #     ifmap_cols = self.ifmap_cols
    #     filter_col = self.filter_cols
    #     r_stride = self.row_stride
    #     c_stride = self.col_stride
    #     Ew = self.ofmap_cols
    #     channel = self.num_input_channels

    #     ofmap_row, ofmap_col = np.divmod(i, Ew)
    #     i_row, i_col = ofmap_row * r_stride, ofmap_col * c_stride
    #     window_addr = (i_row * ifmap_cols + i_col) * channel

    #     c_row, k = np.divmod(j, filter_col * channel)
    #     c_col, c_ch = np.divmod(k, channel)

    #     valid_indices = np.logical_and(c_row + i_row < ifmap_rows, c_col + i_col < ifmap_cols)
    #     ifmap_px_addr = np.full(i.shape, -1)
    #     if valid_indices.any():
    #         internal_address = (c_row[valid_indices] * ifmap_cols + c_col[valid_indices]) * channel + c_ch[valid_indices]
    #         ifmap_px_addr[valid_indices] = internal_address + window_addr[valid_indices] + offset

    #     return ifmap_px_addr

# error
    # def calc_ifmap_elem_addr(self, i: int, j: int) -> list:
    #     """
    #     Calculates IFMAP element addresses for a given OFMAP position (i,j).
        
    #     Args:
    #         i: OFMAP row index
    #         j: OFMAP column index
            
    #     Returns:
    #         List of IFMAP element addresses (length = filter_rows * filter_cols * num_input_channels)
    #         Invalid accesses marked with -1
    #     """
    #     # 1. Retrieve layer configuration parameters
    #     ifmap_base = self.ifmap_offset
    #     filter_h = self.filter_rows
    #     filter_w = self.filter_cols
    #     channels = self.num_input_channels
    #     stride_h = self.row_stride
    #     stride_w = self.col_stride
        
    #     # 2. Calculate IFMAP window coordinates
    #     i_row = i * stride_h
    #     i_col = j * stride_w
        
    #     # 3. Initialize address array with -1 (invalid)
    #     ifmap_px_addr = [-1] * (filter_h * filter_w * channels)
        
    #     # 4. Calculate valid filter positions
    #     for c_row in range(filter_h):
    #         for c_col in range(filter_w):
    #             # Corresponding IFMAP coordinates
    #             ifmap_row = i_row + c_row
    #             ifmap_col = i_col + c_col
                
    #             # Check boundary conditions
    #             if (ifmap_row < self.ifmap_rows and 
    #                 ifmap_col < self.ifmap_cols):
                    
    #                 # 5. Calculate address for each channel
    #                 for c_ch in range(channels):
    #                     idx = c_ch * (filter_h * filter_w) + c_row * filter_w + c_col
    #                     ifmap_px_addr[idx] = (ifmap_base + 
    #                                         ifmap_row * self.ifmap_cols * channels +
    #                                         ifmap_col * channels +
    #                                         c_ch)
        
    #     return ifmap_px_addr

    # creates the ofmap operand
# error
    # def calc_ifmap_elem_addr(self, i: int, j: int) -> np.ndarray:
    #     """
    #     Calculates IFMAP element addresses for given OFMAP positions.
    #     Handles both scalar and array inputs correctly.
        
    #     Args:
    #         i: OFMAP row index (scalar or array)
    #         j: OFMAP column index (scalar or array)
            
    #     Returns:
    #         ndarray of IFMAP element addresses with shape:
    #         (filter_rows, filter_cols, num_input_channels)
    #         Invalid accesses marked with -1
    #     """
    #     # Convert inputs to numpy arrays for consistent handling
    #     i = np.asarray(i)
    #     j = np.asarray(j)
        
    #     # Get layer configuration
    #     ifmap_base = self.ifmap_offset
    #     filter_h = self.filter_rows
    #     filter_w = self.filter_cols
    #     channels = self.num_input_channels
    #     stride_h = self.row_stride
    #     stride_w = self.col_stride
        
    #     # Initialize output array
    #     output_shape = (filter_h, filter_w, channels)
    #     ifmap_px_addr = np.full(output_shape, -1, dtype=int)
        
    #     # Calculate IFMAP window coordinates
    #     i_row = i * stride_h
    #     i_col = j * stride_w
        
    #     # Generate all possible filter positions
    #     c_row, c_col = np.mgrid[:filter_h, :filter_w]
        
    #     # Calculate corresponding IFMAP coordinates
    #     ifmap_row = i_row[..., np.newaxis, np.newaxis] + c_row
    #     ifmap_col = i_col[..., np.newaxis, np.newaxis] + c_col
        
    #     # Create validity mask
    #     valid_mask = ((ifmap_row < self.ifmap_rows) & 
    #                 (ifmap_col < self.ifmap_cols))
        
    #     # Calculate addresses where valid
    #     if valid_mask.any():
    #         # Calculate base addresses for valid positions
    #         base_addrs = (ifmap_base +
    #                     ifmap_row * self.ifmap_cols * channels +
    #                     ifmap_col * channels)
            
    #         # Add channel offsets
    #         channel_offsets = np.arange(channels)
    #         ifmap_px_addr[valid_mask] = (
    #             base_addrs[valid_mask][..., np.newaxis] + channel_offsets
    #         ).reshape(-1)
        
    #     return ifmap_px_addr

    # def calc_ifmap_elem_addr(self, i, j):
    #     """
    #     Calculates IFMAP element addresses for given OFMAP positions.
    #     Handles both scalar and array inputs correctly.
        
    #     Args:
    #         i: OFMAP row index(s) (int or array-like)
    #         j: OFMAP column index(s) (int or array-like)
            
    #     Returns:
    #         ndarray of IFMAP element addresses with shape:
    #         (num_positions, filter_rows, filter_cols, num_input_channels)
    #         or (filter_rows, filter_cols, num_input_channels) for scalar inputs
    #         Invalid accesses marked with -1
    #     """
    #     # Convert inputs to numpy arrays and ensure proper shapes
    #     i = np.atleast_1d(np.asarray(i))
    #     j = np.atleast_1d(np.asarray(j))
        
    #     # Get layer configuration
    #     ifmap_base = self.ifmap_offset
    #     filter_h = self.filter_rows
    #     filter_w = self.filter_cols
    #     channels = self.num_input_channels
    #     stride_h = self.row_stride
    #     stride_w = self.col_stride
        
    #     # Calculate IFMAP window coordinates
    #     i_row = i * stride_h
    #     i_col = j * stride_w
        
    #     # Generate all possible filter positions
    #     c_row, c_col = np.mgrid[:filter_h, :filter_w]
        
    #     # Calculate corresponding IFMAP coordinates
    #     ifmap_row = i_row[:, np.newaxis, np.newaxis] + c_row
    #     ifmap_col = i_col[:, np.newaxis, np.newaxis] + c_col
        
    #     # Initialize output array
    #     output_shape = (len(i), filter_h, filter_w, channels)
    #     ifmap_px_addr = np.full(output_shape, -1, dtype=int)
        
    #     # Create validity mask
    #     valid_mask = ((ifmap_row < self.ifmap_rows) & 
    #                 (ifmap_col < self.ifmap_cols))
        
    #     # Calculate base addresses for valid positions
    #     base_addrs = (ifmap_base +
    #                 ifmap_row * self.ifmap_cols * channels +
    #                 ifmap_col * channels)
        
    #     # Add channel offsets to valid positions
    #     for pos_idx in range(len(i)):
    #         pos_valid = valid_mask[pos_idx]
    #         if pos_valid.any():
    #             channel_offsets = np.arange(channels)
    #             ifmap_px_addr[pos_idx, pos_valid] = (
    #                 base_addrs[pos_idx, pos_valid][:, np.newaxis] + channel_offsets
    #             ).reshape(-1)
        
    #     # Squeeze single position results
    #     if ifmap_px_addr.shape[0] == 1:
    #         return ifmap_px_addr[0]
    #     return ifmap_px_addr


    # ds
    # def calc_ifmap_elem_addr(self, i, j):
    #     """
    #     Calculates IFMAP element addresses with proper broadcasting.
        
    #     Args:
    #         i: OFMAP row indices (int or array-like)
    #         j: OFMAP column indices (int or array-like)
            
    #     Returns:
    #         ndarray of IFMAP addresses with shape:
    #         (num_positions, filter_rows*filter_cols*channels)
    #     """
    #     # Convert inputs to numpy arrays
    #     i = np.asarray(i)
    #     j = np.asarray(j)
        
    #     # Get configuration parameters
    #     ifmap_base = self.ifmap_offset
    #     filter_h = self.filter_rows
    #     filter_w = self.filter_cols
    #     channels = self.num_input_channels
    #     stride_h = self.row_stride
    #     stride_w = self.col_stride
        
    #     # Calculate IFMAP window coordinates
    #     i_row = np.atleast_1d(i * stride_h)
    #     i_col = np.atleast_1d(j * stride_w)
        
    #     # Generate filter positions
    #     c_idx = np.arange(filter_h * filter_w)
    #     c_row, c_col = np.divmod(c_idx, filter_w)
        
    #     # Reshape for proper broadcasting
    #     ifmap_row = i_row.reshape(-1, 1) + c_row.reshape(1, -1)  # shape (n_pos, n_filt)
    #     ifmap_col = i_col.reshape(-1, 1) + c_col.reshape(1, -1)
        
    #     # Initialize output array
    #     num_positions = len(i_row)
    #     output_size = filter_h * filter_w * channels
    #     ifmap_px_addr = np.full((num_positions, output_size), -1, dtype=int)
        
    #     # Create validity mask
    #     valid_mask = ((ifmap_row < self.ifmap_rows) & 
    #                 (ifmap_col < self.ifmap_cols))
        
    #     # Calculate addresses for valid positions
    #     for pos_idx in range(num_positions):
    #         pos_valid = valid_mask[pos_idx]
    #         if pos_valid.any():
    #             # Calculate base addresses
    #             base_addrs = (ifmap_base +
    #                         ifmap_row[pos_idx, pos_valid] * self.ifmap_cols * channels +
    #                         ifmap_col[pos_idx, pos_valid] * channels)
                
    #             # Add channel offsets
    #             channel_offsets = np.arange(channels)
    #             linear_indices = np.flatnonzero(pos_valid)[:, np.newaxis] * channels + channel_offsets
    #             ifmap_px_addr[pos_idx, linear_indices.ravel()] = (
    #                 base_addrs[:, np.newaxis] + channel_offsets
    #             ).ravel()
        
    #     return ifmap_px_addr.squeeze()



    # def create_ofmap_matrix(self):
    #     my_name = 'operand_matrix.create_ofmap_matrix(): '
    #     err_prefix = 'Error: ' + my_name
    #     if not self.params_set_flag:
    #         message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
    #         print(message)
    #         return -1

    #     row_indices = np.expand_dims(np.arange(self.ofmap_px_per_filt), axis=1)
    #     col_indices = np.arange(self.num_filters)
    #     self.ofmap_addr_matrix = self.calc_ofmap_elem_addr(row_indices, col_indices)

    #     return 0

    # ICL1sucess

    def calc_ifmap_elem_addr(self, i, j):
        """
        Calculate ifmap element memory addresses for given indices.
        
        Args:
            i: Output feature map element index (flattened)
            j: Filter window element index (flattened)
            
        Returns:
            Memory address for each (i,j) pair, or -1 for invalid accesses
        """
        # 1. Get basic configuration parameters
        base_addr = self.ifmap_offset
        ifmap_rows = self.ifmap_rows
        ifmap_cols = self.ifmap_cols
        num_channels = self.num_input_channels
        row_stride = self.row_stride
        col_stride = self.col_stride
        filter_rows = self.filter_rows
        filter_cols = self.filter_cols
        
        # 2. Calculate output feature map coordinates
        ofmap_row = i // self.ofmap_cols
        ofmap_col = i % self.ofmap_cols
        
        # 3. Calculate input window starting position
        i_row = ofmap_row * row_stride
        i_col = ofmap_col * col_stride
        
        # 4. Break down filter window coordinates
        window_pos = j // (filter_cols * num_channels)
        window_inner = j % (filter_cols * num_channels)
        
        c_row = window_pos
        c_col = window_inner // num_channels
        ch = window_inner % num_channels
        
        # 5. Calculate actual ifmap coordinates
        ifmap_row = i_row + c_row
        ifmap_col = i_col + c_col
        
        # 6. Check bounds and calculate address
        valid = (ifmap_row < ifmap_rows) & (ifmap_col < ifmap_cols) & \
                (ifmap_row >= 0) & (ifmap_col >= 0)
        
        addr = np.where(valid,
                    base_addr + \
                    (ifmap_row * ifmap_cols + ifmap_col) * num_channels + ch,
                    -1)
        
        return addr

    def create_ofmap_matrix(self):
        """
        Creates the output feature map (OFMAP) address matrix based on layer configuration.
        
        Returns:
            int: 0 if successful, -1 if parameters aren't set
        """
        # Branch structure: Parameter validation
        if not self.params_set_flag:
            error_msg = ("Error: Layer parameters not configured. "
                        "Call set_layer_params() before creating matrices.")
            print(error_msg)
            return -1

        # Sequential structure: Matrix generation
        # Generate row indices (batch * ofmap pixels) as column vector
        rows = np.arange(self.batch_size * self.ofmap_px_per_filt).reshape(-1, 1)
        
        # Generate column indices (filters)
        cols = np.arange(self.num_filters)
        
        # Calculate addresses (automatic broadcasting)
        self.ofmap_addr_matrix = self.calc_ofmap_elem_addr(rows, cols)
        
        return 0

    # logic to translate ofmap into matrix resulting systolic array MACs
    # def calc_ofmap_elem_addr(self, i, j):
    #     offset = self.ofmap_offset
    #     num_filt = self.num_filters
    #     internal_address = num_filt * i + j
    #     ofmap_px_addr = internal_address + offset
    #     return ofmap_px_addr

    def calc_ofmap_elem_addr(self, i, j):
        """
        Calculates the address of an element in the output feature map (OFMAP).

        Args:
            i: Row index of the OFMAP (numpy array or scalar)
            j: Column index of the OFMAP (numpy array or scalar)

        Returns:
            numpy.ndarray or int: Computed OFMAP element address(es)
        """
        # Compute internal address: num_filt * i + j
        internal_addr = self.num_filters * i + j
        
        # Add offset to get final address
        ofmap_px_addr = internal_addr + self.num_filters
        
        return ofmap_px_addr

    # creates the filter operand
    # def create_filter_matrix(self):
    #     my_name = 'operand_matrix.create_filter_matrix(): '
    #     err_prefix = 'Error: ' + my_name
    #     if not self.params_set_flag:
    #         message = err_prefix + 'Parameters not set yet. Run set_params(). Exiting'
    #         print(message)
    #         return -1

    #     row_indices = np.expand_dims(np.arange(self.conv_window_size), axis=1)
    #     col_indices = np.arange(self.num_filters)
    #     self.filter_addr_matrix = self.calc_filter_elem_addr(row_indices, col_indices)

    #     return 0

    def create_filter_matrix(self):
        """
        Creates the filter address matrix based on layer configuration.
        
        Returns:
            int: 0 if successful, -1 if parameters aren't set
        
        Matrix Dimensions:
            Rows: conv_window_size (filter_rows * filter_cols * num_input_channels)
            Columns: num_filters
        """
        # Branch structure: Parameter validation
        if not self.params_set_flag:
            error_msg = ("Error: Layer parameters not configured. "
                        "Call set_layer_params() before creating matrices.")
            print(error_msg)
            return -1

        # Sequential structure: Matrix generation
        # Generate row indices (conv window positions) as column vector
        rows = np.expand_dims(np.arange(self.conv_window_size), axis=1)
        
        # Generate column indices (filters)
        cols = np.arange(self.num_filters)
        
        # Calculate addresses (automatic broadcasting)
        self.filter_addr_matrix = self.calc_filter_elem_addr(rows, cols)
        
        return 0

    # logic to translate filter into matrix fed into systolic array MACs
    # def calc_filter_elem_addr(self, i, j):
    #     offset = self.filter_offset
    #     filter_row = self.filter_rows
    #     filter_col = self.filter_cols
    #     channel = self.num_input_channels
    #     internal_address = j * filter_row * filter_col * channel + i
    #     filter_px_addr = internal_address + offset
    #     return filter_px_addr

# # correct 1
#     def calc_filter_elem_addr(self, i, j):
#         """
#         Calculates the memory address of a filter element based on its position.

#         Args:
#             i: Row index within the filter (0 to filter_rows-1)
#             j: Column index within the filter (0 to filter_cols-1)

#         Returns:
#             int: Absolute memory address of the filter element

#         Note:
#             Assumes CHW memory layout: [channels][rows][columns]
#             Supports scalar and numpy array inputs via broadcasting
#         """
#         # Calculate internal address components
#         row_component = i * self.filter_cols * self.num_input_channels
#         col_component = j * self.num_input_channels
        
#         # Combine components and add offset
#         filter_px_addr = self.filter_offset + row_component + col_component
        
#         return filter_px_addr


    def calc_filter_elem_addr(self, i, j):
        """
        Calculates the memory address of a filter element using custom layout formula.
        
        Memory layout formula: j * (filter_rows * filter_cols * num_input_channels) + i
        
        Args:
            i: Minor dimension index (typically row)
            j: Major dimension index (typically column)
            
        Returns:
            int: Absolute memory address of the specified filter element
            
        Note:
            - Supports scalar and numpy array inputs via broadcasting
            - Formula implies column-major ordering with channel as fastest changing
        """
        # Calculate internal address using specified formula
        internal_addr = j * (self.filter_rows * self.filter_cols * self.num_input_channels) + i
        
        # Apply memory offset
        filter_px_addr = self.filter_offset + internal_addr
        
        return filter_px_addr

    # function to get a part or the full ifmap operand
    # def get_ifmap_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
    #                           num_cols=-1):
    #     if num_rows == -1:
    #         num_rows = self.ofmap_px_per_filt
    #     if num_cols == -1:
    #         num_cols = self.conv_window_size
    #     my_name = 'operand_matrix.get_ifmap_matrix_part(): '
    #     err_prefix = 'Error: ' + my_name
    #     if not self.matrices_ready_flag:
    #         if self.params_set_flag:
    #             self.create_operand_matrices()
    #         else:
    #             message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
    #             print(message)
    #             return -1, np.zeros((1, 1))
    #     if (start_row + num_rows) > self.ofmap_px_per_filt or (start_col + num_cols) > self.conv_window_size:
    #         message = err_prefix + ": Illegal arguments. Exiting!"
    #         print(message)
    #         return -2, np.zeros((1, 1))

    #     # Anand: ISSUE #3. Patch
    #     #end_row = start_row + num_rows + 1
    #     #end_col = start_col + num_cols + 1
    #     #ret_mat = self.ifmap_addr_matrix[start_row: end_row][start_col: end_col]
    #     end_row = start_row + num_rows
    #     end_col = start_col + num_cols
    #     ret_mat = self.ifmap_addr_matrix[start_row: end_row, start_col: end_col]
    #     return 0, ret_mat


    def get_ifmap_matrix_part(self, start_row=0, num_rows=-1, start_col=0, num_cols=-1):
        """
        Retrieves a specified portion of the IFMAP address matrix.
        
        Args:
            start_row: Starting row index (default: 0)
            num_rows: Number of rows to retrieve (default: -1 for all remaining)
            start_col: Starting column index (default: 0)
            num_cols: Number of columns to retrieve (default: -1 for all remaining)
        
        Returns:
            tuple: (status_code, matrix_portion)
                status_code: 0=success, -1=params not set, -2=out of bounds
                matrix_portion: Requested matrix portion or zero matrix on error
        """
        # Handle default values
        if num_rows == -1:
            num_rows = self.ofmap_px_per_filt
        if num_cols == -1:
            num_cols = self.conv_window_size

        # Check matrix readiness
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                error_msg = ("Error: Parameters not set. "
                            "Call set_layer_params() first.")
                print(error_msg)
                return -1, np.zeros((1,1), dtype=int)

        # Boundary checking
        if (start_row + num_rows > self.ifmap_addr_matrix.shape[0] or
            start_col + num_cols > self.ifmap_addr_matrix.shape[1]):
            error_msg = ("Error: Requested portion exceeds matrix dimensions. "
                        f"Matrix shape: {self.ifmap_addr_matrix.shape}, "
                        f"Requested: [{start_row}:{start_row+num_rows}, "
                        f"{start_col}:{start_col+num_cols}]")
            print(error_msg)
            return -2, np.zeros((1,1), dtype=int)

        # Calculate slice indices
        end_row = start_row + num_rows
        end_col = start_col + num_cols

        # Extract matrix portion
        ret_mat = self.ifmap_addr_matrix[start_row:end_row, start_col:end_col]

        return 0, ret_mat

    def get_ifmap_matrix(self):
        return self.get_ifmap_matrix_part()

    # function to get a part or the full filter operand
    # def get_filter_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
    #                            num_cols=-1):

    #     if num_rows == -1:
    #         num_rows = self.conv_window_size
    #     if num_cols == -1:
    #         num_cols = self.num_filters
    #     my_name = 'operand_matrix.get_filter_matrix_part(): '
    #     err_prefix = 'Error: ' + my_name
    #     if not self.matrices_ready_flag:
    #         if self.params_set_flag:
    #             self.create_operand_matrices()
    #         else:
    #             message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
    #             print(message)
    #             return -1, np.zeros((1, 1))
    #     if (start_row + num_rows) > self.conv_window_size or (start_col + num_cols) > self.num_filters:
    #         message = err_prefix + ": Illegal arguments. Exiting!"
    #         print(message)
    #         return -2, np.zeros((1, 1))

    #     # Anand: ISSUE #3. FIX
    #     #end_row = start_row + num_rows + 1
    #     #end_col = start_col + num_cols + 1
    #     end_row = start_row + num_rows
    #     end_col = start_col + num_cols

    #     # Anand: ISSUE #3. FIX
    #     #ret_mat = self.filter_addr_matrix[start_row: end_row][start_col: end_col]
    #     ret_mat = self.filter_addr_matrix[start_row: end_row, start_col: end_col]
    #     return 0, ret_mat

    def get_filter_matrix_part(self, start_row=0, num_rows=-1, start_col=0, num_cols=-1):
        """
        Retrieves a specified portion of the filter address matrix.
        
        Args:
            start_row: Starting row index (default: 0)
            num_rows: Number of rows to retrieve (default: -1 for all remaining)
            start_col: Starting column index (default: 0)
            num_cols: Number of columns to retrieve (default: -1 for all remaining)
        
        Returns:
            tuple: (status_code, matrix_portion)
                status_code: 0=success, -1=params not set, -2=out of bounds
                matrix_portion: Requested matrix portion or zero matrix on error
        """
        # Handle default values
        num_rows = self.conv_window_size if num_rows == -1 else num_rows
        num_cols = self.num_filters if num_cols == -1 else num_cols

        # Check matrix readiness
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                print("Error: Parameters not configured. Call set_layer_params() first.")
                return -1, np.zeros((1,1), dtype=int)

        # Boundary checking
        matrix_rows, matrix_cols = self.filter_addr_matrix.shape
        if (start_row + num_rows > matrix_rows or 
            start_col + num_cols > matrix_cols):
            print(f"Error: Requested slice [{start_row}:{start_row+num_rows}, "
                f"{start_col}:{start_col+num_cols}] exceeds matrix dimensions "
                f"{self.filter_addr_matrix.shape}")
            return -2, np.zeros((1,1), dtype=int)

        # Calculate and return slice
        end_row = start_row + num_rows
        end_col = start_col + num_cols
        return 0, self.filter_addr_matrix[start_row:end_row, start_col:end_col]

    def get_filter_matrix(self):
        return self.get_filter_matrix_part()

    # function to get a part or the full ofmap operand
    # def get_ofmap_matrix_part(self, start_row=0, num_rows=-1, start_col=0,
    #                            num_cols=-1):

    #     # Since we cannot pass self as an argument in the member functions
    #     # This is an alternate way of making the matrix dimensions as defaults
    #     if num_rows == -1:
    #         num_rows = self.ofmap_px_per_filt
    #     if num_cols == -1:
    #         num_cols = self.num_filters
    #     my_name = 'operand_matrix.get_ofmap_matrix_part(): '
    #     err_prefix = 'Error: ' + my_name
    #     if not self.matrices_ready_flag:
    #         if self.params_set_flag:
    #             self.create_operand_matrices()
    #         else:
    #             message = err_prefix + ": Parameters not set yet. Run set_params(). Exiting!"
    #             print(message)
    #             return -1, np.zeros((1, 1))
    #     if (start_row + num_rows) > self.ofmap_px_per_filt or (start_col + num_cols) > self.num_filters:
    #         message = err_prefix + ": Illegal arguments. Exiting!"
    #         print(message)
    #         return -2, np.zeros((1, 1))

    #     # Anand: ISSUE #3. Patch
    #     #end_row = start_row + num_rows + 1
    #     #end_col = start_col + num_cols + 1
    #     #ret_mat = self.filter_addr_matrix[start_row: end_row][start_col: end_col]
    #     end_row = start_row + num_rows
    #     end_col = start_col + num_cols
    #     # Anand: ISSUE #7. Patch
    #     #ret_mat = self.filter_addr_matrix[start_row: end_row, start_col: end_col]
    #     ret_mat = self.ofmap_addr_matrix[start_row: end_row, start_col: end_col]

    #     return 0, ret_mat

    def get_ofmap_matrix_part(self, start_row=0, num_rows=-1, start_col=0, num_cols=-1):
        """
        Retrieves a specified portion of the OFMAP address matrix.
        
        Args:
            start_row: Starting row index (default: 0)
            num_rows: Number of rows to retrieve (default: -1 for all remaining)
            start_col: Starting column index (default: 0)
            num_cols: Number of columns to retrieve (default: -1 for all remaining)
        
        Returns:
            tuple: (status_code, matrix_portion)
                status_code: 0=success, -1=params not set, -2=out of bounds
                matrix_portion: Requested matrix portion or zero matrix on error
        """
        # Handle default values
        num_rows = self.ofmap_px_per_filt if num_rows == -1 else num_rows
        num_cols = self.num_filters if num_cols == -1 else num_cols

        # Check matrix readiness
        if not self.matrices_ready_flag:
            if self.params_set_flag:
                self.create_operand_matrices()
            else:
                print("Error: Parameters not configured. Call set_layer_params() first.")
                return -1, np.zeros((1,1), dtype=int)

        # Boundary checking
        matrix_rows, matrix_cols = self.ofmap_addr_matrix.shape
        if (start_row + num_rows > matrix_rows or 
            start_col + num_cols > matrix_cols):
            print(f"Error: Requested slice [{start_row}:{start_row+num_rows}, "
                f"{start_col}:{start_col+num_cols}] exceeds matrix dimensions "
                f"{self.ofmap_addr_matrix.shape}")
            return -2, np.zeros((1,1), dtype=int)

        # Calculate and return slice
        end_row = start_row + num_rows
        end_col = start_col + num_cols
        return 0, self.ofmap_addr_matrix[start_row:end_row, start_col:end_col]

    def get_ofmap_matrix(self):
        return self.get_ofmap_matrix_part()

    # def get_all_operand_matrix(self):
    #     if not self.matrices_ready_flag:
    #         me = 'operand_matrix.' + 'get_all_operand_matrix()'
    #         message = 'ERROR:' + me + ': Matrices not ready or matrix gen failed'
    #         print(message)
    #         return

    #     return self.ifmap_addr_matrix, \
    #            self.filter_addr_matrix, \
    #            self.ofmap_addr_matrix

    def get_all_operand_matrix(self):
        """
        Retrieves all operand matrices (IFMAP, filter, OFMAP) if they are ready.
        
        Returns:
            tuple: (ifmap_matrix, filter_matrix, ofmap_matrix) if ready
            None: If matrices are not ready
            
        Note:
            Matrices must be created first via create_operand_matrices()
            Check matrices_ready_flag before calling this method
        """
        if not self.matrices_ready_flag:
            print("Error: Operand matrices not ready. Call create_operand_matrices() first.")
            return None
            
        return (
            self.ifmap_addr_matrix,
            self.filter_addr_matrix,
            self.ofmap_addr_matrix
        )


if __name__ == '__main__':
    opmat = operand_matrix()
    tutil = topoutil()
    lid = 3
    topology_file = "../../topologies/mlperf/test.csv"
    tutil.load_arrays(topofile=topology_file)
    for i in range(tutil.get_num_layers()):
        layer_param_arr = tutil.get_layer_params(layer_id=i)
        ofmap_dims = tutil.get_layer_ofmap_dims(layer_id=i)
        ofmap_px_filt = tutil.get_layer_num_ofmap_px(layer_id=i) / tutil.get_layer_num_filters(layer_id=i)
        conv_window_size = tutil.get_layer_window_size(layer_id=i)
        layer_calc_hyper_param_arr = [ofmap_dims[0], ofmap_dims[1], ofmap_px_filt, conv_window_size]
        config_arr = [512, 512, 256, 8, 8]
        #[matrix_set, ifmap_addr_matrix, filter_addr_matrix, ofmap_addr_matrix] \
        #    = opmat.set_params(layer_hyper_param_arr=layer_param_arr[1:],
        #                       layer_calc_hyper_param_arr=layer_calc_hyper_param_arr,
        #                       offset_list=[0, 1000000, 2000000])
