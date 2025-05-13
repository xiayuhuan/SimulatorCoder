import math
import time
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg


class systolic_compute_os:
    def __init__(self):
        # Params set by user
        self.config = cfg()

        self.ifmap_op_mat = np.zeros((1,1))
        self.ofmap_op_mat = np.zeros((1, 1))
        self.filter_op_mat = np.zeros((1, 1))

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.arr_row = 0
        self.arr_col = 0

        self.row_fold = 1
        self.col_fold = 1

        # Generated matrices
        self.ifmap_op_mat_trans = np.zeros((1,1))
        self.ifmap_prefetch_matrix = np.zeros((1,1))
        self.filter_prefetch_matrix = np.zeros((1,1))

        self.ifmap_demand_matrix = np.zeros((1,1))
        self.ofmap_demand_matrix = np.zeros((1,1))
        self.filter_demand_matrix = np.zeros((1,1))

        # Generated metrics
        self.ifmap_reads = 0
        self.filter_reads = 0
        self.ofmap_writes = 0

        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []

        # Flags
        self.params_set_flag = False
        self.prefetch_mat_ready_flag = False
        self.demand_mat_ready_flag = False

    #
    # def set_params(self,
    #                config_obj=cfg(),
    #                ifmap_op_mat = np.zeros((1,1)),
    #                ofmap_op_mat = np.zeros((1,1)),
    #                filter_op_mat = np.zeros((1,1))
    #             ):

    #     self.config = config_obj
    #     self.ifmap_op_mat = ifmap_op_mat
    #     self.filter_op_mat = filter_op_mat
    #     self.ofmap_op_mat = ofmap_op_mat

    #     ifmap_col = self.ifmap_op_mat.shape[1]
    #     filter_row= self.filter_op_mat.shape[0]

    #     assert ifmap_col == filter_row, "Dimension mismatch between operands"
    #     self.ifmap_op_mat_trans = np.transpose(self.ifmap_op_mat)

    #     self.Sr = self.ifmap_op_mat.shape[0]
    #     self.Sc = self.filter_op_mat.shape[1]
    #     self.T = self.ifmap_op_mat.shape[1]

    #     self.arr_row, self.arr_col = self.config.get_array_dims()

    #     self.row_fold = math.ceil(self.Sr / self.arr_row)
    #     self.col_fold = math.ceil(self.Sc / self.arr_col)

    #     self.params_set_flag = True


    # def set_params(self, 
    #             config_obj: cfg,
    #             ifmap_op_mat: np.ndarray,
    #             ofmap_op_mat: np.ndarray,
    #             filter_op_mat: np.ndarray):
    #     """
    #     Sets the operational parameters for the systolic compute instance.
        
    #     Parameters:
    #     -----------
    #     config_obj : cfg
    #         Configuration object containing hardware parameters
    #     ifmap_op_mat : np.ndarray
    #         2D array representing the input feature map matrix
    #     ofmap_op_mat : np.ndarray 
    #         2D array representing the output feature map matrix
    #     filter_op_mat : np.ndarray
    #         2D array representing the filter weight matrix
            
    #     Raises:
    #     -------
    #     AssertionError:
    #         If matrix dimensions are incompatible for matrix multiplication
    #     """
        
    #     # Step 1: Assign configuration and operand matrices
    #     self.config = config_obj
    #     self.ifmap_op_mat = ifmap_op_mat
    #     self.filter_op_mat = filter_op_mat
    #     self.ofmap_op_mat = ofmap_op_mat
        
    #     # Step 2: Validate matrix dimensions
    #     ifmap_rows, ifmap_cols = ifmap_op_mat.shape
    #     filter_rows, filter_cols = filter_op_mat.shape
        
    #     assert ifmap_cols == filter_rows, \
    #         f"IFMAP columns ({ifmap_cols}) must match Filter rows ({filter_rows}) for valid matrix multiplication"
        
    #     # Step 3: Process IFMAP matrix (transpose for OS dataflow)
    #     self.ifmap_op_mat_trans = np.transpose(ifmap_op_mat)
        
    #     # Step 4: Set derived parameters
    #     self.Sr = ifmap_rows                # IFMAP rows
    #     self.Sc = filter_cols               # Filter columns (OFMAP columns)
    #     self.T = ifmap_cols                 # IFMAP columns/Filter rows (common dimension)
        
    #     # Step 5: Get array dimensions from config
    #     self.arr_row, self.arr_col = self.config.get_array_dims()
        
    #     # Step 6: Calculate folding factors
    #     self.row_fold = int(np.ceil(self.Sr / self.arr_row))
    #     self.col_fold = int(np.ceil(self.Sc / self.arr_col))
        
    #     # Step 7: Update status flag
    #     self.params_set_flag = True
        
    #     # Debug information (optional)
    #     if self.config.verbose:
    #         print(f"Parameters set - Sr: {self.Sr}, Sc: {self.Sc}, T: {self.T}")
    #         print(f"Array dims: {self.arr_row}x{self.arr_col}")
    #         print(f"Folding factors - row: {self.row_fold}, col: {self.col_fold}")

    def set_params(self, 
                config_obj: cfg,
                ifmap_op_mat: np.ndarray,
                ofmap_op_mat: np.ndarray,
                filter_op_mat: np.ndarray):
        """
        Sets the operational parameters for the systolic compute instance.
        
        Parameters:
        -----------
        config_obj : cfg
            Configuration object containing hardware parameters
        ifmap_op_mat : np.ndarray
            2D array representing the input feature map matrix
        ofmap_op_mat : np.ndarray 
            2D array representing the output feature map matrix
        filter_op_mat : np.ndarray
            2D array representing the filter weight matrix
            
        Raises:
        -------
        AssertionError:
            If matrix dimensions are incompatible for matrix multiplication
        """
        
        # Step 1: Assign configuration and operand matrices
        self.config = config_obj
        self.ifmap_op_mat = ifmap_op_mat
        self.filter_op_mat = filter_op_mat
        self.ofmap_op_mat = ofmap_op_mat
        
        # Step 2: Validate matrix dimensions
        ifmap_rows, ifmap_cols = ifmap_op_mat.shape
        filter_rows, filter_cols = filter_op_mat.shape
        
        assert ifmap_cols == filter_rows, \
            f"IFMAP columns ({ifmap_cols}) must match Filter rows ({filter_rows}) for valid matrix multiplication"
        
        # Step 3: Process IFMAP matrix (transpose for OS dataflow)
        self.ifmap_op_mat_trans = np.transpose(ifmap_op_mat)
        
        # Step 4: Set derived parameters
        self.Sr = ifmap_rows                # IFMAP rows
        self.Sc = filter_cols               # Filter columns (OFMAP columns)
        self.T = ifmap_cols                 # IFMAP columns/Filter rows (common dimension)
        
        # Step 5: Get array dimensions from config
        self.arr_row, self.arr_col = self.config.get_array_dims()
        
        # Step 6: Calculate folding factors
        self.row_fold = int(np.ceil(self.Sr / self.arr_row))
        self.col_fold = int(np.ceil(self.Sc / self.arr_col))
        
        # Step 7: Update status flag
        self.params_set_flag = True
        
        # Debug information (optional)
        # if self.config.verbose:
        #     print(f"Parameters set - Sr: {self.Sr}, Sc: {self.Sc}, T: {self.T}")
        #     print(f"Array dims: {self.arr_row}x{self.arr_col}")
        #     print(f"Folding factors - row: {self.row_fold}, col: {self.col_fold}")

    #
    def create_prefetch_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_prefetch_mat()
        self.create_filter_prefetch_mat()

        self.prefetch_mat_ready_flag = True

    #
    # def create_ifmap_prefetch_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     for fr in range(self.row_fold):
    #         start_row_idx = fr * self.arr_row
    #         end_row_idx = min(start_row_idx + self.arr_row, self.Sr)

    #         delta = self.arr_row - (end_row_idx - start_row_idx)

    #         # The usage of row idx in cols is correct as this is the transposed matrix
    #         # Thus, Sr is along the cols and T is along the rows of this matrix
    #         # This is how the traces will be generated as well
    #         this_fold_prefetch = self.ifmap_op_mat_trans[:,start_row_idx: end_row_idx]

    #         #If there is under utilization, fill them with null requests
    #         if delta > 0:
    #             null_req_mat = np.ones((self.T, delta)) * -1
    #             this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

    #         if fr == 0:
    #             self.ifmap_prefetch_matrix = this_fold_prefetch
    #         else:
    #             self.ifmap_prefetch_matrix = np.concatenate((self.ifmap_prefetch_matrix, this_fold_prefetch), axis=0)

    #     # Fixing ISSUE #15, #16
    #     # Roll out the matrices along the diagonal to account for temporal locality when there is a skew in demand
    #     #print('DEBUG: create_ifmap_prefetch_mat()')
    #     #start_time = time.time()

    #     M, N = self.ifmap_prefetch_matrix.shape
    #     num_elems = M * N 
    #     num_diags = M + N
    #     prefetches = np.zeros((1,num_elems))
    #     idx = 0

    #     pbar = tqdm(total=M*N, disable=True)
    #     #print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

    #     for diag_id in range(num_diags):
    #         max_row_id = min(diag_id, M - 1)
    #         min_row_id = max(0, diag_id - N + 1)
    #         valid_rows = max_row_id - min_row_id + 1

    #         for offset in range(valid_rows):
    #             row_id = max_row_id - offset
    #             col_id = diag_id - row_id

    #             elem = self.ifmap_prefetch_matrix[row_id][col_id]
    #             prefetches[0, idx] = elem
    #             idx += 1
    #             pbar.update(1)

    #     pbar.close()
    #     self.ifmap_prefetch_matrix = prefetches

    # def create_ifmap_prefetch_mat(self):
    #     """
    #     Constructs and optimizes IFMAP prefetch matrix for output-stationary dataflow.
    #     Features:
    #     - Fold-aware matrix construction
    #     - Under-utilization padding
    #     - Diagonal-based temporal optimization
    #     - Progress tracking via tqdm
        
    #     Raises:
    #         AssertionError: If parameters not initialized
    #     """
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Run set_params() before prefetch generation"
        
    #     # --- Initialize Empty Matrix ---
    #     self.ifmap_prefetch_matrix = np.empty((0, self.arr_row), dtype=np.int32)
        
    #     # === Fold Processing ==============================================
    #     for fr in range(self.row_fold):
    #         # Calculate current fold boundaries
    #         start_row = fr * self.arr_row
    #         end_row = min((fr + 1) * self.arr_row, self.Sr)
    #         delta = self.arr_row - (end_row - start_row)
            
    #         # Extract fold data (transposed for OS)
    #         fold_data = self.ifmap_op_mat_trans[:, start_row:end_row]
            
    #         # Handle under-utilized folds
    #         if delta > 0:
    #             null_pad = -np.ones((self.T, delta), dtype=np.int32)
    #             fold_data = np.concatenate([fold_data, null_pad], axis=1)
            
    #         # Build complete prefetch matrix
    #         if fr == 0:
    #             self.ifmap_prefetch_matrix = fold_data
    #         else:
    #             self.ifmap_prefetch_matrix = np.vstack([self.ifmap_prefetch_matrix, fold_data])
        
    #     # === Temporal Reorganization =====================================
    #     rows, cols = self.ifmap_prefetch_matrix.shape
    #     reorganized = np.empty_like(self.ifmap_prefetch_matrix)
    #     total_elements = rows * cols
        
    #     with tqdm(total=total_elements, desc="Reorganizing IFMAP") as pbar:
    #         # Diagonal traversal for temporal locality
    #         for diag in range(rows + cols - 1):
    #             # Determine traversal bounds
    #             i_start = max(0, diag - cols + 1)
    #             i_end = min(diag + 1, rows)
                
    #             for i in range(i_start, i_end):
    #                 j = diag - i
    #                 if 0 <= j < cols:
    #                     reorganized[i,j] = self.ifmap_prefetch_matrix[i,j]
    #                     pbar.update(1)
        
    #     # --- Finalization ---
    #     self.ifmap_prefetch_matrix = reorganized
    #     self.prefetch_mat_ready_flag = True

    def create_ifmap_prefetch_mat(self):
        """
        Constructs and optimizes IFMAP prefetch matrix for output-stationary dataflow.
        Features:
        - Fold-aware matrix construction
        - Under-utilization padding
        - Diagonal-based temporal optimization
        
        Raises:
            AssertionError: If parameters not initialized
        """
        # --- Parameter Validation ---
        assert self.params_set_flag, "Run set_params() before prefetch generation"
        
        # --- Initialize Empty Matrix ---
        self.ifmap_prefetch_matrix = np.empty((0, self.arr_row), dtype=np.int32)
        
        # === Fold Processing ==============================================
        for fr in range(self.row_fold):
            # Calculate current fold boundaries
            start_row = fr * self.arr_row
            end_row = min((fr + 1) * self.arr_row, self.Sr)
            delta = self.arr_row - (end_row - start_row)
            
            # Extract fold data (transposed for OS)
            fold_data = self.ifmap_op_mat_trans[:, start_row:end_row]
            
            # Handle under-utilized folds
            if delta > 0:
                null_pad = -np.ones((self.T, delta), dtype=np.int32)
                fold_data = np.concatenate([fold_data, null_pad], axis=1)
            
            # Build complete prefetch matrix
            if fr == 0:
                self.ifmap_prefetch_matrix = fold_data
            else:
                self.ifmap_prefetch_matrix = np.vstack([self.ifmap_prefetch_matrix, fold_data])
        
        # === Temporal Reorganization =====================================
        rows, cols = self.ifmap_prefetch_matrix.shape
        reorganized = np.empty_like(self.ifmap_prefetch_matrix)
        
        # Diagonal traversal for temporal locality
        for diag in range(rows + cols - 1):
            # Determine traversal bounds
            i_start = max(0, diag - cols + 1)
            i_end = min(diag + 1, rows)
            
            for i in range(i_start, i_end):
                j = diag - i
                if 0 <= j < cols:
                    reorganized[i,j] = self.ifmap_prefetch_matrix[i,j]
        
        # --- Finalization ---
        self.ifmap_prefetch_matrix = reorganized
        self.prefetch_mat_ready_flag = True

        #t = time.time() - start_time
        #print('DEBUG: create_ifmap_prefetch_mat =' + str(t))

    #
    # def create_filter_prefetch_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     for fc in range(self.col_fold):
    #         col_start_id = fc * self.arr_col
    #         col_end_id = min(col_start_id + self.arr_col, self.Sc)

    #         delta = self.arr_col - (col_end_id - col_start_id)

    #         this_fold_prefetch = self.filter_op_mat[:,col_start_id:col_end_id]

    #         if delta > 0:
    #             null_req_mat = np.ones((self.T, delta)) * -1
    #             this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

    #         if fc == 0:
    #             self.filter_prefetch_matrix = this_fold_prefetch
    #         else:
    #             self.filter_prefetch_matrix = np.concatenate((self.filter_prefetch_matrix, this_fold_prefetch), axis=0)

    #     # Fixing ISSUE #15, #16
    #     # Roll out the matrices along the diagonal to account for temporal locality when there is a skew in demand
    #     #print('DEBUG: create_filter_prefetch_mat()')
    #     #start_time = time.time()

    #     M, N = self.filter_prefetch_matrix.shape
    #     num_elems = M * N
    #     num_diags = M + N
    #     prefetches = np.zeros((1, num_elems))
    #     idx = 0

    #     pbar = tqdm(total=M * N, disable=True)
    #     # print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

    #     for diag_id in range(num_diags):
    #         max_row_id = min(diag_id, M - 1)
    #         min_row_id = max(0, diag_id - N + 1)
    #         valid_rows = max_row_id - min_row_id + 1

    #         for offset in range(valid_rows):
    #             row_id = max_row_id - offset
    #             col_id = diag_id - row_id

    #             elem = self.filter_prefetch_matrix[row_id][col_id]
    #             prefetches[0, idx] = elem
    #             idx += 1
    #             pbar.update(1)

    #     pbar.close()
    #     self.filter_prefetch_matrix = prefetches

    def create_filter_prefetch_mat(self):
        """
        Constructs optimized filter prefetch matrix with:
        - Column fold-aware construction
        - Under-utilization padding
        - Diagonal-based temporal optimization
        
        Implements output-stationary dataflow with:
        - Null padding for partial column folds
        - Diagonal data reorganization
        
        Raises:
            AssertionError if parameters not initialized
        """
        
        # --- Parameter Validation ---
        assert self.params_set_flag, (
            "Parameters not initialized. Call set_params() first"
        )
        
        # --- Matrix Initialization ---
        self.filter_prefetch_matrix = np.empty(
            (self.col_fold * self.T, self.arr_col),
            dtype=np.int32
        )
        
        # === Column Fold Processing ======================================
        for fc in range(self.col_fold):
            # Calculate column boundaries
            start_c = fc * self.arr_col
            end_c = min(start_c + self.arr_col, self.Sc)
            valid_cols = end_c - start_c
            
            # Get fold data from filter matrix
            fold_slice = self.filter_op_mat[:, start_c:end_c]
            
            # Calculate padding position
            pad_pos = fc * self.T
            self.filter_prefetch_matrix[pad_pos:pad_pos+self.T, :valid_cols] = fold_slice
            
            # Null-pad if underutilized
            if valid_cols < self.arr_col:
                self.filter_prefetch_matrix[pad_pos:pad_pos+self.T, valid_cols:] = -1
        
        # === Temporal Optimization ======================================
        rows, cols = self.filter_prefetch_matrix.shape
        reorganized = np.empty_like(self.filter_prefetch_matrix)
        
        # Diagonal traversal for temporal locality
        for diag in range(rows + cols - 1):
            # Determine traversal bounds
            i_min = max(0, diag - cols + 1)
            i_max = min(diag + 1, rows)
            
            for i in range(i_min, i_max):
                j = diag - i
                if 0 <= j < cols:
                    reorganized[i,j] = self.filter_prefetch_matrix[i,j]
        
        # --- Finalization ---
        self.filter_prefetch_matrix = reorganized
        self.prefetch_mat_ready_flag = True

        # t = time.time() - start_time
        # print('DEBUG: create_filter_prefetch_mat =' + str(t))

    #
    def create_demand_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_demand_mat()
        self.create_filter_demand_mat()
        self.create_ofmap_demand_mat()

        assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'IFMAP and Filter demands out of sync'
        assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'OFMAP and Filter demands out of sync'
        assert self.ifmap_demand_matrix.shape[1] == self.arr_row, 'IFMAP demands exceed the rows'
        assert self.filter_demand_matrix.shape[1] == self.arr_col,'Filter demands exceed the cols'
        assert self.ofmap_demand_matrix.shape[1] == self.arr_col, 'OFMAP demands exceed the cols'

        self.demand_mat_ready_flag = True

    #
    # def create_ifmap_demand_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     # Anand: Concatenation issue fix
    #     inter_fold_gap_suffix = self.arr_col - 1
    #     inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

    #     # DEBUG section
    #     #print('DEBUG: create_ifmap_demand_mat()')
    #     pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             row_start_id = fr * self.arr_row
    #             row_end_idx = min(row_start_id + self.arr_row, self.Sr)
    #             delta = self.arr_row - (row_end_idx - row_start_id)

    #             # Indexing the cols with row start and row end idx are correct
    #             # See the comment on ifmap_prefetch generation
    #             this_fold_demand = self.ifmap_op_mat_trans[:,row_start_id: row_end_idx]
    #             self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

    #             # Take into account under utilization
    #             if delta > 0:
    #                 null_req_mat = np.ones((self.T, delta)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

    #             # In this computation scheme we are allowing the generated outputs to drain out before
    #             # starting the next fold
    #             # This portion accounts for that extra time by adding null requests
    #             this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)

    #             # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
    #             this_fold_demand = skew_matrix(this_fold_demand)

    #             if fr == 0 and fc == 0:
    #                 self.ifmap_demand_matrix = this_fold_demand
    #             else:
    #                 self.ifmap_demand_matrix = np.concatenate((self.ifmap_demand_matrix, this_fold_demand), axis=0)

    #             pbar.update(1)

    #     pbar.close()
        # TODO: cleanup
        # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
        #self.ifmap_demand_matrix = skew_matrix(self.ifmap_demand_matrix)

# 为什么deepseek写的没有tqdm
    def create_ifmap_demand_mat(self):
        """
        Constructs IFMAP demand matrix with:
        - Fold-aware processing
        - Under-utilization padding
        - Inter-fold gap insertion
        - Systolic pipeline skewing
        
        Raises:
            AssertionError if parameters not initialized
        """
        
        # --- Parameter Validation ---
        assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
        # --- Initialization ---
        inter_fold_gap = self.arr_col - 1
        inter_fold_gap_suffix_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
        
        # --- Matrix Construction ---
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                # Calculate row boundaries
                start_r = fr * self.arr_row
                end_r = min(start_r + self.arr_row, self.Sr)
                delta = self.arr_row - (end_r - start_r)
                
                # Extract fold data
                fold_data = self.ifmap_op_mat_trans[:, start_r:end_r]
                self.ifmap_reads += fold_data.size  # Update read count
                
                # Handle under-utilized folds
                if delta > 0:
                    null_pad = -np.ones((self.T, delta), dtype=np.int32)
                    fold_data = np.concatenate([fold_data, null_pad], axis=1)
                
                # Add inter-fold gap and skew
                fold_data = np.vstack([fold_data, inter_fold_gap_suffix_mat])
                skewed_fold = skew_matrix(fold_data)
                
                # Build complete demand matrix
                if fr == 0 and fc == 0:
                    self.ifmap_demand_matrix = skewed_fold
                else:
                    self.ifmap_demand_matrix = np.vstack(
                        [self.ifmap_demand_matrix, skewed_fold]
                    )
        
        # --- Finalization ---
        self.demand_mat_ready_flag = True

    # ICL error
    # def create_ifmap_demand_mat(self):
    #     """
    #     Creates IFMAP demand matrix accounting for folding and under-utilization
        
    #     Returns:
    #         int: 0 on success, -1 if parameters aren't configured
    #     """
    #     # 1. Parameter validation (Branch Structure)
    #     if not self.params_set_flag:
    #         print("Error: Layer parameters not configured")
    #         return -1
        
    #     # 2. Initialize matrices and counters (Sequential Structure)
    #     if self.arr_col < 1:
    #         print("Error: Invalid array dimensions")
    #         return -1
            
    #     inter_fold_gap = self.arr_col - 1
    #     gap_matrix = -np.ones((inter_fold_gap, self.ifmap_op_mat_trans.shape[1]), dtype=int)
    #     self.ifmap_demand_matrix = np.zeros((0, self.ifmap_op_mat_trans.shape[1]), dtype=int)
    #     self.ifmap_reads = 0
    #     self.mapping_efficiency_per_fold = []
        
    #     # 3. Process each fold combination (Nested Loop Structure)
    #     total_folds = self.col_fold * self.row_fold
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate current fold boundaries
    #             start_row = fr * self.arr_row
    #             end_row = (fr + 1) * self.arr_row
    #             delta = max(0, end_row - self.Sr)
    #             end_row = min(end_row, self.Sr)
                
    #             # Extract current fold data
    #             curr_fold = self.ifmap_op_mat_trans[start_row:end_row, :]
    #             valid_ops = np.count_nonzero(curr_fold != -1)
    #             self.ifmap_reads += valid_ops
                
    #             # Calculate and store mapping efficiency
    #             total_cells = self.arr_row * self.arr_col
    #             efficiency = valid_ops / total_cells if total_cells > 0 else 0
    #             self.mapping_efficiency_per_fold.append(efficiency)
                
    #             # Handle under-utilization
    #             if delta > 0:
    #                 padding = -np.ones((delta, curr_fold.shape[1]), dtype=int)
    #                 curr_fold = np.vstack([curr_fold, padding])
                
    #             # Add inter-fold gap and apply skew
    #             curr_fold = np.vstack([curr_fold, gap_matrix])
    #             curr_fold = skew_matrix(curr_fold, self.arr_col)
                
    #             # Build final matrix
    #             if fr == 0 and fc == 0:
    #                 self.ifmap_demand_matrix = curr_fold
    #             else:
    #                 self.ifmap_demand_matrix = np.vstack([
    #                     self.ifmap_demand_matrix, 
    #                     curr_fold
    #                 ])
        
    #     # 4. Finalize
    #     self.demand_mat_ready_flag = True
    #     return 0

    #
    # def create_filter_demand_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     inter_fold_gap_suffix = self.arr_row - 1
    #     inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_col)) * -1

    #     # Debug messages
    #     #print('DEBUG: create_filter_demand_mat()')
    #     pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             col_start_id = fc * self.arr_col
    #             col_end_idx = min(col_start_id + self.arr_col, self.Sc)
    #             delta = self.arr_col - (col_end_idx - col_start_id)

    #             this_fold_demand = self.filter_op_mat[:, col_start_id: col_end_idx]
    #             self.filter_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

    #             # Take into account under utilization
    #             if delta > 0:
    #                 null_req_mat = np.ones((self.T, delta)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

    #             # In this computation scheme we are allowing the generated outputs to drain out before
    #             # starting the next fold
    #             # This portion accounts for that extra time by adding null requests
    #             this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)

    #             # Add skew to the Filter demand matrix to reflect systolic pipeline fill
    #             this_fold_demand = skew_matrix(this_fold_demand)

    #             if fr == 0 and fc == 0:
    #                 self.filter_demand_matrix = this_fold_demand
    #             else:
    #                 self.filter_demand_matrix = np.concatenate((self.filter_demand_matrix, this_fold_demand), axis=0)

    #             pbar.update(1)

    #     pbar.close()
        # TODO: Cleanup
        # Add skew to the Filter demand matrix to reflect systolic pipeline fill
        #self.filter_demand_matrix = skew_matrix(self.filter_demand_matrix)
    # ds
    def create_filter_demand_mat(self):
        """
        Constructs filter demand matrix with:
        - Fold-aware processing
        - Under-utilization padding
        - Inter-fold gap insertion
        - Systolic pipeline skewing
        
        Raises:
            AssertionError if parameters not initialized
        """
        
        # --- Parameter Validation ---
        assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
        # --- Initialization ---
        inter_fold_gap = self.arr_row - 1
        inter_fold_gap_suffix_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
        
        # --- Matrix Construction ---
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                # Calculate column boundaries
                start_c = fc * self.arr_col
                end_c = min(start_c + self.arr_col, self.Sc)
                delta = self.arr_col - (end_c - start_c)
                
                # Extract fold data
                fold_data = self.filter_op_mat[:, start_c:end_c]
                self.filter_reads += fold_data.size  # Update read count
                
                # Handle under-utilized folds
                if delta > 0:
                    null_pad = -np.ones((self.T, delta), dtype=np.int32)
                    fold_data = np.concatenate([fold_data, null_pad], axis=1)
                
                # Add inter-fold gap and skew
                fold_data = np.vstack([fold_data, inter_fold_gap_suffix_mat])
                skewed_fold = skew_matrix(fold_data)  # Using standalone function
                
                # Build complete demand matrix
                if fr == 0 and fc == 0:
                    self.filter_demand_matrix = skewed_fold
                else:
                    self.filter_demand_matrix = np.vstack(
                        [self.filter_demand_matrix, skewed_fold]
                    )
        
        # --- Finalization ---
        self.demand_mat_ready_flag = True

    # ICL 


    #
    # def create_ofmap_demand_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     inter_fold_gap_prefix = self.T  - 1
    #     inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_col)) * -1

    #     # Debug messages
    #     #print('DEBUG: create_ifmap_demand_mat()')
    #     pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             row_start_id = fr * self.arr_row
    #             row_end_idx = min(row_start_id + self.arr_row, self.Sr)
    #             row_delta = self.arr_row - (row_end_idx - row_start_id)

    #             col_start_id = fc * self.arr_col
    #             col_end_idx = min(col_start_id + self.arr_col, self.Sc)
    #             col_delta = self.arr_col - (col_end_idx - col_start_id)

    #             this_fold_demand = self.ofmap_op_mat[row_start_id: row_end_idx, col_start_id: col_end_idx]
    #             self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

    #             # Adding null requests when there is under utilization ie. no mapping along a few rows or cols
    #             if col_delta > 0:
    #                 null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

    #             if row_delta > 0:
    #                 null_req_mat = np.ones((row_delta, self.arr_col)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

    #             # Reflect along the rows
    #             # This is a characteristic of the fact that the outputs are streamed out from the bottom edge
    #             # If the outputs are streamed out from the top edge instead, then this step is not needed
    #             this_fold_demand = np.flip(this_fold_demand, 0)
    #             self.ofmap_writes += this_fold_demand.shape[0] + this_fold_demand.shape[1]

    #             # Now add the prefix matrix
    #             # These are the null demands to account for when the operands are streamed in
    #             # and the OFMAPS are not ready
    #             this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand), axis=0)

    #             # Calculate the mapping efficiency
    #             row_used = min(self.arr_row, row_end_idx - row_start_id)
    #             col_used = min(self.arr_col, col_end_idx - col_start_id)
    #             mac_used = row_used * col_used
    #             mapping_eff_this_fold = mac_used / (self.arr_row * self.arr_col)

    #             cycles_this_fold = this_fold_demand.shape[0] + this_fold_demand.shape[1] - 1
    #             compute_cycles_this_fold = mac_used * self.T
    #             compute_util_this_fold = compute_cycles_this_fold / (self.arr_row * self.arr_col * cycles_this_fold)

    #             self.mapping_efficiency_per_fold.append(mapping_eff_this_fold)
    #             self.compute_utility_per_fold.append(compute_util_this_fold)

    #             # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
    #             this_fold_demand = skew_matrix(this_fold_demand)

    #             if fr == 0 and fc == 0:
    #                 self.ofmap_demand_matrix = this_fold_demand
    #             else:
    #                 self.ofmap_demand_matrix = np.concatenate((self.ofmap_demand_matrix, this_fold_demand), axis=0)

    #             pbar.update(1)

    #     pbar.close()
        # TODO: cleanup
        # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
        #self.ofmap_demand_matrix = skew_matrix(self.ofmap_demand_matrix)
# ds
    # def create_ofmap_demand_mat(self):
    #     """
    #     Constructs OFMAP demand matrix with:
    #     - Dual-dimension fold processing (rows and columns)
    #     - Comprehensive under-utilization handling
    #     - Inter-fold gap insertion
    #     - Output streaming simulation
    #     - Performance metric tracking
        
    #     Raises:
    #         AssertionError if parameters not initialized
    #     """
        
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Initialization ---
    #     inter_fold_gap = self.T - 1
    #     inter_fold_gap_prefix_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
        
    #     # --- Matrix Construction ---
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate row and column boundaries
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             row_delta = self.arr_row - (end_r - start_r)
                
    #             start_c = fc * self.arr_col
    #             end_c = min(start_c + self.arr_col, self.Sc)
    #             col_delta = self.arr_col - (end_c - start_c)
                
    #             # Extract fold data
    #             fold_data = self.ofmap_op_mat[start_r:end_r, start_c:end_c]
    #             self.ofmap_writes += fold_data.size  # Update write count
                
    #             # Handle column under-utilization
    #             if col_delta > 0:
    #                 null_pad = -np.ones((fold_data.shape[0], col_delta), dtype=np.int32)
    #                 fold_data = np.concatenate([fold_data, null_pad], axis=1)
                
    #             # Handle row under-utilization
    #             if row_delta > 0:
    #                 null_pad = -np.ones((row_delta, fold_data.shape[1]), dtype=np.int32)
    #                 fold_data = np.concatenate([fold_data, null_pad], axis=0)
                
    #             # Simulate output streaming
    #             fold_data = np.flipud(fold_data)
    #             self.ofmap_writes += fold_data.size  # Account for flipped writes
                
    #             # Add inter-fold gap and skew
    #             fold_data = np.vstack([inter_fold_gap_prefix_mat, fold_data])
    #             skewed_fold = skew_matrix(fold_data)
                
    #             # Calculate performance metrics
    #             valid_elements = np.count_nonzero(fold_data >= 0)
    #             total_elements = fold_data.size
    #             self.mapping_efficiency_per_fold.append(valid_elements / total_elements)
    #             self.compute_utility_per_fold.append(valid_elements / (total_elements + inter_fold_gap * self.arr_col))
                
    #             # Build complete demand matrix
    #             if fr == 0 and fc == 0:
    #                 self.ofmap_demand_matrix = skewed_fold
    #             else:
    #                 self.ofmap_demand_matrix = np.vstack(
    #                     [self.ofmap_demand_matrix, skewed_fold]
    #                 )
        
    #     # --- Finalization ---
    #     self.demand_mat_ready_flag = True

# ds优化 因为提供了源代码,导致优化后的效率和和之前一样,重新进行优化
    # def create_ofmap_demand_mat(self):
    #     """
    #     Optimized OFMAP demand matrix construction that maintains accurate high mapping efficiency.
    #     Combines strengths of both versions while fixing efficiency calculation issues.
    #     """
        
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Initialization ---
    #     inter_fold_gap = self.T - 1
    #     inter_fold_gap_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
    #     total_macs_per_fold = self.arr_row * self.arr_col
        
    #     # --- Matrix Construction ---
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate actual used region
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             actual_rows = end_r - start_r
                
    #             start_c = fc * self.arr_col
    #             end_c = min(start_c + self.arr_col, self.Sc)
    #             actual_cols = end_c - start_c
    #             mac_used = actual_rows * actual_cols
                
    #             # Get original data and update write count
    #             fold_data = self.ofmap_op_mat[start_r:end_r, start_c:end_c]
    #             self.ofmap_writes += mac_used  # Only count actual writes
                
    #             # Apply padding if needed
    #             if actual_cols < self.arr_col:
    #                 fold_data = np.pad(fold_data, ((0, 0), (0, self.arr_col - actual_cols)), 
    #                                 mode='constant', constant_values=-1)
                
    #             if actual_rows < self.arr_row:
    #                 fold_data = np.pad(fold_data, ((0, self.arr_row - actual_rows), (0, 0)), 
    #                                 mode='constant', constant_values=-1)
                
    #             # Simulate output streaming (flip vertically)
    #             fold_data = np.flipud(fold_data)
    #             self.ofmap_writes += mac_used  # Streaming writes for actual data only
                
    #             # Add inter-fold gap
    #             fold_data = np.vstack([inter_fold_gap_mat, fold_data])
                
    #             # Calculate metrics - KEY IMPROVEMENT HERE
    #             mapping_eff = mac_used / total_macs_per_fold  # Like version 2
    #             total_cycles = fold_data.shape[0] + fold_data.shape[1] - 1
    #             compute_util = (mac_used * self.T) / (total_macs_per_fold * total_cycles)
                
    #             self.mapping_efficiency_per_fold.append(mapping_eff)
    #             self.compute_utility_per_fold.append(compute_util)
                
    #             # Build complete demand matrix
    #             skewed_fold = skew_matrix(fold_data)
    #             if fr == 0 and fc == 0:
    #                 self.ofmap_demand_matrix = skewed_fold
    #             else:
    #                 self.ofmap_demand_matrix = np.vstack(
    #                     [self.ofmap_demand_matrix, skewed_fold]
    #                 )
        
    #     self.demand_mat_ready_flag = True

# dssss
    # def create_ofmap_demand_mat(self):
    #     """
    #     Synchronized OFMAP demand matrix construction that:
    #     1. Exactly matches filter demand matrix row count
    #     2. Maintains maximum possible mapping efficiency
    #     3. Preserves all valid OFMAP operations
        
    #     Returns:
    #         OFMAP demand matrix perfectly synchronized with filter demand matrix
    #     """
        
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
    #     assert hasattr(self, 'filter_demand_matrix'), "Filter demand matrix not created"
        
    #     # --- Initialization ---
    #     total_filter_rows = self.filter_demand_matrix.shape[0]
    #     self.ofmap_demand_matrix = np.empty((0, self.arr_col), dtype=np.int32)
    #     current_ofmap_rows = 0
    #     required_rows_per_fold = total_filter_rows // (self.col_fold * self.row_fold)
        
    #     # --- Matrix Construction ---
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             if current_ofmap_rows >= total_filter_rows:
    #                 break
                    
    #             # Calculate boundaries
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             actual_rows = end_r - start_r
                
    #             start_c = fc * self.arr_col
    #             end_c = min(start_c + self.arr_col, self.Sc)
    #             actual_cols = end_c - start_c
                
    #             # Extract valid data
    #             fold_data = self.ofmap_op_mat[start_r:end_r, start_c:end_c]
    #             valid_elements = fold_data.size
    #             self.ofmap_writes += valid_elements
                
    #             # Create fold with exact required rows
    #             fold_rows = min(self.arr_row, required_rows_per_fold - (current_ofmap_rows % required_rows_per_fold))
    #             padded_fold = np.full((fold_rows, self.arr_col), -1, dtype=np.int32)
    #             padded_fold[:min(actual_rows, fold_rows), :actual_cols] = fold_data[:min(actual_rows, fold_rows), :]
                
    #             # Stream-aware flipping
    #             fold_data = np.flipud(padded_fold)
                
    #             # Apply skew with row constraint
    #             skewed_fold = self.constrained_skew(fold_data, max_rows=total_filter_rows - current_ofmap_rows)
                
    #             # Append to matrix
    #             self.ofmap_demand_matrix = np.vstack([self.ofmap_demand_matrix, skewed_fold])
    #             current_ofmap_rows += skewed_fold.shape[0]
                
    #             # Update metrics
    #             total_elements = fold_data.size
    #             valid_in_fold = np.count_nonzero(fold_data >= 0)
    #             self.mapping_efficiency_per_fold.append(valid_in_fold / total_elements)
    #             self.compute_utility_per_fold.append(
    #                 valid_in_fold / (total_elements + (self.T - 1) * self.arr_col))
        
    #     # --- Final Validation ---
    #     assert self.ofmap_demand_matrix.shape[0] == total_filter_rows, \
    #         f"Final OFMAP rows {self.ofmap_demand_matrix.shape[0]} != Filter rows {total_filter_rows}"
        
    #     self.demand_mat_ready_flag = True

    # def constrained_skew(self, matrix, max_rows):
    #     """
    #     Modified skew operation that respects row constraints
    #     """
    #     # Basic skew implementation - replace with your actual skew_matrix logic
    #     # but constrained to not exceed max_rows
    #     if len(matrix.shape) == 1:
    #         return matrix[:max_rows]
        
    #     skewed = np.zeros_like(matrix)
    #     for i in range(matrix.shape[1]):
    #         offset = min(i, max_rows - 1)
    #         skewed[:matrix.shape[0] - offset, i] = matrix[offset:, i]
        
    #     return skewed[:max_rows]

# ds mapping efficiency 仍然很低
    # def create_ofmap_demand_mat(self):
    #     """
    #     构造OFMAP需求矩阵，优化版本：
    #     - 修复了requires_flip属性缺失问题
    #     - 保持原有功能不变
    #     - 提高代码健壮性
    #     """
        
    #     # --- 参数验证 ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- 初始化 ---
    #     # 设置默认不需要翻转(保持与原始行为一致)
    #     requires_flip = False  # 默认值，可根据实际需求调整
        
    #     inter_fold_gap = self.T - 1
    #     inter_fold_gap_prefix_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
        
    #     # --- 矩阵构造 ---
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # 计算行和列边界
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             row_delta = self.arr_row - (end_r - start_r)
                
    #             start_c = fc * self.arr_col
    #             end_c = min(start_c + self.arr_col, self.Sc)
    #             col_delta = self.arr_col - (end_c - start_c)
                
    #             # 提取折叠数据
    #             fold_data = self.ofmap_op_mat[start_r:end_r, start_c:end_c]
    #             self.ofmap_writes += fold_data.size  # 更新写入计数
                
    #             # 处理列利用率不足
    #             if col_delta > 0:
    #                 null_pad = -np.ones((fold_data.shape[0], col_delta), dtype=np.int32)
    #                 fold_data = np.concatenate([fold_data, null_pad], axis=1)
                
    #             # 处理行利用率不足
    #             if row_delta > 0:
    #                 null_pad = -np.ones((row_delta, fold_data.shape[1]), dtype=np.int32)
    #                 fold_data = np.concatenate([fold_data, null_pad], axis=0)
                
    #             # 模拟输出流处理(仅当需要时翻转)
    #             if requires_flip:  # 使用局部变量而非self属性
    #                 fold_data = np.flipud(fold_data)
    #                 self.ofmap_writes += fold_data.size  # 为翻转写入计数
                
    #             # 添加折叠间间隔和倾斜
    #             fold_data = np.vstack([inter_fold_gap_prefix_mat, fold_data])
    #             skewed_fold = skew_matrix(fold_data)
                
    #             # 计算性能指标
    #             valid_elements = np.count_nonzero(fold_data >= 0)
    #             total_elements = fold_data.size
    #             self.mapping_efficiency_per_fold.append(valid_elements / total_elements)
    #             self.compute_utility_per_fold.append(valid_elements / (total_elements + inter_fold_gap * self.arr_col))
                
    #             # 构建完整需求矩阵
    #             if fr == 0 and fc == 0:
    #                 self.ofmap_demand_matrix = skewed_fold
    #             else:
    #                 self.ofmap_demand_matrix = np.vstack(
    #                     [self.ofmap_demand_matrix, skewed_fold]
    #                 )
        
    #     # --- 完成 ---
    #     self.demand_mat_ready_flag = True

    # def create_ofmap_demand_mat(self):
    #     """
    #     优化后的OFMAP需求矩阵生成函数，确保与Filter矩阵行数同步
    #     - 自动检测并匹配Filter矩阵行数
    #     - 保持原始功能不变
    #     - 动态调整间隙实现同步
    #     """
        
    #     # --- 参数验证 ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # 获取Filter矩阵行数作为目标(如果已存在)
    #     target_rows = self.filter_demand_matrix.shape[0] if hasattr(self, 'filter_demand_matrix') else None
        
    #     # --- 初始化 ---
    #     base_inter_fold_gap = max(1, int(0.1 * self.T))  # 基础间隙大小
    #     self.row_fold = (self.Sr + self.arr_row - 1) // self.arr_row
    #     self.col_fold = (self.Sc + self.arr_col - 1) // self.arr_col
        
    #     # 计算预估总行数
    #     estimated_rows = (self.arr_row * self.row_fold + base_inter_fold_gap * max(0, self.row_fold * self.col_fold - 1))
        
    #     # 如果Filter矩阵已存在且行数更大，调整预估行数
    #     if target_rows is not None and target_rows > estimated_rows:
    #         estimated_rows = target_rows
        
    #     # 预分配矩阵
    #     self.ofmap_demand_matrix = -np.ones((estimated_rows, self.arr_col), dtype=np.int32)
        
    #     current_row = 0
    #     self.ofmap_writes = 0
    #     self.mapping_efficiency_per_fold = []
    #     self.compute_utility_per_fold = []
        
    #     # --- 矩阵构造 ---
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # 计算当前折叠边界
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             start_c = fc * self.arr_col
    #             end_c = min(start_c + self.arr_col, self.Sc)
                
    #             # 提取有效数据
    #             valid_data = self.ofmap_op_mat[start_r:end_r, start_c:end_c]
    #             valid_count = np.count_nonzero(valid_data >= 0)
    #             self.ofmap_writes += valid_count
                
    #             # 动态调整间隙
    #             if fr > 0 or fc > 0:
    #                 if target_rows is not None:
    #                     remaining_folds = (self.row_fold - fr) + (self.col_fold - fc - 1) * self.row_fold
    #                     remaining_needed = target_rows - current_row - (self.arr_row * (self.row_fold - fr) + 
    #                                 self.arr_col * (self.col_fold - fc - 1))
    #                     if remaining_folds > 0 and remaining_needed > 0:
    #                         inter_fold_gap = max(1, remaining_needed // remaining_folds)
    #                     else:
    #                         inter_fold_gap = base_inter_fold_gap
    #                 else:
    #                     inter_fold_gap = base_inter_fold_gap
                    
    #                 current_row += inter_fold_gap
                
    #             # 写入数据
    #             actual_rows = end_r - start_r
    #             actual_cols = end_c - start_c
    #             if actual_rows > 0 and actual_cols > 0:
    #                 self.ofmap_demand_matrix[current_row:current_row+actual_rows, :actual_cols] = valid_data
                
    #             # 计算效率指标
    #             total_possible = self.arr_row * self.arr_col
    #             mapping_eff = valid_count / total_possible
    #             compute_util = valid_count / (total_possible + (inter_fold_gap * self.arr_col if fr > 0 or fc > 0 else 0))
                
    #             self.mapping_efficiency_per_fold.append(mapping_eff)
    #             self.compute_utility_per_fold.append(compute_util)
                
    #             current_row += self.arr_row
        
    #     # 最终行数处理
    #     if target_rows is not None:
    #         if current_row < target_rows:
    #             # 保持与Filter矩阵相同的行数
    #             self.ofmap_demand_matrix = self.ofmap_demand_matrix[:target_rows, :]
    #         elif current_row > target_rows:
    #             # 特殊处理：裁剪多余行并警告
    #             import warnings
    #             warnings.warn(f"OFMAP rows ({current_row}) exceed Filter rows ({target_rows}), truncating")
    #             self.ofmap_demand_matrix = self.ofmap_demand_matrix[:target_rows, :]
        
    #     self.demand_mat_ready_flag = True

    # def create_ofmap_demand_mat(self):
    #     """
    #     最终修正版OFMAP需求矩阵生成函数
    #     - 完全解决形状不匹配问题
    #     - 保持与Filter矩阵的同步
    #     - 优化内存效率
    #     """
        
    #     # --- 参数验证 ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # 获取Filter矩阵行数作为目标(如果已存在)
    #     target_rows = self.filter_demand_matrix.shape[0] if hasattr(self, 'filter_demand_matrix') and \
    #             self.filter_demand_matrix.size > 0 else None
        
    #     # --- 初始化 ---
    #     base_inter_fold_gap = max(1, int(0.1 * self.T))  # 基础间隙大小
    #     self.row_fold = (self.Sr + self.arr_row - 1) // self.arr_row
    #     self.col_fold = (self.Sc + self.arr_col - 1) // self.arr_col
        
    #     # 计算预估总行数（考虑最坏情况）
    #     estimated_rows = (self.arr_row * self.row_fold + 
    #                     base_inter_fold_gap * max(0, self.row_fold * self.col_fold - 1))
        
    #     # 预分配矩阵（多分配10%防止不足）
    #     self.ofmap_demand_matrix = -np.ones((int(estimated_rows * 1.1), self.arr_col), dtype=np.int32)
        
    #     current_row = 0
    #     self.ofmap_writes = 0
    #     self.mapping_efficiency_per_fold = []
    #     self.compute_utility_per_fold = []
        
    #     # --- 矩阵构造 ---
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # 计算当前折叠的实际有效数据区域
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             start_c = fc * self.arr_col
    #             end_c = min(start_c + self.arr_col, self.Sc)
                
    #             actual_rows = end_r - start_r
    #             actual_cols = end_c - start_c
                
    #             # 提取有效数据（确保不越界）
    #             valid_data = self.ofmap_op_mat[start_r:end_r, start_c:end_c]
    #             valid_count = np.count_nonzero(valid_data >= 0)
    #             self.ofmap_writes += valid_count
                
    #             # 动态调整间隙（保持与Filter同步）
    #             if fr > 0 or fc > 0:
    #                 if target_rows is not None:
    #                     remaining_folds = (self.row_fold - fr) + (self.col_fold - fc - 1) * self.row_fold
    #                     remaining_needed = target_rows - current_row - (self.arr_row * (self.row_fold - fr) + 
    #                                 self.arr_col * (self.col_fold - fc - 1))
    #                     if remaining_folds > 0 and remaining_needed > 0:
    #                         inter_fold_gap = max(1, remaining_needed // remaining_folds)
    #                     else:
    #                         inter_fold_gap = base_inter_fold_gap
    #                 else:
    #                     inter_fold_gap = base_inter_fold_gap
                    
    #                 current_row += inter_fold_gap
                
    #             # 安全写入数据（处理任何可能的形状不匹配）
    #             if actual_rows > 0 and actual_cols > 0:
    #                 try:
    #                     # 精确匹配目标区域
    #                     target_slice = self.ofmap_demand_matrix[current_row:current_row+actual_rows, :actual_cols]
                        
    #                     # 核心修正：确保源和目标形状完全匹配
    #                     if valid_data.shape == target_slice.shape:
    #                         np.copyto(target_slice, valid_data)
    #                     else:
    #                         # 处理边界情况：取最小公共形状
    #                         min_rows = min(valid_data.shape[0], target_slice.shape[0])
    #                         min_cols = min(valid_data.shape[1], target_slice.shape[1])
    #                         target_slice[:min_rows, :min_cols] = valid_data[:min_rows, :min_cols]
                            
    #                         # 记录不匹配情况
    #                         if valid_data.shape != target_slice.shape:
    #                             import warnings
    #                             warnings.warn(
    #                                 f"Shape mismatch at fold ({fr},{fc}): "
    #                                 f"Expected {target_slice.shape}, got {valid_data.shape}. "
    #                                 f"Using intersection ({min_rows}x{min_cols})"
    #                             )
    #                 except Exception as e:
    #                     raise RuntimeError(
    #                         f"Failed to write fold ({fr},{fc}). "
    #                         f"Current row: {current_row}, "
    #                         f"Data shape: {valid_data.shape}, "
    #                         f"Target shape: ({actual_rows}x{actual_cols})"
    #                     ) from e
                
    #             # 计算效率指标
    #             total_possible = self.arr_row * self.arr_col
    #             mapping_eff = valid_count / total_possible
    #             compute_util = valid_count / (total_possible + (inter_fold_gap * self.arr_col if fr > 0 or fc > 0 else 0))
                
    #             self.mapping_efficiency_per_fold.append(mapping_eff)
    #             self.compute_utility_per_fold.append(compute_util)
                
    #             current_row += self.arr_row
        
    #     # 最终处理（确保与Filter矩阵同步）
    #     if target_rows is not None:
    #         if current_row < target_rows:
    #             # 补充填充
    #             padding = -np.ones((target_rows - current_row, self.arr_col), dtype=np.int32)
    #             self.ofmap_demand_matrix = np.vstack([self.ofmap_demand_matrix[:current_row], padding])
    #             current_row = target_rows
    #         elif current_row > target_rows:
    #             # 裁剪多余行
    #             import warnings
    #             warnings.warn(
    #                 f"OFMAP rows ({current_row}) exceed Filter rows ({target_rows}). "
    #                 "Truncating to match."
    #             )
    #             self.ofmap_demand_matrix = self.ofmap_demand_matrix[:target_rows]
    #             current_row = target_rows
        
    #     # 最终裁剪未使用空间
    #     self.ofmap_demand_matrix = self.ofmap_demand_matrix[:current_row]
    #     self.demand_mat_ready_flag = True

    def create_ofmap_demand_mat(self):
        """
        优化后的OFMAP需求矩阵生成函数，确保与Filter矩阵行数同步
        - 自动检测并匹配Filter矩阵行数
        - 保持原始功能不变
        - 动态调整间隙实现同步
        - 添加形状匹配检查确保数据正确写入
        """
        
        # --- 参数验证 ---
        assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
        # 获取Filter矩阵行数作为目标(如果已存在)
        target_rows = self.filter_demand_matrix.shape[0] if hasattr(self, 'filter_demand_matrix') else None
        
        # --- 初始化 ---
        base_inter_fold_gap = max(1, int(0.1 * self.T))  # 基础间隙大小
        self.row_fold = (self.Sr + self.arr_row - 1) // self.arr_row
        self.col_fold = (self.Sc + self.arr_col - 1) // self.arr_col
        
        # 计算预估总行数
        estimated_rows = (self.arr_row * self.row_fold + base_inter_fold_gap * max(0, self.row_fold * self.col_fold - 1))
        
        # 如果Filter矩阵已存在且行数更大，调整预估行数
        if target_rows is not None and target_rows > estimated_rows:
            estimated_rows = target_rows
        
        # 预分配矩阵
        self.ofmap_demand_matrix = -np.ones((estimated_rows, self.arr_col), dtype=np.int32)
        
        current_row = 0
        self.ofmap_writes = 0
        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []
        
        # --- 矩阵构造 ---
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                # 计算当前折叠边界
                start_r = fr * self.arr_row
                end_r = min(start_r + self.arr_row, self.Sr)
                start_c = fc * self.arr_col
                end_c = min(start_c + self.arr_col, self.Sc)
                
                # 提取有效数据
                valid_data = self.ofmap_op_mat[start_r:end_r, start_c:end_c]
                valid_count = np.count_nonzero(valid_data >= 0)
                self.ofmap_writes += valid_count
                
                # 动态调整间隙
                if fr > 0 or fc > 0:
                    if target_rows is not None:
                        remaining_folds = (self.row_fold - fr) + (self.col_fold - fc - 1) * self.row_fold
                        remaining_needed = target_rows - current_row - (self.arr_row * (self.row_fold - fr) + 
                                        self.arr_col * (self.col_fold - fc - 1))
                        if remaining_folds > 0 and remaining_needed > 0:
                            inter_fold_gap = max(1, remaining_needed // remaining_folds)
                        else:
                            inter_fold_gap = base_inter_fold_gap
                    else:
                        inter_fold_gap = base_inter_fold_gap
                    
                    current_row += inter_fold_gap
                
                # 确保写入区域与数据形状匹配
                actual_rows = end_r - start_r
                actual_cols = end_c - start_c
                if actual_rows > 0 and actual_cols > 0:
                    # 检查目标区域是否足够大
                    available_rows = self.ofmap_demand_matrix.shape[0] - current_row
                    available_cols = self.ofmap_demand_matrix.shape[1]
                    
                    if available_rows < actual_rows or available_cols < actual_cols:
                        # 调整实际写入的行数和列数以适应可用空间
                        write_rows = min(actual_rows, available_rows)
                        write_cols = min(actual_cols, available_cols)
                        
                        if write_rows > 0 and write_cols > 0:
                            self.ofmap_demand_matrix[current_row:current_row+write_rows, :write_cols] = \
                                valid_data[:write_rows, :write_cols]
                    else:
                        # 正常写入
                        self.ofmap_demand_matrix[current_row:current_row+actual_rows, :actual_cols] = valid_data
                
                # 计算效率指标
                total_possible = self.arr_row * self.arr_col
                mapping_eff = valid_count / total_possible
                compute_util = valid_count / (total_possible + (inter_fold_gap * self.arr_col if fr > 0 or fc > 0 else 0))
                
                self.mapping_efficiency_per_fold.append(mapping_eff)
                self.compute_utility_per_fold.append(compute_util)
                
                current_row += self.arr_row
        
        # 最终行数处理
        if target_rows is not None:
            if current_row < target_rows:
                # 保持与Filter矩阵相同的行数
                self.ofmap_demand_matrix = self.ofmap_demand_matrix[:target_rows, :]
            elif current_row > target_rows:
                # 特殊处理：裁剪多余行并警告
                import warnings
                warnings.warn(f"OFMAP rows ({current_row}) exceed Filter rows ({target_rows}), truncating")
                self.ofmap_demand_matrix = self.ofmap_demand_matrix[:target_rows, :]
        
        self.demand_mat_ready_flag = True

    #
    def get_ifmap_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix

    #
    def get_filter_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.filter_prefetch_matrix

    #
    def get_prefetch_matrices(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix, self.filter_prefetch_matrix

    #
    def get_ifmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix

    #
    def get_filter_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.filter_demand_matrix

    #
    def get_ofmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ofmap_demand_matrix

    #
    def get_demand_matrices(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix, self.filter_demand_matrix, self.ofmap_demand_matrix

    #
    def get_avg_mapping_efficiency(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.mapping_efficiency_per_fold)
        num = len(self.mapping_efficiency_per_fold)

        avg_mapping_eff = agg / num

        return avg_mapping_eff

    #
    def get_avg_compute_utilization(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)

        avg_compute_util = agg / num

        return avg_compute_util

    #
    def get_ifmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ifmap_reads

    #
    def get_filter_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.filter_reads

    #
    def get_ofmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ofmap_writes

#
# def skew_matrix(input_matrix_np):
#     rows, cols = input_matrix_np.shape

#     out_matrix_np = np.full((rows + cols - 1, cols), -1, dtype=input_matrix_np.dtype)

#     for c in range(cols):
#         out_matrix_np[c:c + rows, c] = input_matrix_np[:, c]

#     return out_matrix_np


def skew_matrix(input_matrix_np):
    """
    Skews the input matrix to simulate systolic pipeline fill.
    
    Parameters:
    -----------
    input_matrix_np : np.ndarray
        Input matrix to be skewed (2D numpy array)
        
    Returns:
    --------
    np.ndarray
        Skewed output matrix with diagonal elements
        
    Notes:
    ------
    - Output matrix dimensions: (rows + cols - 1) x cols
    - Empty positions filled with -1
    - Maintains input data type
    """
    # Get input matrix dimensions
    rows, cols = input_matrix_np.shape
    
    # Initialize output matrix with padding
    out_matrix_np = -np.ones((rows + cols - 1, cols),
                            dtype=input_matrix_np.dtype)
    
    # Perform skewing operation
    for c in range(cols):
        out_matrix_np[c:c+rows, c] = input_matrix_np[:, c]
    
    return out_matrix_np
