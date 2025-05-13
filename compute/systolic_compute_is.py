import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg


class systolic_compute_is:
    def __init__(self):

        # Params set by user
        self.config = cfg()

        self.ifmap_op_mat = np.zeros((1, 1))
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

    #     self.ifmap_op_mat_trans = np.transpose(self.ifmap_op_mat)

    #     ifmap_col = self.ifmap_op_mat.shape[1]
    #     filter_row= self.filter_op_mat.shape[0]

    #     assert ifmap_col == filter_row, "Dimension mismatch between operands"

    #     self.Sr = self.ifmap_op_mat.shape[1]
    #     self.Sc = self.ifmap_op_mat.shape[0]
    #     self.T = self.filter_op_mat.shape[1]

    #     self.arr_row, self.arr_col = self.config.get_array_dims()

    #     self.row_fold = math.ceil(self.Sr / self.arr_row)
    #     self.col_fold = math.ceil(self.Sc / self.arr_col)

    #     self.params_set_flag = True

    def set_params(self, config_obj, ifmap_op_mat, ofmap_op_mat, filter_op_mat):
        """
        Configures the systolic compute instance with operational parameters.
        
        Args:
            config_obj (cfg): Configuration object with array specifications
            ifmap_op_mat (ndarray): IFMAP operand matrix (2D)
            ofmap_op_mat (ndarray): OFMAP operand matrix (2D)
            filter_op_mat (ndarray): Filter operand matrix (2D)
            
        Returns:
            None
            
        Raises:
            AssertionError: If matrix dimensions are incompatible
        """
        # 1. Store configuration and matrices
        self.config = config_obj
        self.ifmap_op_mat = ifmap_op_mat
        self.filter_op_mat = filter_op_mat
        self.ofmap_op_mat = ofmap_op_mat
        
        # Create transposed IFMAP matrix
        self.ifmap_op_mat_trans = self.ifmap_op_mat.T
        
        # 2. Validate matrix dimensions
        ifmap_cols = self.ifmap_op_mat.shape[1]
        filter_rows = self.filter_op_mat.shape[0]
        assert ifmap_cols == filter_rows, \
            f"Dimension mismatch: IFMAP cols ({ifmap_cols}) != Filter rows ({filter_rows})"
        
        # 3. Calculate derived parameters
        self.Sr = self.ifmap_op_mat.shape[1]  # IFMAP columns
        self.Sc = self.ifmap_op_mat.shape[0]  # IFMAP rows
        self.T = self.filter_op_mat.shape[1]  # Filter columns
        
        # Get array dimensions from config
        self.arr_row, self.arr_col = self.config.get_array_dims()
        
        # Compute folding factors
        self.row_fold = int(np.ceil(self.Sc / self.arr_row))
        self.col_fold = int(np.ceil(self.Sr / self.arr_col))
        
        # Mark parameters as set
        self.params_set_flag = True

    #
    def create_prefetch_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_prefetch_mat()
        self.create_filter_prefetch_mat()

        self.prefetch_mat_ready_flag = True

    #
    # def create_ifmap_prefetch_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     for fc in range(self.col_fold):
    #         start_col_idx = fc * self.arr_col
    #         end_col_idx = min(start_col_idx + self.arr_col, self.Sc)

    #         delta = self.arr_col - (end_col_idx - start_col_idx)

    #         this_fold_prefetch = self.ifmap_op_mat_trans[:,start_col_idx: end_col_idx]

    #         #If there is under utilization, fill them with null requests
    #         if delta > 0:
    #             null_req_mat = np.ones((self.Sr, delta)) * -1
    #             this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

    #         if fc == 0:
    #             self.ifmap_prefetch_matrix = this_fold_prefetch
    #         else:
    #             self.ifmap_prefetch_matrix = np.concatenate((self.ifmap_prefetch_matrix, this_fold_prefetch), axis=0)

    # def create_ifmap_prefetch_mat(self):
    #     """
    #     Creates IFMAP prefetch matrix accounting for column folding.
        
    #     Returns:
    #         None (modifies self.ifmap_prefetch_matrix in-place)
            
    #     Raises:
    #         AssertionError: If parameters are not set
    #     """
    #     # 1. Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first."
        
    #     # 2. Process each column fold
    #     for fc in range(self.col_fold):
    #         # Calculate column bounds
    #         start_col = fc * self.arr_col
    #         end_col = (fc + 1) * self.arr_col
    #         delta = max(0, end_col - self.Sc)  # Underutilization
            
    #         # Extract current fold
    #         fold_mat = self.ifmap_op_mat_trans[:, start_col:min(end_col, self.Sc)]
            
    #         # Handle underutilization
    #         if delta > 0:
    #             null_mat = -np.ones((fold_mat.shape[0], delta), dtype=int)
    #             fold_mat = np.concatenate((fold_mat, null_mat), axis=1)
            
    #         # 3. Build final matrix
    #         if fc == 0:
    #             self.ifmap_prefetch_matrix = fold_mat
    #         else:
    #             self.ifmap_prefetch_matrix = np.concatenate(
    #                 (self.ifmap_prefetch_matrix, fold_mat),
    #                 axis=0
    #             )

    def create_ifmap_prefetch_mat(self):
        """
        Creates IFMAP prefetch matrix accounting for column folding with robust dimension handling.
        
        Returns:
            None (modifies self.ifmap_prefetch_matrix in-place)
            
        Raises:
            AssertionError: If parameters are not set
        """
        # 1. Parameter validation
        assert self.params_set_flag, "Parameters not set. Call set_params() first."
        assert hasattr(self, 'ifmap_op_mat_trans'), "IFMAP operand matrix not initialized"
        
        # 2. Initialize with empty list to collect folds
        fold_matrices = []
        max_cols = 0  # Track maximum columns across folds
        
        # 3. Process each column fold
        for fc in range(self.col_fold):
            # Calculate column bounds
            start_col = fc * self.arr_col
            end_col = (fc + 1) * self.arr_col
            current_cols = min(end_col, self.Sc) - start_col
            
            # Extract current fold
            fold_mat = self.ifmap_op_mat_trans[:, start_col:min(end_col, self.Sc)]
            
            # Handle underutilization
            if current_cols < self.arr_col:
                null_mat = -np.ones((fold_mat.shape[0], self.arr_col - current_cols), dtype=int)
                fold_mat = np.concatenate((fold_mat, null_mat), axis=1)
            
            # Track maximum columns
            max_cols = max(max_cols, fold_mat.shape[1])
            fold_matrices.append(fold_mat)
        
        # 4. Standardize all folds to same column dimension
        for i in range(len(fold_matrices)):
            if fold_matrices[i].shape[1] < max_cols:
                pad_cols = max_cols - fold_matrices[i].shape[1]
                fold_matrices[i] = np.pad(
                    fold_matrices[i],
                    ((0,0), (0,pad_cols)),
                    'constant',
                    constant_values=-1
                )
        
        # 5. Build final matrix
        if fold_matrices:
            self.ifmap_prefetch_matrix = np.concatenate(fold_matrices, axis=0)
        else:
            self.ifmap_prefetch_matrix = np.empty((0, self.arr_col), dtype=int)

        # Note: ISSUE #15: no skewing happens in the IFMAP for IS so this issue does not apply.

    #
    # def create_filter_prefetch_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     for fr in range(self.row_fold):
    #         row_start_id = fr * self.arr_row
    #         row_end_id = min(row_start_id + self.arr_row, self.Sr)

    #         delta = self.arr_row - (row_end_id - row_start_id)

    #         this_fold_prefetch = self.filter_op_mat[row_start_id:row_end_id, :]
    #         this_fold_prefetch = np.transpose(this_fold_prefetch)

    #         if delta > 0:
    #             null_req_mat = np.ones((self.T, delta)) * -1
    #             this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

    #         if fr == 0:
    #             self.filter_prefetch_matrix = this_fold_prefetch
    #         else:
    #             self.filter_prefetch_matrix = np.concatenate((self.filter_prefetch_matrix, this_fold_prefetch), axis=0)

    #     # Fixing ISSUE #15, #16
    #     # Roll out the matrices along the diagonal to account for temporal locality when there is a skew in demand

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
        Creates and temporally reorganizes filter prefetch matrix.
        
        Returns:
            None (modifies self.filter_prefetch_matrix in-place)
            
        Raises:
            AssertionError: If parameters are not set
        """
        # 1. Parameter validation
        assert self.params_set_flag, "Parameters not set. Call set_params() first."
        
        # 2. Process each row fold
        for fr in range(self.row_fold):
            # Calculate row bounds
            start_row = fr * self.arr_row
            end_row = (fr + 1) * self.arr_row
            delta = max(0, end_row - self.Sr)  # Underutilization
            
            # Extract and transpose current fold
            fold_mat = self.filter_op_mat[start_row:min(end_row, self.Sr), :].T
            
            # Handle underutilization
            if delta > 0:
                null_mat = -np.ones((delta, fold_mat.shape[1]), dtype=int)
                fold_mat = np.concatenate((fold_mat, null_mat), axis=0)
            
            # Build final folded matrix
            if fr == 0:
                self.filter_prefetch_matrix = fold_mat
            else:
                self.filter_prefetch_matrix = np.concatenate(
                    (self.filter_prefetch_matrix, fold_mat),
                    axis=0
                )

        # 3. Reorganize for temporal locality
        total_elements = self.filter_prefetch_matrix.size
        prefetches = np.full_like(self.filter_prefetch_matrix, -1)
        pbar = tqdm(total=total_elements, desc="Reorganizing filters")
        
        rows, cols = self.filter_prefetch_matrix.shape
        for diag in range(rows + cols - 1):
            # Get valid indices for current diagonal
            row_ids = np.arange(max(0, diag - cols + 1), min(diag + 1, rows))
            col_ids = diag - row_ids
            
            # Extract and place elements
            valid_mask = (row_ids < rows) & (col_ids < cols)
            if np.any(valid_mask):
                elements = self.filter_prefetch_matrix[row_ids[valid_mask], col_ids[valid_mask]]
                prefetches[row_ids[valid_mask], col_ids[valid_mask]] = elements
                pbar.update(len(elements))
        
        # 4. Finalize
        pbar.close()
        self.filter_prefetch_matrix = prefetches

    #
    def create_demand_matrices(self):
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_demand_mat()
        self.create_filter_demand_mat()
        self.create_ofmap_demand_mat()

        assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'IFMAP and Filter demands out of sync'
        assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'OFMAP and Filter demands out of sync'
        assert self.ifmap_demand_matrix.shape[1] == self.arr_col, 'IFMAP demands exceed the rows'
        assert self.filter_demand_matrix.shape[1] == self.arr_row,'Filter demands exceed the cols'
        assert self.ofmap_demand_matrix.shape[1] == self.arr_col, 'OFMAP demands exceed the cols'

        self.demand_mat_ready_flag = True

    #
    # def create_ifmap_demand_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     inter_fold_gap_suffix = self.arr_row + self.arr_col + self.T - 2
    #     inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_col)) * -1

    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             row_start_id = fr * self.arr_row
    #             row_end_idx = min(row_start_id + self.arr_row, self.Sr)
    #             row_delta = self.arr_row - (row_end_idx - row_start_id)

    #             col_start_id = fc * self.arr_col
    #             col_end_idx = min(col_start_id + self.arr_col, self.Sc)
    #             col_delta = self.arr_col - (col_end_idx - col_start_id)

    #             # Indexing the cols with row start and row end idx are correct
    #             # See the comment on ifmap_prefetch generation
    #             this_fold_demand = self.ifmap_op_mat_trans[row_start_id:row_end_idx, col_start_id: col_end_idx]
    #             self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

    #             # Take into account under utilization
    #             if col_delta > 0:
    #                 null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

    #             if row_delta > 0:
    #                 null_req_mat = np.ones((row_delta, self.arr_col)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

    #             # The IFMAP elems are needed to be filled in reverse order to ensure that
    #             # top element is pushed in last to maintain alignment with the input elements
    #             this_fold_demand = np.flip(this_fold_demand, 0)

    #             # Account for the cycles for partial sum generation and accumulation
    #             this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)

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

    #             if fr == 0 and fc == 0:
    #                 self.ifmap_demand_matrix = this_fold_demand
    #             else:
    #                 self.ifmap_demand_matrix = np.concatenate((self.ifmap_demand_matrix, this_fold_demand), axis=0)

        # Skew is not needed in IFMAP for IS

    def create_ifmap_demand_mat(self):
        """
        Creates IFMAP demand matrix accounting for folding and utilization.
        
        Returns:
            None (modifies self.ifmap_demand_matrix and metrics in-place)
            
        Raises:
            AssertionError: If parameters are not set
        """
        # 1. Parameter validation
        assert self.params_set_flag, "Parameters not set. Call set_params() first."
        
        # 2. Initialize inter-fold gap
        inter_fold_gap_suffix = self.arr_row * self.arr_col * self.T
        inter_fold_gap_suffix_mat = -np.ones((inter_fold_gap_suffix, 
                                            self.arr_col), dtype=int)
        
        # 3. Process each fold combination
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                # Calculate fold boundaries
                start_row = fr * self.arr_row
                end_row = (fr + 1) * self.arr_row
                start_col = fc * self.arr_col
                end_col = (fc + 1) * self.arr_col
                
                # Calculate underutilization
                row_delta = max(0, end_row - self.Sc)
                col_delta = max(0, end_col - self.Sr)
                
                # Extract current demand matrix
                demand_mat = self.ifmap_op_mat_trans[
                    start_col:min(end_col, self.Sr),
                    start_row:min(end_row, self.Sc)
                ]
                self.ifmap_reads += demand_mat.size
                
                # Handle column underutilization
                if col_delta > 0:
                    null_cols = -np.ones((col_delta, demand_mat.shape[1]), dtype=int)
                    demand_mat = np.concatenate((demand_mat, null_cols), axis=0)
                
                # Handle row underutilization
                if row_delta > 0:
                    null_rows = -np.ones((demand_mat.shape[0], row_delta), dtype=int)
                    demand_mat = np.concatenate((demand_mat, null_rows), axis=1)
                
                # Reverse rows for alignment
                demand_mat = np.flipud(demand_mat)
                
                # Add inter-fold gap
                demand_mat = np.concatenate(
                    (demand_mat, inter_fold_gap_suffix_mat), 
                    axis=0
                )
                
                # Calculate metrics
                row_used = self.arr_row - row_delta
                col_used = self.arr_col - col_delta
                mac_used = row_used * col_used
                
                mapping_eff_this_fold = mac_used / (self.arr_row * self.arr_col)
                cycles_this_fold = demand_mat.shape[0]
                compute_cycles_this_fold = min(self.Sc - fr * self.arr_row, 
                                            self.arr_row)
                compute_util_this_fold = mac_used * compute_cycles_this_fold / \
                                    (self.arr_row * self.arr_col * cycles_this_fold)
                
                self.mapping_efficiency_per_fold.append(mapping_eff_this_fold)
                self.compute_utility_per_fold.append(compute_util_this_fold)
                
                # Build final matrix
                if fr == 0 and fc == 0:
                    self.ifmap_demand_matrix = demand_mat
                else:
                    self.ifmap_demand_matrix = np.concatenate(
                        (self.ifmap_demand_matrix, demand_mat),
                        axis=0
                    )

    #
    # def create_filter_demand_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     inter_fold_gap_prefix = self.arr_row
    #     inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_row)) * -1

    #     inter_fold_gap_suffix = self.arr_col - 1
    #     inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             row_start_id = fr * self.arr_row
    #             row_end_idx = min(row_start_id + self.arr_row, self.Sr)
    #             delta = self.arr_row - (row_end_idx - row_start_id)

    #             # Indexing the cols with row start and row end idx are correct
    #             # See the comment on ifmap_prefetch generation
    #             this_fold_demand = self.filter_op_mat[row_start_id: row_end_idx, :]
    #             this_fold_demand = np.transpose(this_fold_demand)
    #             self.filter_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

    #             # Take into account under utilization
    #             if delta > 0:
    #                 null_req_mat = np.ones((self.T, delta)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

    #             # Account for the cycles for weights to load
    #             this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand), axis=0)

    #             # Account for the cycles for final output to drain out
    #             this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)

    #             # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
    #             this_fold_demand = skew_matrix(this_fold_demand)

    #             if fr == 0 and fc == 0:
    #                 self.filter_demand_matrix = this_fold_demand
    #             else:
    #                 self.filter_demand_matrix = np.concatenate((self.filter_demand_matrix, this_fold_demand), axis=0)
    # END of filter demand generation

    def create_filter_demand_mat(self):
        """
        Creates filter demand matrix accounting for folding and systolic timing.
        
        Returns:
            None (modifies self.filter_demand_matrix in-place)
            
        Raises:
            AssertionError: If parameters are not set
        """
        # 1. Parameter validation
        assert self.params_set_flag, "Parameters not set. Call set_params() first."
        
        # 2. Initialize inter-fold gaps
        inter_fold_gap_prefix = self.arr_col * self.T
        inter_fold_gap_suffix = self.arr_row * self.arr_col * self.T
        
        gap_prefix_mat = -np.ones((inter_fold_gap_prefix, self.arr_col), dtype=int)
        gap_suffix_mat = -np.ones((inter_fold_gap_suffix, self.arr_col), dtype=int)
        
        # 3. Process each fold combination
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                # Calculate fold boundaries
                start_row = fr * self.arr_row
                end_row = (fr + 1) * self.arr_row
                delta = max(0, end_row - self.Sr)  # Underutilization
                
                # Extract and transpose current demand matrix
                demand_mat = self.filter_op_mat[start_row:min(end_row, self.Sr), :].T
                self.filter_reads += demand_mat.size
                
                # Handle underutilization
                if delta > 0:
                    null_cols = -np.ones((demand_mat.shape[0], delta), dtype=int)
                    demand_mat = np.concatenate((demand_mat, null_cols), axis=1)
                
                # Add inter-fold gaps
                demand_mat = np.concatenate(
                    (gap_prefix_mat, demand_mat, gap_suffix_mat),
                    axis=0
                )
                
                # Skew matrix for systolic timing
                rows, cols = demand_mat.shape
                skewed_mat = np.full_like(demand_mat, -1)
                for c in range(cols):
                    offset = min(c, rows - 1)
                    skewed_mat[offset:, c] = demand_mat[:rows-offset, c]
                
                # Build final matrix
                if fr == 0 and fc == 0:
                    self.filter_demand_matrix = skewed_mat
                else:
                    self.filter_demand_matrix = np.concatenate(
                        (self.filter_demand_matrix, skewed_mat),
                        axis=0
                    )

    #
    def create_ofmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = 2 * self.arr_row - 1
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_col)) * -1

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = self.ofmap_op_mat[col_start_id: col_end_idx, :]
                this_fold_demand = np.transpose(this_fold_demand)
                self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Adding null requests when there is under utilization ie. no mapping along a few rows or cols
                if col_delta > 0:
                    null_req_mat = np.ones((self.T, col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # Now add the prefix matrix
                # These are the null demands to account for when the operands are streamed in
                # and the OFMAPS are not ready
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand), axis=0)

                # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                if fr == 0 and fc == 0:
                    self.ofmap_demand_matrix = this_fold_demand
                else:
                    self.ofmap_demand_matrix = np.concatenate((self.ofmap_demand_matrix, this_fold_demand), axis=0)
    # def create_ofmap_demand_mat(self):
    #     """
    #     Creates OFMAP demand matrix accounting for folding and systolic timing.
        
    #     Returns:
    #         None (modifies self.ofmap_demand_matrix in-place)
            
    #     Raises:
    #         AssertionError: If parameters are not set
    #     """
    #     # 1. Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first."
    #     assert hasattr(self, 'ofmap_op_mat'), "OFMAP operand matrix not initialized"
        
    #     # 2. Initialize matrix collection
    #     demand_matrices = []
        
    #     # 3. Process each fold combination
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate fold boundaries
    #             start_col = fc * self.arr_col
    #             end_col = (fc + 1) * self.arr_col
    #             valid_cols = min(end_col, self.Sc) - start_col
                
    #             # Extract and transpose current demand matrix
    #             demand_mat = self.ofmap_op_mat[:, start_col:start_col + valid_cols].T
    #             self.ofmap_writes += demand_mat.size
                
    #             # Handle column underutilization
    #             if valid_cols < self.arr_col:
    #                 null_cols = -np.ones((demand_mat.shape[0], self.arr_col - valid_cols), dtype=int)
    #                 demand_mat = np.concatenate((demand_mat, null_cols), axis=1)
                
    #             # Add inter-fold gap prefix
    #             gap_prefix_mat = -np.ones((self.arr_col * self.T, self.arr_col), dtype=int)
    #             demand_mat = np.concatenate((gap_prefix_mat, demand_mat), axis=0)
                
    #             # Skew matrix for systolic timing
    #             skewed_mat = skew_matrix(demand_mat)
                
    #             # Collect matrices for final concatenation
    #             demand_matrices.append(skewed_mat)
        
    #     # 4. Build final matrix
    #     if demand_matrices:
    #         # Find maximum width to ensure consistent dimensions
    #         max_cols = max(mat.shape[1] for mat in demand_matrices)
            
    #         # Standardize all matrices to same width
    #         standardized_mats = []
    #         for mat in demand_matrices:
    #             if mat.shape[1] < max_cols:
    #                 pad_cols = max_cols - mat.shape[1]
    #                 mat = np.pad(mat, ((0,0), (0,pad_cols)), 'constant', constant_values=-1)
    #             standardized_mats.append(mat)
            
    #         self.ofmap_demand_matrix = np.concatenate(standardized_mats, axis=0)
    #     else:
    #         self.ofmap_demand_matrix = np.empty((0, self.arr_col), dtype=int)
    # END of OFMAP demand generation

    # def create_ofmap_demand_mat(self):
    #     """
    #     Creates OFMAP demand matrix accounting for folding and systolic timing.
        
    #     Returns:
    #         None (modifies self.ofmap_demand_matrix in-place)
            
    #     Raises:
    #         AssertionError: If parameters are not set
    #     """
    #     # 1. Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first."
    #     assert hasattr(self, 'ofmap_op_mat'), "OFMAP operand matrix not initialized"
        
    #     # 2. Initialize matrix collection
    #     demand_matrices = []
    #     target_cols = self.arr_col  # Target column size for all matrices
        
    #     # 3. Process each fold combination
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate fold boundaries
    #             start_col = fc * target_cols
    #             end_col = (fc + 1) * target_cols
    #             valid_cols = min(end_col, self.Sc) - start_col
                
    #             # Extract and transpose current demand matrix
    #             demand_mat = self.ofmap_op_mat[:, start_col:start_col + valid_cols].T
    #             self.ofmap_writes += demand_mat.size
                
    #             # Handle column underutilization by padding to target_cols
    #             if demand_mat.shape[1] < target_cols:
    #                 pad_cols = target_cols - demand_mat.shape[1]
    #                 demand_mat = np.pad(demand_mat, ((0,0), (0,pad_cols)), 
    #                                 'constant', constant_values=-1)
                
    #             # Create gap prefix matrix with same number of columns
    #             gap_prefix_mat = -np.ones((self.arr_col * self.T, target_cols), dtype=int)
                
    #             # Concatenate matrices (now with matching column dimensions)
    #             demand_mat = np.concatenate((gap_prefix_mat, demand_mat), axis=0)
                
    #             # Skew matrix for systolic timing
    #             skewed_mat = skew_matrix(demand_mat)
                
    #             # Collect matrices for final concatenation
    #             demand_matrices.append(skewed_mat)
        
    #     # 4. Build final matrix
    #     if demand_matrices:
    #         self.ofmap_demand_matrix = np.concatenate(demand_matrices, axis=0)
    #     else:
    #         self.ofmap_demand_mat = np.empty((0, target_cols), dtype=int)

    # def create_ofmap_demand_mat(self):
    #     """
    #     Creates the OFMAP demand matrix based on layer configuration and folding factors,
    #     accounting for under-utilization and temporal locality.
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Run set_params() first."
        
    #     # Calculate inter-fold gap prefix
    #     inter_fold_gap_prefix = self.arr_row - 1
    #     inter_fold_gap_prefix_mat = -1 * np.ones((inter_fold_gap_prefix, self.arr_col))
        
    #     # Initialize OFMAP writes counter
    #     self.ofmap_writes = 0
        
    #     # Process each fold
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate column indices for current fold
    #             col_start = fc * self.arr_col
    #             col_end = (fc + 1) * self.arr_col
    #             col_delta = col_end - self.Sc if col_end > self.Sc else 0
                
    #             # Get current fold's demand matrix
    #             curr_fold_mat = self.ofmap_op_mat[fr*self.arr_row:(fr+1)*self.arr_row, 
    #                                             col_start:col_end-col_delta]
    #             curr_fold_mat = curr_fold_mat.T  # Transpose
                
    #             # Update writes counter
    #             self.ofmap_writes += curr_fold_mat.size
                
    #             # Handle under-utilization
    #             if col_delta > 0:
    #                 null_mat = -1 * np.ones((curr_fold_mat.shape[0], col_delta))
    #                 curr_fold_mat = np.concatenate((curr_fold_mat, null_mat), axis=1)
                
    #             # Add inter-fold gap prefix
    #             curr_fold_mat = np.concatenate((inter_fold_gap_prefix_mat, curr_fold_mat), axis=0)
                
    #             # Skew matrix for systolic pipeline
    #             curr_fold_mat = skew_matrix(curr_fold_mat)
                
    #             # Add to final demand matrix
    #             if fr == 0 and fc == 0:
    #                 self.ofmap_demand_matrix = curr_fold_mat
    #             else:
    #                 self.ofmap_demand_matrix = np.concatenate((self.ofmap_demand_matrix, curr_fold_mat), axis=0)
        
    #     # Set flag indicating demand matrix is ready
    #     self.demand_mat_ready_flag = True
    # def create_ofmap_demand_mat(self):
    #     """
    #     Creates the OFMAP demand matrix based on layer configuration and folding factors,
    #     accounting for under-utilization and temporal locality.
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Run set_params() first."
        
    #     # Calculate inter-fold gap prefix
    #     inter_fold_gap_prefix = self.arr_row - 1
    #     inter_fold_gap_prefix_mat = -1 * np.ones((inter_fold_gap_prefix, self.arr_col))
        
    #     # Initialize OFMAP writes counter
    #     self.ofmap_writes = 0
        
    #     # Process each fold
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate column indices for current fold
    #             col_start = fc * self.arr_col
    #             col_end = (fc + 1) * self.arr_col
    #             col_delta = col_end - self.Sc if col_end > self.Sc else 0
                
    #             # Get current fold's demand matrix
    #             curr_fold_mat = self.ofmap_op_mat[fr*self.arr_row:(fr+1)*self.arr_row, 
    #                                             col_start:col_end-col_delta]
    #             curr_fold_mat = curr_fold_mat.T  # Transpose
                
    #             # Update writes counter
    #             self.ofmap_writes += curr_fold_mat.size
                
    #             # Handle under-utilization
    #             if col_delta > 0:
    #                 null_mat = -1 * np.ones((curr_fold_mat.shape[0], col_delta))
    #                 curr_fold_mat = np.concatenate((curr_fold_mat, null_mat), axis=1)
                
    #             # Ensure matrices have same width before concatenation
    #             if inter_fold_gap_prefix_mat.shape[1] != curr_fold_mat.shape[1]:
    #                 # Adjust inter_fold_gap_prefix_mat width to match current fold
    #                 adjusted_gap_mat = -1 * np.ones((inter_fold_gap_prefix, curr_fold_mat.shape[1]))
    #                 curr_fold_mat = np.concatenate((adjusted_gap_mat, curr_fold_mat), axis=0)
    #             else:
    #                 # Original concatenation
    #                 curr_fold_mat = np.concatenate((inter_fold_gap_prefix_mat, curr_fold_mat), axis=0)
                
    #             # Skew matrix for systolic pipeline
    #             curr_fold_mat = skew_matrix(curr_fold_mat)
                
    #             # Add to final demand matrix
    #             if fr == 0 and fc == 0:
    #                 self.ofmap_demand_matrix = curr_fold_mat
    #             else:
    #                 self.ofmap_demand_matrix = np.concatenate((self.ofmap_demand_matrix, curr_fold_mat), axis=0)
        
    #     # Set flag indicating demand matrix is ready
    #     self.demand_mat_ready_flag = True


    # def create_ofmap_demand_mat(self):
    #     """
    #     Creates the OFMAP demand matrix based on layer configuration and folding factors,
    #     accounting for under-utilization and temporal locality.
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Run set_params() first."
        
    #     # Calculate inter-fold gap prefix
    #     inter_fold_gap_prefix = self.arr_row - 1
    #     inter_fold_gap_prefix_mat = -1 * np.ones((inter_fold_gap_prefix, self.arr_col))
        
    #     # Initialize OFMAP writes counter
    #     self.ofmap_writes = 0
    #     self.ofmap_demand_matrix = None
        
    #     # Process each fold
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate column indices for current fold
    #             col_start = fc * self.arr_col
    #             col_end = min((fc + 1) * self.arr_col, self.Sc)  # Ensure we don't exceed total columns
                
    #             # Get current fold's demand matrix
    #             curr_fold_mat = self.ofmap_op_mat[fr*self.arr_row:(fr+1)*self.arr_row, 
    #                                             col_start:col_end]
                
    #             # Transpose and ensure correct dimensions
    #             curr_fold_mat = curr_fold_mat.T
    #             if curr_fold_mat.shape[1] < self.arr_col:
    #                 # Pad with -1 if underutilized
    #                 pad_width = self.arr_col - curr_fold_mat.shape[1]
    #                 curr_fold_mat = np.pad(curr_fold_mat, 
    #                                     ((0, 0), (0, pad_width)), 
    #                                     constant_values=-1)
                
    #             # Update writes counter (only count actual data, not padding)
    #             self.ofmap_writes += (curr_fold_mat != -1).sum()
                
    #             # Add inter-fold gap prefix
    #             curr_fold_mat = np.concatenate((inter_fold_gap_prefix_mat, curr_fold_mat), axis=0)
                
    #             # Skew matrix for systolic pipeline
    #             curr_fold_mat = skew_matrix(curr_fold_mat)
                
    #             # Add to final demand matrix
    #             if self.ofmap_demand_matrix is None:
    #                 self.ofmap_demand_matrix = curr_fold_mat
    #             else:
    #                 # Ensure consistent width before concatenation
    #                 if self.ofmap_demand_matrix.shape[1] != curr_fold_mat.shape[1]:
    #                     # Find max width and pad both matrices
    #                     max_width = max(self.ofmap_demand_matrix.shape[1], curr_fold_mat.shape[1])
    #                     if self.ofmap_demand_matrix.shape[1] < max_width:
    #                         pad_width = max_width - self.ofmap_demand_matrix.shape[1]
    #                         self.ofmap_demand_matrix = np.pad(self.ofmap_demand_matrix,
    #                                                         ((0, 0), (0, pad_width)),
    #                                                         constant_values=-1)
    #                     if curr_fold_mat.shape[1] < max_width:
    #                         pad_width = max_width - curr_fold_mat.shape[1]
    #                         curr_fold_mat = np.pad(curr_fold_mat,
    #                                             ((0, 0), (0, pad_width)),
    #                                             constant_values=-1)
    #                 self.ofmap_demand_matrix = np.concatenate((self.ofmap_demand_matrix, curr_fold_mat), axis=0)
        
    #     # Set flag indicating demand matrix is ready
    #     self.demand_mat_ready_flag = True

    # def create_ofmap_demand_mat(self):
    #     """
    #     Creates the OFMAP demand matrix synchronized with IFMAP and Filter demand matrices
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Run set_params() first."
        
    #     # Get inter-fold gap parameters
    #     inter_fold_gap = self.arr_row - 1
    #     inter_fold_gap_mat = -1 * np.ones((inter_fold_gap, self.arr_col))
        
    #     # Initialize
    #     self.ofmap_writes = 0
    #     self.ofmap_demand_matrix = np.empty((0, self.arr_col))  # Start with empty matrix
        
    #     # Process folds
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate column range
    #             col_start = fc * self.arr_col
    #             col_end = min((fc + 1) * self.arr_col, self.Sc)
                
    #             # Get current fold data
    #             curr_fold_data = self.ofmap_op_mat[fr*self.arr_row:(fr+1)*self.arr_row, 
    #                                             col_start:col_end]
    #             curr_fold_mat = curr_fold_data.T
                
    #             # Pad if needed
    #             if curr_fold_mat.shape[1] < self.arr_col:
    #                 pad_width = self.arr_col - curr_fold_mat.shape[1]
    #                 curr_fold_mat = np.pad(curr_fold_mat, 
    #                                     ((0, 0), (0, pad_width)), 
    #                                     constant_values=-1)
                
    #             # Add inter-fold gap
    #             curr_fold_mat = np.vstack([inter_fold_gap_mat, curr_fold_mat])
                
    #             # Skew for pipeline
    #             curr_fold_mat = skew_matrix(curr_fold_mat)
                
    #             # Synchronize with other matrices
    #             current_cycles = curr_fold_mat.shape[0]
    #             expected_cycles = self.ifmap_demand_matrix.shape[0] - self.ofmap_demand_matrix.shape[0]
                
    #             if current_cycles > expected_cycles:
    #                 # Trim extra cycles (shouldn't normally happen)
    #                 curr_fold_mat = curr_fold_mat[:expected_cycles, :]
    #             elif current_cycles < expected_cycles:
    #                 # Pad with null cycles
    #                 pad_cycles = expected_cycles - current_cycles
    #                 curr_fold_mat = np.vstack([curr_fold_mat, 
    #                                         -1 * np.ones((pad_cycles, self.arr_col))])
                
    #             # Update writes counter (only actual data)
    #             self.ofmap_writes += np.count_nonzero(curr_fold_mat != -1)
                
    #             # Concatenate
    #             self.ofmap_demand_matrix = np.vstack([self.ofmap_demand_matrix, 
    #                                                 curr_fold_mat])
        
    #     # Final synchronization check
    #     assert self.ofmap_demand_matrix.shape[0] == self.ifmap_demand_matrix.shape[0], \
    #         "OFMAP demand out of sync with IFMAP demand"
    #     assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], \
    #         "OFMAP demand out of sync with Filter demand"
        
    #     self.demand_mat_ready_flag = True

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
    Skews a matrix to simulate systolic array pipeline fill.
    
    Args:
        input_matrix_np (ndarray): Input matrix to be skewed (2D)
        
    Returns:
        ndarray: Skewed output matrix with dimensions (rows+cols-1, cols)
                 Filled with -1 in empty positions
    """
    # Get matrix dimensions
    rows, cols = input_matrix_np.shape
    
    # Initialize output matrix
    out_matrix_np = -np.ones((rows + cols - 1, cols), dtype=input_matrix_np.dtype)
    
    # Skew each column
    for c in range(cols):
        # Calculate valid rows for this column
        valid_rows = min(rows, rows + cols - 1 - c)
        
        # Place elements on the diagonal
        out_matrix_np[c:c+rows, c] = input_matrix_np[:, c]
    
    return out_matrix_np