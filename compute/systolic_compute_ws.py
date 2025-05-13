import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg


class systolic_compute_ws:
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
    def set_params(self,
                   config_obj=cfg(),
                   ifmap_op_mat = np.zeros((1,1)),
                   ofmap_op_mat = np.zeros((1,1)),
                   filter_op_mat = np.zeros((1,1))
                ):

        self.config = config_obj
        self.ifmap_op_mat = ifmap_op_mat
        self.filter_op_mat = filter_op_mat
        self.ofmap_op_mat = ofmap_op_mat

        ifmap_col = self.ifmap_op_mat.shape[1]
        filter_row= self.filter_op_mat.shape[0]

        assert ifmap_col == filter_row, "Dimension mismatch between operands"

        self.Sr = self.ifmap_op_mat.shape[1]
        self.Sc = self.filter_op_mat.shape[1]
        self.T = self.ifmap_op_mat.shape[0]

        self.arr_row, self.arr_col = self.config.get_array_dims()

        self.row_fold = math.ceil(self.Sr / self.arr_row)
        self.col_fold = math.ceil(self.Sc / self.arr_col)

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

    #     for fr in range(self.row_fold):
    #         start_col_idx = fr * self.arr_row
    #         end_col_idx = min(start_col_idx + self.arr_row, self.Sr)

    #         delta = self.arr_row - (end_col_idx - start_col_idx)

    #         this_fold_prefetch = self.ifmap_op_mat[:,start_col_idx: end_col_idx]

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

    #     M, N = self.ifmap_prefetch_matrix.shape
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

    #             elem = self.ifmap_prefetch_matrix[row_id][col_id]
    #             prefetches[0, idx] = elem
    #             idx += 1
    #             pbar.update(1)

    #     pbar.close()
    #     self.ifmap_prefetch_matrix = prefetches

    def create_ifmap_prefetch_mat(self):    
        """
        Creates the IFMAP prefetch matrix based on layer configuration and folding factors,
        then reorganizes it to account for temporal locality.
        
        Steps:
        1. Validate parameters are set
        2. Process each row fold
        3. Handle under-utilized folds
        4. Reorganize matrix diagonally for temporal locality
        """
        # Parameter validation
        assert self.params_set_flag, "Parameters not set. Call set_params() first."
        assert hasattr(self, 'ifmap_op_mat_trans'), "Transposed IFMAP matrix not initialized"
        assert self.row_fold > 0, "Row folding factor must be positive"
        
        # Initialize empty matrix
        rows, cols = self.ifmap_op_mat_trans.shape
        self.ifmap_prefetch_matrix = np.full((rows, cols), -1, dtype=int)
        
        # Process each row fold
        for fr in range(self.row_fold):
            # Calculate row indices
            start_row = fr * self.arr_row
            end_row = min((fr + 1) * self.arr_row, self.Sr)
            actual_rows = end_row - start_row
            
            # Extract current fold
            curr_fold = self.ifmap_op_mat_trans[start_row:end_row, :]
            
            # Handle under-utilization
            if actual_rows < self.arr_row:
                padding = np.full((self.arr_row - actual_rows, cols), -1, dtype=int)
                curr_fold = np.vstack((curr_fold, padding))
            
            # Concatenate folds
            dest_start = fr * self.arr_row
            dest_end = dest_start + self.arr_row
            self.ifmap_prefetch_matrix[dest_start:dest_end, :] = curr_fold
        
        # Reorganize for temporal locality
        reorganized = np.full_like(self.ifmap_prefetch_matrix, -1)
        total_elements = rows * cols
        progress = tqdm(total=total_elements, desc="Reorganizing IFMAP prefetch")
        
        # Diagonal traversal
        for d in range(rows + cols - 1):
            for i in range(max(0, d - cols + 1), min(rows, d + 1)):
                j = d - i
                if 0 <= j < cols:
                    reorganized[i, j] = self.ifmap_prefetch_matrix[i, j]
                    progress.update(1)
        
        progress.close()
        self.ifmap_prefetch_matrix = reorganized
        self.prefetch_mat_ready_flag = True



    # def create_ifmap_prefetch_mat(self):
    #     """
    #     Constructs IFMAP prefetch matrix with:
    #     - Row fold-aware processing
    #     - Under-utilization padding
    #     - Diagonal-based temporal optimization
        
    #     Raises:
    #         AssertionError: If parameters not initialized
    #     """
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Matrix Initialization ---
    #     self.ifmap_prefetch_matrix = np.empty((0, self.arr_row), dtype=np.int32)
        
    #     # === Fold Processing ==============================================
    #     for fr in range(self.row_fold):
    #         # Calculate row boundaries
    #         start_row = fr * self.arr_row
    #         end_row = min((fr + 1) * self.arr_row, self.Sr)
    #         delta = self.arr_row - (end_row - start_row)
            
    #         # Extract fold data (transposed for OS dataflow)
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
        
    #     # Diagonal traversal for temporal locality
    #     for diag in range(rows + cols - 1):
    #         # Determine traversal bounds
    #         i_start = max(0, diag - cols + 1)
    #         i_end = min(diag + 1, rows)
            
    #         for i in range(i_start, i_end):
    #             j = diag - i
    #             if 0 <= j < cols:
    #                 reorganized[i,j] = self.ifmap_prefetch_matrix[i,j]
        
    #     # --- Finalization ---
    #     self.ifmap_prefetch_matrix = reorganized
    #     self.prefetch_mat_ready_flag = True

    # def create_ifmap_prefetch_mat(self):
    #     """
    #     Constructs IFMAP prefetch matrix with:
    #     - Row fold-aware processing
    #     - Under-utilization padding
    #     - Diagonal-based temporal optimization
        
    #     Raises:
    #         AssertionError: If parameters not initialized
    #     """
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Matrix Initialization ---
    #     self.ifmap_prefetch_matrix = np.empty((0, self.arr_row), dtype=np.int32)
        
    #     # === Fold Processing ==============================================
    #     for fr in range(self.row_fold):
    #         # Calculate row boundaries
    #         start_row = fr * self.arr_row
    #         end_row = min((fr + 1) * self.arr_row, self.Sr)
    #         delta = self.arr_row - (end_row - start_row)
            
    #         # Extract fold data (transposed for OS dataflow)
    #         fold_data = self.ifmap_op_mat_trans[:, start_row:end_row]
            
    #         # Handle under-utilized folds
    #         if delta > 0:
    #             null_pad = -np.ones((fold_data.shape[0], delta), dtype=np.int32)
    #             fold_data = np.concatenate([fold_data, null_pad], axis=1)
            
    #         # Ensure consistent dimensions before concatenation
    #         if self.ifmap_prefetch_matrix.size > 0:
    #             if fold_data.shape[1] != self.ifmap_prefetch_matrix.shape[1]:
    #                 # Pad to match column dimension
    #                 pad_cols = self.ifmap_prefetch_matrix.shape[1] - fold_data.shape[1]
    #                 if pad_cols > 0:
    #                     fold_data = np.pad(fold_data, 
    #                                     ((0, 0), (0, pad_cols)), 
    #                                     mode='constant', 
    #                                     constant_values=-1)
            
    #         # Build complete prefetch matrix
    #         if fr == 0:
    #             self.ifmap_prefetch_matrix = fold_data
    #         else:
    #             self.ifmap_prefetch_matrix = np.vstack([self.ifmap_prefetch_matrix, fold_data])
        
    #     # === Temporal Reorganization =====================================
    #     rows, cols = self.ifmap_prefetch_matrix.shape
    #     reorganized = np.empty_like(self.ifmap_prefetch_matrix)
        
    #     # Diagonal traversal for temporal locality
    #     for diag in range(rows + cols - 1):
    #         # Determine traversal bounds
    #         i_start = max(0, diag - cols + 1)
    #         i_end = min(diag + 1, rows)
            
    #         for i in range(i_start, i_end):
    #             j = diag - i
    #             if 0 <= j < cols:
    #                 reorganized[i,j] = self.ifmap_prefetch_matrix[i,j]
        
    #     # --- Finalization ---
    #     self.ifmap_prefetch_matrix = reorganized
    #     self.prefetch_mat_ready_flag = True

    #
    # def create_filter_prefetch_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     for fc in range(self.col_fold):
    #         col_start_id = fc * self.arr_col
    #         col_end_id = min(col_start_id + self.arr_col, self.Sc)

    #         delta = self.arr_col - (col_end_id - col_start_id)

    #         this_fold_prefetch = self.filter_op_mat[:,col_start_id:col_end_id]

    #         if delta > 0:
    #             null_req_mat = np.ones((self.Sr, delta)) * -1
    #             this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

    #         if fc == 0:
    #             self.filter_prefetch_matrix = this_fold_prefetch
    #         else:
    #             self.filter_prefetch_matrix = np.concatenate((self.filter_prefetch_matrix, this_fold_prefetch), axis=0)


    def create_filter_prefetch_mat(self):
        """
        Constructs filter prefetch matrix with:
        - Column fold-aware processing
        - Under-utilization padding
        - Efficient matrix construction
        
        Raises:
            AssertionError: If parameters not initialized
        """
        
        # --- Parameter Validation ---
        assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
        # --- Matrix Initialization ---
        self.filter_prefetch_matrix = np.empty((0, self.arr_col), dtype=np.int32)
        
        # === Column Fold Processing ======================================
        for fc in range(self.col_fold):
            # Calculate column boundaries
            start_col = fc * self.arr_col
            end_col = min((fc + 1) * self.arr_col, self.Sc)
            delta = self.arr_col - (end_col - start_col)
            
            # Extract fold data from filter matrix
            fold_data = self.filter_op_mat[:, start_col:end_col]
            
            # Handle under-utilized folds
            if delta > 0:
                null_pad = -np.ones((self.T, delta), dtype=np.int32)
                fold_data = np.concatenate([fold_data, null_pad], axis=1)
            
            # Build complete prefetch matrix
            if fc == 0:
                self.filter_prefetch_matrix = fold_data
            else:
                self.filter_prefetch_matrix = np.vstack([self.filter_prefetch_matrix, fold_data])
        
        # --- Finalization ---
        self.prefetch_mat_ready_flag = True

        # Note: ISSUE #15: no skewing happens in the Filter for WS so this issue does not apply.

    #
    # def create_demand_matrices(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     self.create_ifmap_demand_mat()
    #     self.create_filter_demand_mat()
    #     self.create_ofmap_demand_mat()

    #     assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'IFMAP and Filter demands out of sync'
    #     assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], 'OFMAP and Filter demands out of sync'
    #     assert self.ifmap_demand_matrix.shape[1] == self.arr_row, 'IFMAP demands exceed the rows'
    #     assert self.filter_demand_matrix.shape[1] == self.arr_col,'Filter demands exceed the cols'
    #     assert self.ofmap_demand_matrix.shape[1] == self.arr_col, 'OFMAP demands exceed the cols'

    #     self.demand_mat_ready_flag = True



    #
    def create_ifmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = self.arr_row
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_row)) * -1

        inter_fold_gap_suffix = self.arr_col - 1

        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        ifmap_demand_matrix_list = []
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                col_start_id = fr * self.arr_row
                col_end_idx = min(col_start_id + self.arr_row, self.Sr)
                delta = self.arr_row - (col_end_idx - col_start_id)

                # Indexing the cols with row start and row end idx are correct
                # See the comment on ifmap_prefetch generation
                this_fold_demand = self.ifmap_op_mat[:,col_start_id: col_end_idx]
                self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((self.T, delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # Account for the cycles for weights to load
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand), axis=0)

                # Account for the cycles for final output to drain out
                this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat), axis=0)

                # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                ifmap_demand_matrix_list.append(this_fold_demand)
                #if fr == 0 and fc == 0:
                #    self.ifmap_demand_matrix = this_fold_demand
                #else:
                #    self.ifmap_demand_matrix = np.concatenate((self.ifmap_demand_matrix, this_fold_demand), axis=0)
        self.ifmap_demand_matrix = np.concatenate(ifmap_demand_matrix_list)

    # ds failed
    # def create_ifmap_demand_mat(self):
    #     """
    #     完全同步的IFMAP需求矩阵生成方法
    #     保证与Filter矩阵行数完全一致
    #     """
    #     # 1. 参数验证
    #     if not self.params_set_flag:
    #         raise RuntimeError("Parameters not set. Call set_params() first")
        
    #     if not hasattr(self, 'ifmap_op_mat') or self.ifmap_op_mat.size <= 1:
    #         raise ValueError("IFMAP operand matrix not initialized")

    #     # 2. 获取参考行数（优先使用filter矩阵的行数）
    #     reference_rows = None
    #     if hasattr(self, 'filter_demand_matrix'):
    #         reference_rows = self.filter_demand_matrix.shape[0]
    #         print(f"Using filter matrix rows ({reference_rows}) as reference")

    #     # 3. 初始化转置矩阵
    #     if not hasattr(self, 'ifmap_op_mat_trans') or self.ifmap_op_mat_trans.size <= 1:
    #         self.ifmap_op_mat_trans = self.ifmap_op_mat.T.copy()
    #         print(f"Initialized transposed IFMAP matrix: {self.ifmap_op_mat_trans.shape}")

    #     # 4. 计算基础参数
    #     arr_rows = self.arr_row
    #     arr_cols = self.arr_col
    #     inter_fold_gap = arr_cols * arr_rows - 1
        
    #     # 5. 计算理论总行数
    #     total_folds = self.col_fold * self.row_fold
    #     rows_per_fold = arr_rows + inter_fold_gap
    #     theoretical_rows = total_folds * rows_per_fold

    #     # 6. 确定最终行数（优先使用参考行数）
    #     final_rows = reference_rows if reference_rows is not None else theoretical_rows
    #     print(f"Final target rows: {final_rows}")

    #     # 7. 初始化输出矩阵
    #     self.ifmap_demand_matrix = -1 * np.ones((final_rows, arr_cols + inter_fold_gap), dtype=np.int16)
    #     self.ifmap_reads = 0

    #     # 8. 生成需求矩阵
    #     current_row = 0
    #     with tqdm(total=final_rows, desc="Generating Synced IFMAP Demand") as pbar:
    #         for fc in range(self.col_fold):
    #             for fr in range(self.row_fold):
    #                 if current_row >= final_rows:
    #                     break
                    
    #                 # 计算当前fold边界
    #                 start_r = fr * arr_rows
    #                 end_r = min(start_r + arr_rows, self.Sr)
                    
    #                 # 提取有效数据
    #                 valid_data = self.ifmap_op_mat_trans[start_r:end_r, :]
    #                 self.ifmap_reads += valid_data.size
                    
    #                 # 处理未充分利用
    #                 if valid_data.shape[0] < arr_rows:
    #                     valid_data = np.vstack([
    #                         valid_data,
    #                         -1 * np.ones((arr_rows - valid_data.shape[0], arr_cols), dtype=np.int16)
    #                     ])
                    
    #                 # 偏斜处理
    #                 skewed_data = skew_matrix(valid_data)
                    
    #                 # 计算实际可写入行数
    #                 write_rows = min(skewed_data.shape[0], final_rows - current_row)
                    
    #                 # 写入输出矩阵
    #                 self.ifmap_demand_matrix[current_row:current_row + write_rows, :skewed_data.shape[1]] = skewed_data[:write_rows, :]
    #                 current_row += write_rows
    #                 pbar.update(write_rows)
                    
    #                 # 添加间隙（如果需要且空间足够）
    #                 if write_rows < rows_per_fold and current_row < final_rows:
    #                     gap_size = min(rows_per_fold - write_rows, final_rows - current_row)
    #                     current_row += gap_size
    #                     pbar.update(gap_size)

    #     # 9. 最终验证
    #     if hasattr(self, 'filter_demand_matrix'):
    #         assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], \
    #             f"Final sync failed: IFMAP {self.ifmap_demand_matrix.shape[0]} vs Filter {self.filter_demand_matrix.shape[0]}"
        
    #     self.demand_mat_ready_flag = True
    #     print("IFMAP demand matrix generation completed with sync guarantee")

    # def create_ifmap_demand_mat(self):
    #     """
    #     Creates the IFMAP demand matrix based on layer configuration and folding factors,
    #     accounting for under-utilization and temporal locality.
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Run set_params() first."
    #     assert self.ifmap_op_mat_trans.size > 1, "IFMAP operand matrix not properly initialized"
        
    #     # Calculate inter-fold gap suffix
    #     inter_fold_gap_suffix = self.arr_col * self.arr_row - 1
    #     inter_fold_gap_suffix_mat = -1 * np.ones((1, inter_fold_gap_suffix), dtype=np.int16)
        
    #     # Initialize demand matrix and metrics
    #     self.ifmap_demand_matrix = np.zeros((0, self.arr_col + inter_fold_gap_suffix), dtype=np.int16)
    #     self.ifmap_reads = 0
    #     total_folds = self.col_fold * self.row_fold
        
    #     # Initialize progress bar
    #     pbar = tqdm(total=total_folds, desc="Creating IFMAP demand matrix")
        
    #     # Main processing loop
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate current fold boundaries
    #             start_row = fr * self.arr_row
    #             end_row = (fr + 1) * self.arr_row
    #             delta = max(end_row - self.Sr, 0)  # Underutilization amount
                
    #             # Extract current fold demand
    #             curr_fold_demand = self.ifmap_op_mat_trans[start_row:end_row - delta, :]
    #             self.ifmap_reads += curr_fold_demand.size
                
    #             # Handle underutilization
    #             if delta > 0:
    #                 null_mat = -1 * np.ones((delta, self.arr_col), dtype=np.int16)
    #                 curr_fold_demand = np.concatenate((curr_fold_demand, null_mat), axis=0)
                
    #             # Add inter-fold gap and skew
    #             curr_fold_demand = np.concatenate(
    #                 (curr_fold_demand, inter_fold_gap_suffix_mat), 
    #                 axis=1
    #             )
    #             curr_fold_demand = skew_matrix(curr_fold_demand)
                
    #             # Concatenate to final matrix
    #             if self.ifmap_demand_matrix.size == 0:
    #                 self.ifmap_demand_matrix = curr_fold_demand
    #             else:
    #                 self.ifmap_demand_matrix = np.concatenate(
    #                     (self.ifmap_demand_matrix, curr_fold_demand), 
    #                     axis=0
    #                 )
                
    #             pbar.update(1)
        
    #     pbar.close()
    #     self.demand_mat_ready_flag = True
    
    # def create_ifmap_demand_mat(self):
    #     """
    #     Optimized version of IFMAP demand matrix creation with reduced numpy operations.
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Run set_params() first."
    #     assert hasattr(self, 'ifmap_op_mat_trans'), "Transposed IFMAP matrix not initialized"
    #     assert self.ifmap_op_mat_trans.size > 1, "IFMAP matrix contains no data"
        
    #     # Pre-compute constants
    #     arr_rows = self.arr_row
    #     arr_cols = self.arr_col
    #     total_folds = self.col_fold * self.row_fold
    #     inter_fold_gap = arr_cols * arr_rows - 1
        
    #     # Initialize output matrix
    #     rows_per_fold = arr_rows + inter_fold_gap
    #     total_output_rows = total_folds * rows_per_fold
    #     self.ifmap_demand_matrix = -1 * np.ones((total_output_rows, arr_cols + inter_fold_gap), dtype=np.int16)
        
    #     # Pre-compute gap matrix pattern
    #     gap_pattern = -1 * np.ones(inter_fold_gap, dtype=np.int16)
        
    #     # Process folds
    #     with tqdm(total=total_folds, desc="Optimized IFMAP Demand") as pbar:
    #         output_row = 0
    #         for fc in range(self.col_fold):
    #             for fr in range(self.row_fold):
    #                 # Calculate fold boundaries
    #                 start_r = fr * arr_rows
    #                 end_r = min(start_r + arr_rows, self.Sr)
    #                 valid_rows = end_r - start_r
                    
    #                 # Get fold data
    #                 fold_data = self.ifmap_op_mat_trans[start_r:end_r, :]
    #                 self.ifmap_reads += fold_data.size
                    
    #                 # Handle underutilization
    #                 if valid_rows < arr_rows:
    #                     padding = -1 * np.ones((arr_rows - valid_rows, arr_cols), dtype=np.int16)
    #                     fold_data = np.vstack((fold_data, padding))
                    
    #                 # Skew the matrix (optimized version)
    #                 skewed_data = skew_matrix(fold_data)
                    
    #                 # Fill output matrix
    #                 output_end = output_row + skewed_data.shape[0]
    #                 self.ifmap_demand_matrix[output_row:output_end, :skewed_data.shape[1]] = skewed_data
                    
    #                 output_row += rows_per_fold
    #                 pbar.update(1)
        
    #     self.demand_mat_ready_flag = True


    # def create_ifmap_demand_mat(self):
    #     """
    #     Constructs IFMAP demand matrix with:
    #     - Dual fold dimension processing (rows and columns)
    #     - Comprehensive under-utilization handling
    #     - Inter-fold gap insertion (prefix and suffix)
    #     - Systolic pipeline skewing
        
    #     Raises:
    #         AssertionError if parameters not initialized
    #     """
        
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Initialization ---
    #     inter_fold_gap = self.arr_col - 1
    #     inter_fold_gap_prefix_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
    #     inter_fold_gap_suffix_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
    #     ifmap_demand_matrix_list = []
        
    #     # === Fold Processing =============================================
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate row boundaries
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             delta = self.arr_row - (end_r - start_r)
                
    #             # Extract fold data (transposed for OS dataflow)
    #             fold_data = self.ifmap_op_mat_trans[:, start_r:end_r]
    #             self.ifmap_reads += fold_data.size  # Update read count
                
    #             # Handle under-utilized folds
    #             if delta > 0:
    #                 null_pad = -np.ones((self.T, delta), dtype=np.int32)
    #                 fold_data = np.concatenate([fold_data, null_pad], axis=1)
                
    #             # Add inter-fold gaps and skew
    #             fold_data = np.vstack([
    #                 inter_fold_gap_prefix_mat,
    #                 fold_data,
    #                 inter_fold_gap_suffix_mat
    #             ])
    #             skewed_fold = skew_matrix(fold_data)
    #             ifmap_demand_matrix_list.append(skewed_fold)
        
    #     # --- Final Assembly ---
    #     self.ifmap_demand_matrix = np.vstack(ifmap_demand_matrix_list)
    #     self.demand_mat_ready_flag = True

    # def create_ifmap_demand_mat(self):
    #     """
    #     Creates IFMAP demand matrix with proper dimension handling
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # Initialize demand matrix
    #     self.ifmap_demand_matrix = np.empty((0, self.arr_col), dtype=np.int32)
    #     self.ifmap_reads = 0
        
    #     # Process each fold
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate column range
    #             col_start = fc * self.arr_col
    #             col_end = min((fc + 1) * self.arr_col, self.Sc)
    #             col_delta = self.arr_col - (col_end - col_start)
                
    #             # Get current fold data
    #             fold_data = self.ifmap_op_mat_trans[fr*self.arr_row:(fr+1)*self.arr_row, 
    #                                             col_start:col_end]
                
    #             # Pad if underutilized
    #             if col_delta > 0:
    #                 pad = -np.ones((fold_data.shape[0], col_delta), dtype=np.int32)
    #                 fold_data = np.hstack([fold_data, pad])
                
    #             # Ensure consistent dimensions
    #             if fold_data.shape[1] != self.arr_col:
    #                 # Force reshape to correct columns
    #                 fold_data = np.pad(fold_data,
    #                                 ((0, 0), (0, self.arr_col - fold_data.shape[1])),
    #                                 mode='constant',
    #                                 constant_values=-1)
                
    #             # Add pipeline fill prefix
    #             prefix = -np.ones((self.arr_row-1, self.arr_col), dtype=np.int32)
    #             fold_data = np.vstack([prefix, fold_data])
                
    #             # Skew for systolic timing
    #             fold_data = skew_matrix(fold_data)
                
    #             # Update reads counter (only actual data)
    #             self.ifmap_reads += np.count_nonzero(fold_data >= 0)
                
    #             # Concatenate to demand matrix
    #             if self.ifmap_demand_matrix.size == 0:
    #                 self.ifmap_demand_matrix = fold_data
    #             else:
    #                 # Ensure matching columns before vstack
    #                 if self.ifmap_demand_matrix.shape[1] != fold_data.shape[1]:
    #                     max_cols = max(self.ifmap_demand_matrix.shape[1], fold_data.shape[1])
    #                     if self.ifmap_demand_matrix.shape[1] < max_cols:
    #                         self.ifmap_demand_matrix = np.pad(self.ifmap_demand_matrix,
    #                                                         ((0, 0), (0, max_cols - self.ifmap_demand_matrix.shape[1])),
    #                                                         constant_values=-1)
    #                     if fold_data.shape[1] < max_cols:
    #                         fold_data = np.pad(fold_data,
    #                                         ((0, 0), (0, max_cols - fold_data.shape[1])),
    #                                         constant_values=-1)
    #                 self.ifmap_demand_matrix = np.vstack([self.ifmap_demand_matrix, fold_data])
        
    #     # Final validation
    #     assert np.all(self.ifmap_demand_matrix.shape[1] == self.arr_col), \
    #         "IFMAP demand matrix column dimension mismatch"
    #     self.demand_mat_ready_flag = True

    #
    # def create_filter_demand_mat(self):
    #     assert self.params_set_flag, 'Parameters are not set'

    #     inter_fold_gap_suffix = self.arr_row + self.arr_col + self.T - 2
    #     inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_col)) * -1

    #     filter_demand_matrix_list = []
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             row_start_id = fr * self.arr_row
    #             row_end_idx = min(row_start_id + self.arr_row, self.Sr)
    #             row_delta = self.arr_row - (row_end_idx - row_start_id)

    #             col_start_id = fc * self.arr_col
    #             col_end_idx = min(col_start_id + self.arr_col, self.Sc)
    #             col_delta = self.arr_col - (col_end_idx - col_start_id)

    #             this_fold_demand = self.filter_op_mat[row_start_id:row_end_idx, col_start_id: col_end_idx]
    #             self.filter_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

    #             # Take into account under utilization
    #             if col_delta > 0:
    #                 null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

    #             if row_delta > 0:
    #                 null_req_mat = np.ones((row_delta, self.arr_col)) * -1
    #                 this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

    #             # The filters are needed to be filled in reverse order to ensure that
    #             # top element is pushed in last to maintain alignment with the input elements
    #             this_fold_demand = np.flip(this_fold_demand, 0)

    #             # Time for inputs to stream and the partial sums to drain out
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

    #             filter_demand_matrix_list.append(this_fold_demand)
    #             #if fr == 0 and fc == 0:
    #             #    self.filter_demand_matrix = this_fold_demand
    #             #else:
    #             #    self.filter_demand_matrix = np.concatenate((self.filter_demand_matrix, this_fold_demand), axis=0)
    #     self.filter_demand_matrix = np.concatenate(filter_demand_matrix_list)
        # No skew needed in filters for weight stationary


    # def create_filter_demand_mat(self):
    #     """
    #     Creates the filter demand matrix accounting for folding and under-utilization.
    #     Maintains temporal locality with inter-fold gaps and handles systolic array requirements.
    #     """
    #     # Parameter validation
    #     assert self.params_set_flag, "Parameters not set. Run set_params() first."
    #     assert hasattr(self, 'filter_op_mat'), "Filter operand matrix not initialized"
    #     assert self.filter_op_mat.size > 1, "Filter matrix contains no data"

    #     # Initialization
    #     inter_fold_gap = self.arr_row * self.arr_col - 1
    #     gap_suffix_mat = -1 * np.ones((1, inter_fold_gap), dtype=np.int16)
        
    #     # Initialize metrics
    #     self.filter_reads = 0
    #     total_folds = self.col_fold * self.row_fold
    #     pbar = tqdm(total=total_folds, desc="Creating Filter Demand Matrix")

    #     # Main processing loops
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate current fold boundaries
    #             start_col = fc * self.arr_col
    #             end_col = (fc + 1) * self.arr_col
    #             delta = end_col - self.Sc if end_col > self.Sc else 0

    #             # Extract current fold data
    #             curr_fold_mat = self.filter_op_mat[:, start_col:end_col - delta]
    #             self.filter_reads += curr_fold_mat.size

    #             # Handle underutilization
    #             if delta > 0:
    #                 null_mat = -1 * np.ones((self.filter_op_mat.shape[0], delta), dtype=np.int16)
    #                 curr_fold_mat = np.concatenate((curr_fold_mat, null_mat), axis=1)

    #             # Add inter-fold gap and skew
    #             curr_fold_mat = np.concatenate((curr_fold_mat, gap_suffix_mat), axis=0)
    #             curr_fold_mat = skew_matrix(curr_fold_mat)

    #             # Build final matrix
    #             if fr == 0 and fc == 0:
    #                 self.filter_demand_matrix = curr_fold_mat
    #             else:
    #                 self.filter_demand_matrix = np.concatenate(
    #                     (self.filter_demand_matrix, curr_fold_mat), 
    #                     axis=0
    #                 )

    #             pbar.update(1)

    #     # Finalization
    #     pbar.close()
    #     self.demand_mat_ready_flag = True
    def create_filter_demand_mat(self):
        """
        Creates the filter demand matrix with correct gap matrix dimensions.
        """
        # Parameter validation
        assert self.params_set_flag, "Parameters not set. Run set_params() first."
        assert hasattr(self, 'filter_op_mat'), "Filter operand matrix not initialized"
        assert self.filter_op_mat.size > 1, "Filter matrix contains no data"

        # Initialization
        inter_fold_gap = self.arr_row * self.arr_col - 1
        gap_suffix_mat = -1 * np.ones((inter_fold_gap, self.arr_col), dtype=np.int16)  # Fixed dimensions
        
        # Initialize metrics
        self.filter_reads = 0
        total_folds = self.col_fold * self.row_fold
        pbar = tqdm(total=total_folds, desc="Creating Filter Demand Matrix")

        # Main processing loops
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                # Calculate current fold boundaries
                start_col = fc * self.arr_col
                end_col = (fc + 1) * self.arr_col
                delta = end_col - self.Sc if end_col > self.Sc else 0

                # Extract current fold data
                curr_fold_mat = self.filter_op_mat[:, start_col:end_col - delta]
                self.filter_reads += curr_fold_mat.size

                # Handle underutilization
                if delta > 0:
                    null_mat = -1 * np.ones((self.filter_op_mat.shape[0], delta), dtype=np.int16)
                    curr_fold_mat = np.concatenate((curr_fold_mat, null_mat), axis=1)

                # Add inter-fold gap and skew (now correct dimensions)
                curr_fold_mat = np.concatenate((curr_fold_mat, gap_suffix_mat), axis=0)  # Now compatible
                curr_fold_mat = skew_matrix(curr_fold_mat)

                # Build final matrix
                if fr == 0 and fc == 0:
                    self.filter_demand_matrix = curr_fold_mat
                else:
                    self.filter_demand_matrix = np.concatenate(
                        (self.filter_demand_matrix, curr_fold_mat), 
                        axis=0
                    )

                pbar.update(1)

        # Finalization
        pbar.close()
        self.demand_mat_ready_flag = True
    #
    def create_ofmap_demand_mat(self):
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = 2 * self.arr_row - 1
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_col)) * -1

        ofmap_demand_matrix_list = []
        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = self.ofmap_op_mat[:, col_start_id: col_end_idx]
                self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Adding null requests when there is under utilization ie. no mapping along a few rows or cols
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # Now add the prefix matrix
                # These are the null demands to account for when the operands are streamed in
                # and the OFMAPS are not ready
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand), axis=0)

                # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                ofmap_demand_matrix_list.append(this_fold_demand)
                #if fr == 0 and fc == 0:
                #    self.ofmap_demand_matrix = this_fold_demand
                #else:
                #    self.ofmap_demand_matrix = np.concatenate((self.ofmap_demand_matrix, this_fold_demand), axis=0)
        self.ofmap_demand_matrix = np.concatenate(ofmap_demand_matrix_list)
    # END of OFMAP demand generation

    # def create_filter_demand_mat(self):
    #     """
    #     Constructs filter demand matrix with:
    #     - Dual-dimension fold processing (rows and columns)
    #     - Comprehensive under-utilization handling
    #     - Output alignment flipping
    #     - Inter-fold gap insertion
    #     - Performance metric tracking
        
    #     Raises:
    #         AssertionError if parameters not initialized
    #     """
        
    #     # --- Parameter Validation ---
    #     assert self.params_set_flag, "Parameters not set. Call set_params() first"
        
    #     # --- Initialization ---
    #     inter_fold_gap = self.T - 1  # Temporal dimension gap
    #     inter_fold_gap_suffix_mat = -np.ones((inter_fold_gap, self.arr_col), dtype=np.int32)
    #     filter_demand_matrix_list = []
        
    #     # === Fold Processing =============================================
    #     for fc in range(self.col_fold):
    #         for fr in range(self.row_fold):
    #             # Calculate boundaries
    #             start_r = fr * self.arr_row
    #             end_r = min(start_r + self.arr_row, self.Sr)
    #             row_delta = self.arr_row - (end_r - start_r)
                
    #             start_c = fc * self.arr_col
    #             end_c = min(start_c + self.arr_col, self.Sc)
    #             col_delta = self.arr_col - (end_c - start_c)
                
    #             # Extract fold data
    #             fold_data = self.filter_op_mat[start_r:end_r, start_c:end_c]
    #             self.filter_reads += fold_data.size  # Update read count
                
    #             # Handle column under-utilization
    #             if col_delta > 0:
    #                 null_pad = -np.ones((fold_data.shape[0], col_delta), dtype=np.int32)
    #                 fold_data = np.concatenate([fold_data, null_pad], axis=1)
                
    #             # Handle row under-utilization
    #             if row_delta > 0:
    #                 null_pad = -np.ones((row_delta, fold_data.shape[1]), dtype=np.int32)
    #                 fold_data = np.concatenate([fold_data, null_pad], axis=0)
                
    #             # Align with input elements
    #             fold_data = np.flipud(fold_data)
                
    #             # Add inter-fold gap and skew
    #             fold_data = np.vstack([fold_data, inter_fold_gap_suffix_mat])
    #             skewed_fold = skew_matrix(fold_data)
                
    #             # Calculate performance metrics
    #             valid_elements = np.count_nonzero(fold_data >= 0)
    #             total_elements = fold_data.size
    #             self.mapping_efficiency_per_fold.append(valid_elements / total_elements)
    #             self.compute_utility_per_fold.append(
    #                 valid_elements / (total_elements + inter_fold_gap * self.arr_col)
    #             )
                
    #             filter_demand_matrix_list.append(skewed_fold)
        
    #     # --- Final Assembly ---
    #     self.filter_demand_matrix = np.vstack(filter_demand_matrix_list)
    #     self.demand_mat_ready_flag = True
    # #
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
def skew_matrix(input_matrix_np):
    rows, cols = input_matrix_np.shape

    out_matrix_np = np.full((rows + cols - 1, cols), -1, dtype=input_matrix_np.dtype)

    for c in range(cols):
        out_matrix_np[c:c + rows, c] = input_matrix_np[:, c]

    return out_matrix_np
