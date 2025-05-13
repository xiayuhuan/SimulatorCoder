import math
import os

class topologies(object):

    # def __init__(self):
    #     self.current_topo_name = ""
    #     self.topo_file_name = ""
    #     self.topo_arrays = []
    #     self.spatio_temp_dim_arrays = []
    #     self.layers_calculated_hyperparams = []
    #     self.num_layers = 0
    #     self.topo_load_flag = False
    #     self.topo_calc_hyper_param_flag = False
    #     self.topo_calc_spatiotemp_params_flag = False

    def __init__(self):
        """
        Initializes a topology manager for neural network configurations.
        
        Establishes data structures for:
        - Topology identification (name, source file)
        - Network parameter storage 
        - Computation state tracking
        - Dimensional analysis arrays
        """
        # 1. Topology Identification
        self.current_topo_name = "uninitialized"
        self.topo_file_name = ""
        
        # 2. Data Storage Containers
        self.topo_arrays = []
        self.spatio_temp_dim_arrays = []
        self.layers_calculated_hyperparams = []
        
        # 3. Network Dimensions
        self.num_layers = 0
        
        # 4. State Tracking Flags
        self.topo_load_flag = False
        self.topo_calc_hyper_param_flag = False
        self.topo_calc_spatiotemp_params_flag = False

    # reset topology parameters
    def reset(self):
        print("All data reset")
        self.current_topo_name = ""
        self.topo_file_name = ""
        self.topo_load_flag = False
        self.topo_arrays = []
        self.num_layers = 0
        self.topo_calc_hyper_param_flag = False
        self.layers_calculated_hyperparams = []

    #
    # def load_layer_params_from_list(self, layer_name, elems_list=[]):
    #     self.topo_file_name = ''
    #     self.current_toponame = ''
    #     self.layer_name = layer_name
    #     self.append_topo_arrays(layer_name, elems_list)

    #     self.num_layers += 1
    #     self.topo_load_flag = True

    def load_layer_params_from_list(self, layer_name: str, elems_list: list) -> None:
        """
        Loads layer parameters from a list and updates topology configuration.
        
        Args:
            layer_name: Name identifier for the layer
            elems_list: List of layer parameters to be loaded
            
        Raises:
            ValueError: If empty layer name or invalid parameter list provided
            TypeError: If incorrect parameter types are supplied
            
        Effects:
            - Updates topology arrays with new layer parameters
            - Increments layer counter
            - Sets topology loaded flag
        """
        # 1. Input Validation
        if not isinstance(layer_name, str) or not layer_name.strip():
            raise ValueError("Layer name must be a non-empty string")
        if not isinstance(elems_list, list):
            raise TypeError("Layer parameters must be provided as a list")

        # 2. Reset Topology Metadata
        self.topo_file_name = ""
        self.current_topo_name = ""

        # 3. Store Layer Identification
        self.layer_name = layer_name

        # 4. Process Parameters
        self.append_topo_arrays(layer_name, elems_list)

        # 5. Update Topology State
        self.num_layers += 1
        self.topo_load_flag = True

    #
    def load_arrays(self, topofile='', mnk_inputs=False):
        if mnk_inputs:
            self.load_arrays_gemm(topofile)
        else:
            self.load_arrays_conv(topofile)

    # def load_arrays(topofile: str, mnk_inputs: bool):
    #     """
    #     Loads arrays from a specified file based on the input type (GEMM or convolution).
        
    #     Parameters:
    #     - topofile: str - Path to the file containing the array data
    #     - mnk_inputs: bool - Flag indicating whether to load GEMM inputs (True) 
    #                         or convolution inputs (False)
                            
    #     Returns:
    #     - The loaded arrays (specific format depends on the called method)
        
    #     Raises:
    #     - ValueError: If the input file is not found or is invalid
    #     """
    #     # Input validation (sequential structure)
    #     if not isinstance(topofile, str):
    #         raise ValueError("topofile must be a string")
    #     if not isinstance(mnk_inputs, bool):
    #         raise ValueError("mnk_inputs must be a boolean")
        
    #     # Branch structure to determine loading method
    #     if mnk_inputs:
    #         # GEMM array loading path
    #         arrays = load_arrays_gemm(topofile)
    #     else:
    #         # Convolution array loading path
    #         arrays = load_arrays_conv(topofile)
            
    #     return arrays

    #
    # def load_arrays_gemm(self, topofile: str) -> list:

    #     self.topo_file_name = topofile.split('/')[-1]
    #     name_arr = self.topo_file_name.split('.')
    #     if len(name_arr) > 1:
    #         self.current_topo_name = self.topo_file_name.split('.')[-2]
    #     else:
    #         self.current_topo_name = self.topo_file_name

    #     f = open(topofile, 'r')
    #     first = True

    #     for row in f:
    #         row = row.strip()
    #         if first:
    #             first = False
    #             continue
    #         elif row == '':
    #             continue
    #         else:
    #             elems = row.split(',')[:-1]
    #             assert len(elems) > 3, 'There should be at least 4 entries per row'
    #             layer_name = elems[0].strip()
    #             m = elems[1].strip()
    #             n = elems[2].strip()
    #             k = elems[3].strip()

    #             # Entries: layer name, Ifmap h, ifmap w, filter h, filter w, num_ch, num_filt, stride h, stride w
    #             entries = [layer_name, m, k, 1, k, 1, n, 1, 1]
    #             #entries are later iterated from index 1. Index 0 is used to store layer name in convolution mode. So, to rectify assignment of M, N and K in GEMM mode, layer name has been added at index 0 of entries. 
    #             self.append_topo_arrays(layer_name=layer_name, elems=entries)

    #     self.num_layers = len(self.topo_arrays)
    #     self.topo_load_flag = True

    def load_arrays_gemm(self, topofile: str) -> None:
        """
        Loads GEMM array configurations from a specified topology file.
        
        Args:
            topofile: Path to the topology file containing GEMM configurations
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If file format is invalid or rows don't contain required entries
            
        Sets:
            self.topo_file_name: Extracted filename
            self.current_topo_name: Base topology name
            self.topo_arrays: List of GEMM configurations
            self.num_layers: Number of loaded layers
            self.topo_load_flag: True after successful loading
        """
        # Sequential: File path processing
        self.topo_file_name = topofile.split('/')[-1].split('\\')[-1]
        self.current_topo_name = self.topo_file_name.split('.')[0]
        
        try:
            with open(topofile, 'r') as f:
                # Initialize parsing variables
                first_row = True
                self.topo_arrays = []
                
                # Loop: Process each row
                for line in f:
                    line = line.strip()
                    
                    # Branch: Skip processing conditions
                    if not line:
                        continue
                    if first_row:
                        first_row = False
                        continue
                    
                    # Branch: Validate row format
                    entries = line.split()
                    if len(entries) < 4:
                        raise ValueError(
                            f"Row must contain at least 4 entries (layer_name, m, n, k). Got: {line}"
                        )
                    
                    # Sequential: Extract and format data
                    layer_name, m, n, k = entries[0], entries[1], entries[2], entries[3]
                    gemm_entry = [
                        layer_name,
                        'gemm',
                        int(m),
                        int(n),
                        int(k),
                        1,  # Default batch size
                        1   # Default stride
                    ]
                    
                    # Sequential: Append configuration
                    self.append_topo_arrays(gemm_entry)
                
                # Sequential: Finalize loading
                self.num_layers = len(self.topo_arrays)
                self.topo_load_flag = True
                
        except FileNotFoundError:
            raise FileNotFoundError(f"GEMM topology file not found: {topofile}")
        except ValueError as e:
            raise ValueError(f"Invalid data format in {topofile}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading GEMM arrays: {e}")


    # Load the topology data from the file
    # def load_arrays_conv(self, topofile=""):
    #     first = True
    #     self.topo_file_name = topofile.split('/')[-1]
    #     name_arr = self.topo_file_name.split('.')
    #     if len(name_arr) > 1:
    #         self.current_topo_name = self.topo_file_name.split('.')[-2]
    #     else:
    #         self.current_topo_name = self.topo_file_name
    #     f = open(topofile, 'r')
    #     for row in f:
    #         row = row.strip()
    #         if first or row == '':
    #             first = False
    #         else:
    #             elems = row.split(',')[:-1]
    #             # depth-wise convolution
    #             if 'DP' in elems[0].strip():
    #                 for dp_layer in range(int(elems[5].strip())):
    #                     layer_name = elems[0].strip() + "Channel_" + str(dp_layer)
    #                     elems[5] = str(1)
    #                     self.append_topo_arrays(layer_name, elems)
    #             else:
    #                 layer_name = elems[0].strip()
    #                 self.append_topo_arrays(layer_name, elems)

    #     self.num_layers = len(self.topo_arrays)
    #     self.topo_load_flag = True

    # ds
    def load_arrays_conv(self, topofile: str) -> None:
        """
        Load convolution configurations from file, handling comma-separated numbers.
        
        Args:
            topofile: Path to topology file with format:
                    layer_name, h, w, r, s, c, f, stride
        """
        def clean_num(num_str: str) -> int:
            """Remove commas/whitespace and convert to int"""
            return int(num_str.replace(',', '').strip())

        self.topo_arrays = []
        
        with open(topofile, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                entries = [e.strip() for e in line.split(',') if e.strip()]
                if len(entries) < 8:
                    continue
                    
                try:
                    # Process all numerical parameters
                    params = [clean_num(x) for x in entries[1:8]]
                    
                    # Build layer entry
                    entry = [
                        entries[0],        # layer_name
                        *params[:2],       # h, w
                        *params[2:4],      # r, s
                        params[4],         # c
                        params[5],         # f
                        params[6],         # stride_h
                        params[6]          # stride_w (duplicated)
                    ]
                    
                    # Validate dimensions
                    if params[2] > params[0] or params[3] > params[1]:
                        print(f"Warning: Invalid filter size in {entries[0]}")
                        continue
                        
                    self.topo_arrays.append(entry)
                    
                except ValueError as e:
                    print(f"Skipping invalid line: {line} - Error: {e}")
                    continue

        self.num_layers = len(self.topo_arrays)
        self.topo_load_flag = bool(self.topo_arrays)
    
    # ICL error1
    # def load_arrays_conv(self, topofile: str) -> None:
    #     """Load convolution layer configurations from file with depth-wise conv support
        
    #     Args:
    #         topofile: Path to topology configuration file
    #     """
    #     first_row = True
    #     self.topo_file_name = topofile.split('/')[-1]
        
    #     # Extract topology name
    #     name_parts = self.topo_file_name.split('.')
    #     self.current_topo_name = name_parts[-2] if len(name_parts) > 1 else name_parts[0]
        
    #     with open(topofile, 'r') as f:
    #         for row in f:
    #             row = row.strip()
    #             if not row or first_row:  # Skip header and empty lines
    #                 first_row = False
    #                 continue
                    
    #             elements = [elem.strip() for elem in row.split(',')]
    #             layer_name = elements[0]
                
    #             # Depth-wise convolution handling
    #             if 'DP' in layer_name:
    #                 channels = int(elements[7])  # Channel count at index 7
    #                 for ch in range(channels):
    #                     channel_elements = elements.copy()
    #                     channel_elements[0] = f"{layer_name}_ch{ch}"
    #                     self.topo_arrays.append(channel_elements)
    #             else:
    #                 self.topo_arrays.append(elements)
        
    #     # Finalize loading
    #     self.num_layers = len(self.topo_arrays)
    #     self.topo_load_flag = True

    # iCl error2
    # def load_arrays_conv(self, topofile: str) -> None:
    #     """Load convolution layer configurations from file with depth-wise conv support
        
    #     Args:
    #         topofile: Path to topology configuration file
        
    #     Raises:
    #         ValueError: If any numerical parameter cannot be converted to int
    #     """
    #     first_row = True
    #     self.topo_file_name = topofile.split('/')[-1]
        
    #     # Extract topology name
    #     name_parts = self.topo_file_name.split('.')
    #     self.current_topo_name = name_parts[-2] if len(name_parts) > 1 else name_parts[0]
        
    #     with open(topofile, 'r') as f:
    #         for row_idx, row in enumerate(f, 1):
    #             row = row.strip()
    #             if not row or first_row:  # Skip header and empty lines
    #                 first_row = False
    #                 continue
                    
    #             elements = [elem.strip() for elem in row.split(',')]
    #             layer_name = elements[0]
                
    #             try:
    #                 # Convert all numerical parameters to integers
    #                 converted_elems = [layer_name]  # Keep layer name as string
    #                 for elem in elements[1:]:
    #                     converted_elems.append(int(elem))
                    
    #                 # Depth-wise convolution handling
    #                 if 'DP' in layer_name:
    #                     channels = converted_elems[7]  # Channel count at index 7
    #                     for ch in range(channels):
    #                         channel_elems = converted_elems.copy()
    #                         channel_elems[0] = f"{layer_name}_ch{ch}"
    #                         self.topo_arrays.append(channel_elems)
    #                 else:
    #                     self.topo_arrays.append(converted_elems)
                        
    #             except ValueError as e:
    #                 raise ValueError(
    #                     f"Error in row {row_idx} (layer '{layer_name}'): "
    #                     f"Invalid numerical parameter - {str(e)}"
    #                 ) from e
        
    #     # Finalize loading
    #     self.num_layers = len(self.topo_arrays)
    #     self.topo_load_flag = True
    
    # ICL SUCESS
    # def load_arrays_conv(self, topofile: str) -> None:
    #     """Load convolution layer configurations with automatic default values
        
    #     Args:
    #         topofile: Path to topology configuration file
        
    #     Raises:
    #         ValueError: If essential parameters are missing
    #         FileNotFoundError: If input file doesn't exist
    #     """
    #     # Initialize tracking variables
    #     first_row = True
    #     self.topo_arrays = []
        
    #     try:
    #         # Validate file existence
    #         if not os.path.exists(topofile):
    #             raise FileNotFoundError(f"Topology file not found: {topofile}")
                
    #         self.topo_file_name = os.path.basename(topofile)
    #         self.current_topo_name = os.path.splitext(self.topo_file_name)[0]

    #         with open(topofile, 'r', encoding='utf-8') as f:
    #             for row_idx, row in enumerate(f, 1):
    #                 row = row.strip()
    #                 if not row or row.startswith('#') or first_row:
    #                     first_row = False
    #                     continue
                        
    #                 elements = [elem.strip() for elem in row.split(',')]
    #                 if len(elements) < 9:
    #                     print(f"Warning: Row {row_idx} has only {len(elements)} columns, skipping")
    #                     continue
                        
    #                 layer_name = elements[0]
    #                 try:
    #                     # Default values: [ifmap_h, ifmap_w, filt_h, filt_w, channels, filters, stride_h, stride_w]
    #                     defaults = [32, 32, 3, 3, 1, 1, 1, 1]  # Reasonable CNN defaults
                        
    #                     params = []
    #                     for i in range(1, 9):  # Process columns 1-8
    #                         elem = elements[i] if i < len(elements) else ''
    #                         if not elem:
    #                             default_val = defaults[i-1]
    #                             print(f"Warning: Row {row_idx} column {i} empty, using default {default_val}")
    #                             params.append(default_val)
    #                         else:
    #                             try:
    #                                 params.append(int(float(elem)))
    #                             except ValueError:
    #                                 raise ValueError(f"Invalid number '{elem}' in column {i}")

    #                     # Create validated layer entry
    #                     layer_entry = [layer_name] + params
                        
    #                     # Handle depth-wise convolutions
    #                     if 'DP' in layer_name:
    #                         channels = max(1, params[4])  # channel count
    #                         for ch in range(channels):
    #                             ch_entry = layer_entry.copy()
    #                             ch_entry[0] = f"{layer_name}_ch{ch}"
    #                             self.topo_arrays.append(ch_entry)
    #                     else:
    #                         self.topo_arrays.append(layer_entry)
                            
    #                 except ValueError as e:
    #                     raise ValueError(
    #                         f"Error in row {row_idx} (layer '{layer_name}'): {str(e)}\n"
    #                         f"Row content: {row}"
    #                     ) from e

    #         if not self.topo_arrays:
    #             raise ValueError(f"No valid layers found in {topofile}")
                
    #         self.num_layers = len(self.topo_arrays)
    #         self.topo_load_flag = True

    #     except Exception as e:
    #         self.topo_arrays = []
    #         self.topo_load_flag = False
    #         raise

    # def load_arrays_conv(self, topofile: str) -> None:
    #     """
    #     Loads convolution array configurations from a specified topology file.
    #     Assumes caller handles file operations and validation.
        
    #     Args:
    #         topofile: Path to the topology file containing convolution configurations
            
    #     Requires:
    #         - File exists and is readable
    #         - File format matches expected convolution specification
    #         - Caller handles any file-related exceptions
            
    #     Sets:
    #         self.topo_file_name: Extracted filename
    #         self.current_topo_name: Base topology name
    #         self.topo_arrays: List of convolution configurations
    #         self.num_layers: Number of loaded layers
    #         self.topo_load_flag: True after successful loading
    #     """
    #     # Sequential initialization
    #     self.topo_arrays = []
    #     first_row = True
        
    #     # File path processing
    #     self.topo_file_name = topofile.split('/')[-1].split('\\')[-1]
    #     name_parts = self.topo_file_name.split('.')
    #     self.current_topo_name = name_parts[-2] if len(name_parts) > 1 else name_parts[0]
        
    #     # File processing (assumes file exists and is readable)
    #     with open(topofile, 'r') as f:
    #         for line in f:
    #             line = line.strip()
                
    #             # Skip empty lines and header
    #             if not line or first_row:
    #                 first_row = False
    #                 continue
                
    #             entries = line.split()
    #             layer_name = entries[0]
    #             layer_type = entries[1].lower()
                
    #             # Convert all numerical values
    #             params = [int(x) for x in entries[2:]]
                
    #             # Depth-wise convolution handling
    #             if 'dp' in layer_name.lower():
    #                 channels = params[2]  # c dimension
    #                 for ch in range(channels):
    #                     conv_entry = [
    #                         f"{layer_name}_ch{ch}",
    #                         'dp_conv',
    #                         params[0],  # h
    #                         params[1],  # w
    #                         1,          # c=1 for depth-wise
    #                         params[3],  # f
    #                         params[4],  # r
    #                         params[5] if len(params) > 5 else 1  # s
    #                     ]
    #                     self.append_topo_arrays(conv_entry)
    #             else:
    #                 # Regular convolution
    #                 conv_entry = [
    #                     layer_name,
    #                     layer_type,
    #                     *params[:5],  # h, w, c, f, r
    #                     params[5] if len(params) > 5 else 1  # s
    #                 ]
    #                 self.append_topo_arrays(conv_entry)
        
    #     # Finalization
    #     self.num_layers = len(self.topo_arrays)
    #     self.topo_load_flag = True




    # Write the contents into a csv file
    def write_topo_file(self,
                      path="",
                      filename=""
                      ):
        if path == "":
            print("WARNING: topology_utils.write_topo_file: No path specified writing to the cwd")
            path = "./" 

        if filename == "":
            print("ERROR: topology_utils.write_topo_file: No filename provided")
            return

        filename = path + "/" + filename

        if not self.topo_load_flag:
            print("ERROR: topology_utils.write_topo_file: No data loaded")
            return

        header = [
                    "Layer name",
                    "IFMAP height",
                    "IFMAP width",
                    "Filter height",
                    "Filter width",
                    "Channels",
                    "Num filter",
                    "Stride height",
                    "Stride width"
                ]

        f = open(filename, 'w')
        log = ",".join(header)
        log += ",\n"
        f.write(log)

        for param_arr in self.topo_arrays:
            log = ",".join([str(x) for x in param_arr])
            log += ",\n"
            f.write(log)

        f.close()

    # LEGACY
    # def append_topo_arrays(self, layer_name, elems):
    #     entry = [layer_name]

    #     for i in range(1, len(elems)):
    #         val = int(str(elems[i]).strip())
    #         entry.append(val)
    #         if i == 7 and len(elems) < 9:
    #             entry.append(val)  # Add the same stride in the col direction automatically

    #     # ISSUE #9 Fix
    #     assert entry[3] <= entry[1], 'Filter height cannot be larger than IFMAP height'
    #     assert entry[4] <= entry[2], 'Filter width cannot be larger than IFMAP width'

    #     self.topo_arrays.append(entry)

    # def append_topo_arrays(self, layer_name: str, elems: list) -> None:
    #     """
    #     Appends a new layer configuration to the topology arrays.
    #     Validates dimensions and handles stride duplication when needed.
        
    #     Args:
    #         layer_name: Name of the layer being added
    #         elems: List of topology parameters as strings
            
    #     Raises:
    #         AssertionError: For invalid filter/IFMAP dimensions
    #     """
    #     # Initialize entry with layer name
    #     entry = [layer_name]
        
    #     # Process topology parameters
    #     for i in range(1, len(elems)):
    #         # Convert to integer and strip whitespace
    #         val = int(elems[i].strip())
    #         entry.append(val)
            
    #         # Handle stride duplication for 2D case
    #         if i == 7 and len(elems) < 9:
    #             entry.append(val)  # Duplicate stride for column direction
        
    #     # Validate filter dimensions
    #     assert entry[3] <= entry[1], \
    #         f"Filter height {entry[3]} cannot exceed IFMAP height {entry[1]}"
    #     assert entry[4] <= entry[2], \
    #         f"Filter width {entry[4]} cannot exceed IFMAP width {entry[2]}"
        
    #     # Append to topology arrays
    #     self.topo_arrays.append(entry)
    def append_topo_arrays(self, layer_name: str, elems: list) -> None:
        """Appends a new layer configuration with validation
        
        Args:
            layer_name: Name identifier for the layer
            elems: List of parameters in order:
                [IFMAP height, IFMAP width, 
                Filter height, Filter width,
                Stride (h), Stride (w), 
                Channels, Padding]
        """
        # Initialize entry with layer name
        entry = [layer_name]
        
        # Process each parameter
        for i in range(1, len(elems)):
            # Convert and clean numeric parameters
            val = int(str(elems[i]).replace(',', '').strip())
            entry.append(val)
            
            # Handle stride duplication if needed
            if i == 7 and len(elems) < 9:
                entry.append(val)  # Duplicate last stride for w direction
        
        # Dimension validation
        assert entry[3] <= entry[1], \
            f"Filter height {entry[3]} exceeds IFMAP height {entry[1]}"
        assert entry[4] <= entry[2], \
            f"Filter width {entry[4]} exceeds IFMAP width {entry[2]}"
        
        # Append to topology arrays
        self.topo_arrays.append(entry)
    # create network topology array
    # def append_topo_entry_from_list(self, layer_entry_list=[]):
    #     assert 7 < len(layer_entry_list) < 10, 'Incorrect number of parameters'

    #     entry = [str(layer_entry_list[0])]

    #     for i in range(1, len(layer_entry_list)):
    #         val = int(str(layer_entry_list[i]).strip())
    #         entry.append(val)
    #         if i == 7 and len(layer_entry_list) < 9:
    #             entry.append(val)           # Add the same stride in the col direction automatically

    #     self.append_layer_entry(entry,toponame=self.current_topo_name)

    def append_topo_entry_from_list(self, layer_entry_list: list) -> None:
        """
        Appends a new layer configuration from a parameter list to the topology arrays.
        Handles stride duplication and validates parameter count.
        
        Args:
            layer_entry_list: List of layer parameters (8-9 elements)
            
        Raises:
            AssertionError: For incorrect parameter count
        """
        # --- Parameter Validation ---
        assert 8 <= len(layer_entry_list) <= 9, \
            f"Layer entry requires 8-9 parameters, got {len(layer_entry_list)}"
        
        # --- Entry Initialization ---
        entry = [str(layer_entry_list[0])]  # Layer name as string
        
        # --- Parameter Processing ---
        for i in range(1, len(layer_entry_list)):
            # Convert to integer, handling string or numeric input
            val = int(str(layer_entry_list[i]).strip())
            entry.append(val)
            
            # Duplicate stride if only one provided
            if i == 7 and len(layer_entry_list) < 9:
                entry.append(val)  # Same stride for column direction
        
        # --- Append to Topology ---
        self.append_layer_entry(self.current_topo_name, entry)

    # add to the existing data from a list
    # def append_layer_entry(self, entry, toponame=""):
    #     assert len(entry) == 9, 'Incorrect number of parameters'

    #     if not toponame == "":
    #         self.current_topo_name = toponame

    #     self.topo_arrays.append(entry)
    #     self.topo_load_flag = True
    #     self.topo_calc_hyperparams()
    #     self.num_layers += 1

    def append_layer_entry(self, toponame: str, entry: list) -> None:
        """
        Appends a new layer entry to topology arrays and updates system state.
        
        Args:
            toponame: Name of the topology (empty string to keep current)
            entry: Layer configuration parameters (must have exactly 9 elements)
            
        Raises:
            AssertionError: If entry doesn't contain exactly 9 parameters
        """
        # --- Parameter Validation ---
        assert len(entry) == 9, \
            f"Layer entry requires exactly 9 parameters, got {len(entry)}"
        
        # --- Topology Name Update ---
        if toponame:  # Only update if non-empty string provided
            self.current_topo_name = toponame
        
        # --- Topology Array Update ---
        self.topo_arrays.append(entry)
        
        # --- State Updates ---
        self.topo_load_flag = True
        self.num_layers += 1
        
        # --- Hyperparameter Calculation ---
        self.topo_calc_hyperparams()

    # calculate hyper-parameters (ofmap dimensions, number of MACs, and window size of filter)
    # def topo_calc_hyperparams(self, topofilename=""):
    #     if not self.topo_load_flag:
    #         self.load_arrays(topofilename)
    #     self.layers_calculated_hyperparams = []
    #     for array in self.topo_arrays:
    #         ifmap_h = array[1]
    #         ifmap_w = array[2]
    #         filt_h = array[3]
    #         filt_w = array[4]
    #         num_ch   = array[5]
    #         num_filt = array[6]
    #         stride_h = array[7]
    #         stride_w = array[8]
    #         ofmap_h = int(math.ceil((ifmap_h - filt_h + stride_h) / stride_h))
    #         ofmap_w = int(math.ceil((ifmap_w - filt_w + stride_w) / stride_w))
    #         num_mac = ofmap_h * ofmap_w * filt_h * filt_w * num_ch * num_filt
    #         window_size = filt_h * filt_w * num_ch
    #         entry = [ofmap_h, ofmap_w, num_mac, window_size]
    #         self.layers_calculated_hyperparams.append(entry)
    #     self.topo_calc_hyper_param_flag = True

    # def topo_calc_hyperparams(self, topofilename: str = "") -> None:
    #     """
    #     Calculates and stores hyperparameters for each layer in the topology.
        
    #     Args:
    #         topofilename: Optional topology filename to load if not already loaded
    #     """
    #     # --- Topology Loading Check ---
    #     if not self.topo_load_flag:
    #         self.load_arrays(topofilename)
        
    #     # --- Initialization ---
    #     self.layers_calculated_hyperparams = []
        
    #     # --- Layer Processing ---
    #     for layer in self.topo_arrays:
    #         # Extract layer parameters
    #         ifmap_h, ifmap_w = layer[1], layer[2]
    #         filt_h, filt_w = layer[3], layer[4]
    #         num_ch, num_filt = layer[5], layer[6]
    #         stride_h, stride_w = layer[7], layer[8]
            
    #         # Calculate OFMAP dimensions
    #         ofmap_h = ((ifmap_h - filt_h) // stride_h) + 1
    #         ofmap_w = ((ifmap_w - filt_w) // stride_w) + 1
            
    #         # Calculate MAC operations
    #         num_mac = ofmap_h * ofmap_w * filt_h * filt_w * num_ch * num_filt
            
    #         # Calculate window size
    #         window_size = filt_h * filt_w * num_ch
            
    #         # Store calculated parameters
    #         self.layers_calculated_hyperparams.append([
    #             ofmap_h, ofmap_w, 
    #             num_mac, 
    #             window_size
    #         ])
        
    #     # --- Completion Flag ---
    #     self.topo_calc_hyper_param_flag = True
    # ds
    def topo_calc_hyperparams(self, topofilename: str = "") -> None:
        """
        Calculates and stores hyperparameters for each layer in the topology.
        
        Args:
            topofilename: Optional topology filename to load if not already loaded
        """
        # --- Topology Loading Check ---
        if not self.topo_load_flag:
            self.load_arrays(topofilename)
        
        # --- Initialization ---
        self.layers_calculated_hyperparams = []
        
        # --- Layer Processing ---
        for layer in self.topo_arrays:
            # Extract layer parameters
            ifmap_h, ifmap_w = layer[1], layer[2]
            filt_h, filt_w = layer[3], layer[4]
            num_ch, num_filt = layer[5], layer[6]
            stride_h, stride_w = layer[7], layer[8]
            
            # Calculate OFMAP dimensions
            ofmap_h = ((ifmap_h - filt_h) // stride_h) + 1
            ofmap_w = ((ifmap_w - filt_w) // stride_w) + 1
            
            # Calculate MAC operations
            num_mac = ofmap_h * ofmap_w * filt_h * filt_w * num_ch * num_filt
            
            # Calculate window size
            window_size = filt_h * filt_w * num_ch
            
            # Store calculated parameters
            self.layers_calculated_hyperparams.append([
                ofmap_h, ofmap_w, 
                num_mac, 
                window_size
            ])
        
        # --- Completion Flag ---
        self.topo_calc_hyper_param_flag = True

    # def calc_spatio_temporal_params(self, df='os', layer_id=0):
    #     s_row = -1
    #     s_col = -1
    #     t_time = -1
    #     if self.topo_calc_hyper_param_flag:
    #         num_filt  = self.get_layer_num_filters(layer_id= layer_id)
    #         num_ofmap = self.get_layer_num_ofmap_px(layer_id=layer_id)
    #         num_ofmap = int(num_ofmap / num_filt)
    #         window_sz = self.get_layer_window_size(layer_id=layer_id)
    #         if df == 'os':
    #             s_row = num_ofmap
    #             s_col = num_filt
    #             t_time = window_sz
    #         elif df == 'ws':
    #             s_row = window_sz
    #             s_col = num_filt
    #             t_time = num_ofmap
    #         elif df == 'is':
    #             s_row = window_sz
    #             s_col = num_ofmap
    #             t_time = num_filt
    #     else:
    #         self.topo_calc_hyperparams(self.topo_file_name)
    #     return s_row, s_col, t_time

    def calc_spatio_temporal_params(self, df: str, layer_id: int) -> tuple:
        """
        Calculates spatial and temporal parameters for a given layer based on dataflow type.
        
        Args:
            df: Dataflow type ('os', 'ws', or 'is')
            layer_id: Index of layer in topology
            
        Returns:
            Tuple of (s_row, s_col, t_time) parameters
            
        Note:
            Requires hyperparameters to be pre-calculated or will trigger calculation
        """
        # Initialize default values
        s_row, s_col, t_time = -1, -1, -1
        
        # Verify hyperparameters are calculated
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()
        
        # Get layer parameters
        layer = self.topo_arrays[layer_id]
        num_filt = layer[6]
        ofmap_h, ofmap_w, _, window_sz = self.layers_calculated_hyperparams[layer_id]
        num_ofmap = (ofmap_h * ofmap_w) // num_filt
        
        # Calculate based on dataflow type
        if df == 'os':  # Output stationary
            s_row, s_col, t_time = num_ofmap, num_filt, window_sz
        elif df == 'ws':  # Weight stationary
            s_row, s_col, t_time = window_sz, num_filt, num_ofmap
        elif df == 'is':  # Input stationary
            s_row, s_col, t_time = window_sz, num_ofmap, num_filt
        
        return s_row, s_col, t_time

    # def set_spatio_temporal_params(self):
    #     if not self.topo_calc_hyper_param_flag:
    #         self.topo_calc_hyperparams(self.topo_file_name)
    #     for i  in range(self.num_layers):
    #         this_layer_params_arr = []
    #         for df in ['os', 'ws', 'is']:
    #             sr, sc, tt = self.calc_spatio_temporal_params(df=df, layer_id=i)
    #             this_layer_params_arr.append([sr, sc, tt])
    #         self.spatio_temp_dim_arrays.append(this_layer_params_arr)
    #     self.topo_calc_spatiotemp_params_flag = True

    def set_spatio_temporal_params(self) -> None:
        """
        Calculates and stores spatial/temporal parameters for all layers and dataflows.
        Generates a 3D array of parameters indexed by [layer][dataflow][parameter].
        """
        # --- Hyperparameter Check ---
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()

        # --- Initialize Storage ---
        self.spatio_temp_dim_arrays = []

        # --- Process Each Layer ---
        for layer_id in range(self.num_layers):
            layer_params = []
            
            # --- Calculate for Each Dataflow ---
            for df in ['os', 'ws', 'is']:  # Output, Weight, Input stationary
                sr, sc, tt = self.calc_spatio_temporal_params(df, layer_id)
                layer_params.append([sr, sc, tt])
            
            self.spatio_temp_dim_arrays.append(layer_params)

        # --- Mark Completion ---
        self.topo_calc_spatiotemp_params_flag = True

    # def get_transformed_mnk_dimensions(self):
    #     if not self.topo_calc_hyper_param_flag:
    #         self.topo_calc_hyperparams(self.topo_file_name)

    #     mnk_dims_arr = []
    #     for i in range(self.num_layers):
    #         M = self.get_layer_num_ofmap_px(layer_id=i)
    #         N = self.get_layer_num_filters(layer_id=i)
    #         K = self.get_layer_window_size(layer_id=i)

    #         mnk_dims_arr.append([M, N, K])

    #     return mnk_dims_arr

    def get_transformed_mnk_dimensions(self) -> list:
        """
        Retrieves the transformed (M, N, K) dimensions for all layers in the topology.
        
        Returns:
            List of [M, N, K] tuples for each layer, where:
            - M: Number of OFMAP pixels
            - N: Number of filters
            - K: Window size (filter_h * filter_w * channels)
        """
        # --- Hyperparameter Check ---
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()

        mnk_dims_arr = []

        # --- Process Each Layer ---
        for layer_id in range(self.num_layers):
            # Retrieve pre-calculated hyperparameters
            ofmap_h, ofmap_w, _, window_sz = self.layers_calculated_hyperparams[layer_id]
            num_filt = self.topo_arrays[layer_id][6]
            
            # Calculate M (OFMAP pixels)
            M = ofmap_h * ofmap_w
            # N is number of filters
            N = num_filt
            # K is window size
            K = window_sz
            
            mnk_dims_arr.append([M, N, K])

        return mnk_dims_arr
    def get_current_topo_name(self):
        current_topo_name = ""
        if self.topo_load_flag:
            current_topo_name = self.current_topo_name
        else:
            print('Error: get_current_topo_name(): Topo file not read')
        return current_topo_name

    def get_num_layers(self):
        if not self.topo_load_flag:
            print("ERROR: topologies.get_num_layers: No array loaded")
            return
        return self.num_layers

    #
    def get_layer_ifmap_dims(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_ifmap_dims: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[1:3]    # Idx = 1, 2

    #
    def get_layer_filter_dims(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_ifmap_dims: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[3:5]    # Idx = 3, 4

    #
    def get_layer_num_filters(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        layer_params = self.topo_arrays[layer_id]
        return layer_params[6]

    def get_layer_num_channels(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        layer_params = self.topo_arrays[layer_id]
        return layer_params[5]

    #
    def get_layer_strides(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_strides: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[7:9]


    def get_layer_window_size(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()
        layer_calc_params = self.layers_calculated_hyperparams[layer_id]
        return layer_calc_params[3]

    def get_layer_num_ofmap_px(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_num_filter: Invalid layer id")
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()
        layer_calc_params = self.layers_calculated_hyperparams[layer_id]
        num_filters = self.get_layer_num_filters(layer_id)
        num_ofmap_px = layer_calc_params[0] * layer_calc_params[1] * num_filters 
        return num_ofmap_px

    def get_layer_ofmap_dims(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_ofmap_dims: Invalid layer id")
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams()
        ofmap_dims = self.layers_calculated_hyperparams[layer_id][0:2]
        return ofmap_dims

    def get_layer_params(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_params: Invalid layer id")
            return
        layer_params = self.topo_arrays[layer_id]
        return layer_params

    def get_layer_id_from_name(self, layer_name=""):
        if (not self.topo_load_flag) or layer_name == "":
            print("ERROR")
            return
        indx = -1
        for i in range(len(self.topo_arrays)):
            if layer_name == self.topo_arrays[i]:
                indx = i
        if indx == -1:
            print("WARNING: Not found")
        return indx

    #
    def get_layer_name(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_name: Invalid layer id")
            return

        name = self.topo_arrays[layer_id][0]
        return str(name)

    #
    def get_layer_names(self):
        if not self.topo_load_flag:
            print("ERROR")
            return
        layer_names = []
        for entry in self.topo_arrays:
            layer_name = str(entry[0])
            layer_names.append(layer_name)
        return layer_names

    def get_layer_mac_ops(self, layer_id=0):
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(topofilename=self.topo_file_name)
        layer_hyper_param = self.layers_calculated_hyperparams[layer_id]
        mac_ops = layer_hyper_param[2]
        return mac_ops

    def get_all_mac_ops(self):
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(topofilename=self.topo_file_name)
        total_mac = 0
        for layer in range(self.num_layers):
            total_mac += self.get_layer_mac_ops(layer)
        return total_mac

    # spatio-temporal dimensions specific to dataflow
    def get_spatiotemporal_dims(self, layer_id=0, df=''):
        if df == '':
            df = self.df
        if not self.topo_calc_spatiotemp_params_flag:
            self.set_spatio_temporal_params()
        df_list = ['os', 'ws', 'is']
        df_idx = df_list.index(df)
        s_row = self.spatio_temp_dim_arrays[layer_id][df_idx][0]
        s_col = self.spatio_temp_dim_arrays[layer_id][df_idx][1]
        t_time = self.spatio_temp_dim_arrays[layer_id][df_idx][2]
        return s_row, s_col, t_time


if __name__ == '__main__':
    tp = topologies()
