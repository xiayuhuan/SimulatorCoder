import configparser as cp
import os
import sys


class scale_config:
    def __init__(self):
        self.run_name = "scale_run"
        # Anand: ISSUE #2. Patch
        self.use_user_bandwidth = False

        self.array_rows = 4
        self.array_cols = 4
        self.ifmap_sz_kb = 256
        self.filter_sz_kb = 256
        self.ofmap_sz_kb = 128
        self.df = 'ws'
        self.ifmap_offset = 0
        self.filter_offset = 10000000
        self.ofmap_offset = 20000000
        self.topofile = ""
        self.bandwidths = []
        self.valid_conf_flag = False

        self.valid_df_list = ['os', 'ws', 'is']

    #
    def read_conf_file(self, conf_file_in):

        me = 'scale_config.' + 'read_conf_file()'

        config = cp.ConfigParser()
        config.read(conf_file_in)

        section = 'general'
        self.run_name = config.get(section, 'run_name')

        # Anand: ISSUE #2. Patch
        section = 'run_presets'
        bw_mode_string = config.get(section, 'InterfaceBandwidth')
        if bw_mode_string == 'USER':
            self.use_user_bandwidth = True
        elif bw_mode_string == 'CALC':
            self.use_user_bandwidth = False
        else:
            message = 'ERROR: ' + me
            message += 'Use either USER or CALC in InterfaceBandwidth feild. Aborting!'
            return

        section = 'architecture_presets'
        self.array_rows = int(config.get(section, 'ArrayHeight'))
        self.array_cols = int(config.get(section, 'ArrayWidth'))
        self.ifmap_sz_kb = int(config.get(section, 'ifmapsramszkB'))
        self.filter_sz_kb = int(config.get(section, 'filtersramszkB'))
        self.ofmap_sz_kb = int(config.get(section, 'ofmapsramszkB'))
        self.ifmap_offset = int(config.get(section, 'IfmapOffset'))
        self.filter_offset = int(config.get(section, 'FilterOffset'))
        self.ofmap_offset = int(config.get(section, 'OfmapOffset'))
        self.df = config.get(section, 'Dataflow')

        # Anand: ISSUE #2. Patch
        if self.use_user_bandwidth:
            self.bandwidths = [int(x.strip())
                               for x in config.get(section, 'Bandwidth').strip().split(',')]

        if self.df not in self.valid_df_list:
            print("WARNING: Invalid dataflow")

        if config.has_section('network_presets'):  # Read network_presets
            self.topofile = config.get(section, 'TopologyCsvLoc').split('"')[1]

        self.valid_conf_flag = True

    # def read_conf_file(self, conf_file_in):
    #     """
    #     Reads and parses configuration file to setup scaling parameters.
    #     Maintains consistency with class initialization defaults.
        
    #     Parameters:
    #         conf_file_in (str): Path to configuration file
            
    #     Returns:
    #         bool: True if configuration was successfully loaded
    #     """
    #     # Initialize ConfigParser
    #     config = cp.ConfigParser()
    #     config.read(conf_file_in)
        
    #     # --- General Parameters ---
    #     self.run_name = config.get('general', 'run_name', fallback='scale_run')
        
    #     # --- Bandwidth Configuration ---
    #     bw_setting = config.get('run_presets', 'InterfaceBandwidth', fallback='CALC').upper()
    #     if bw_setting == 'USER':
    #         self.use_user_bandwidth = True
    #     elif bw_setting == 'CALC':
    #         self.use_user_bandwidth = False
    #     else:
    #         print(f"Error: Invalid bandwidth setting '{bw_setting}'. Must be USER or CALC")
    #         return False
        
    #     # --- Architecture Parameters ---
    #     arch_presets = config['architecture_presets']
    #     self.array_rows = int(arch_presets.get('ArrayRows', '4'))
    #     self.array_cols = int(arch_presets.get('ArrayCols', '4'))
    #     self.ifmap_sz_kb = int(arch_presets.get('IfmapSramKb', '256'))
    #     self.filter_sz_kb = int(arch_presets.get('FilterSramKb', '256'))
    #     self.ofmap_sz_kb = int(arch_presets.get('OfmapSramKb', '128'))
    #     self.ifmap_offset = int(arch_presets.get('IfmapOffset', '0'))
    #     self.filter_offset = int(arch_presets.get('FilterOffset', '10000000'))
    #     self.ofmap_offset = int(arch_presets.get('OfmapOffset', '20000000'))
    #     self.df = arch_presets.get('Dataflow', 'ws').lower()
        
    #     # --- Bandwidth Values (if user-specified) ---
    #     if self.use_user_bandwidth:
    #         bw_str = config.get('run_presets', 'Bandwidth', fallback='')
    #         self.bandwidths = [float(x) for x in bw_str.split(',') if x.strip()]
        
    #     # --- Dataflow Validation ---
    #     if self.df not in self.valid_df_list:
    #         print(f"Warning: Unsupported dataflow '{self.df}'. Using 'ws' instead")
    #         self.df = 'ws'
        
    #     # --- Topology File ---
    #     if 'network_presets' in config:
    #         self.topofile = config.get('network_presets', 'TopologyFile', fallback='')
        
    #     # --- Final Validation ---
    #     self.valid_conf_flag = True
    #     return True

    # def read_conf_file(self, conf_file_in: str) -> None:
    #         """
    #         Reads and parses configuration file to setup scaling parameters.
            
    #         Args:
    #             conf_file_in: Path to configuration file
                
    #         Raises:
    #             ValueError: For invalid bandwidth or dataflow settings
    #             FileNotFoundError: If config file doesn't exist
    #             KeyError: For missing required sections
                
    #         Note:
    #             - Sets valid_conf_flag=True on successful parsing
    #             - Handles both user-defined and calculated bandwidth modes
    #             - Validates dataflow type against supported options
    #         """
    #         from configparser import ConfigParser
    #         import sys

    #         # 1. Initialize parser and read file
    #         config = ConfigParser()
    #         config.read(conf_file_in)

    #         # 2. Parse general settings
    #         self.run_name = config.get('general', 'run_name', fallback='scale_run')

    #         # 3. Handle bandwidth configuration
    #         bw_mode = config.get('run_presets', 'InterfaceBandwidth').upper()
    #         if bw_mode == 'USER':
    #             self.use_user_bandwidth = True
    #             bw_str = config.get('run_presets', 'Bandwidth')
    #             self.bandwidths = [float(x) for x in bw_str.split(',')]
    #         elif bw_mode == 'CALC':
    #             self.use_user_bandwidth = False
    #             self.bandwidths = []
    #         else:
    #             print(f"Error: Invalid InterfaceBandwidth '{bw_mode}'. Must be USER or CALC")
    #             sys.exit(1)

    #         # 4. Parse architecture presets
    #         arch = config['architecture_presets']
    #         self.array_rows = arch.getint('ArrayRows', 8)
    #         self.array_cols = arch.getint('ArrayCols', 8)
    #         self.ifmap_kb = arch.getint('IfmapSramKb', 32)
    #         self.filter_kb = arch.getint('FilterSramKb', 32)
    #         self.ofmap_kb = arch.getint('OfmapSramKb', 32)
    #         self.ifmap_offset = arch.getint('IfmapOffset', 0)
    #         self.filter_offset = arch.getint('FilterOffset', 0)
    #         self.ofmap_offset = arch.getint('OfmapOffset', 0)
            
    #         # 5. Validate dataflow
    #         self.df = config.get('architecture_presets', 'Dataflow').lower()
    #         if self.df not in self.valid_df_list:
    #             print(f"Warning: Unsupported dataflow '{self.df}'. Using default 'os'")
    #             self.df = 'os'  # Default to output stationary

    #         # 6. Handle optional network presets
    #         if 'network_presets' in config:
    #             self.topofile = config.get('network_presets', 'TopologyFile', fallback='')

    #         # 7. Mark configuration valid
    #         self.valid_conf_flag = True

# def read_conf_file(self, conf_file_in: str) -> None:
#     """
#     Optimized configuration file reader with:
#     - Batch section access
#     - Early validation
#     - Vectorized parsing
#     - Proper exception handling
    
#     Args:
#         conf_file_in: Path to configuration file
        
#     Raises:
#         ValueError: For invalid settings
#         FileNotFoundError: If config file doesn't exist
#         KeyError: For missing required sections
#     """
#     from configparser import ConfigParser
#     import sys

#     # 1. Initialize with interpolation disabled for faster parsing
#     config = ConfigParser(interpolation=None)
#     try:
#         files_read = config.read(conf_file_in)
#         if not files_read:
#             raise FileNotFoundError(f"Config file not found: {conf_file_in}")
            
#         # 2. Batch read sections
#         general = config['general']
#         run_presets = config['run_presets']
#         arch = config['architecture_presets']
        
#         # 3. Parse general settings
#         self.run_name = general.get('run_name', 'scale_run')
        
#         # 4. Optimized bandwidth handling
#         bw_mode = run_presets['InterfaceBandwidth'].strip().upper()
#         if bw_mode == 'USER':
#             self.use_user_bandwidth = True
#             try:
#                 self.bandwidths = list(map(float, run_presets['Bandwidth'].split(',')))
#             except ValueError as e:
#                 raise ValueError(f"Invalid bandwidth format: {e}")
#         elif bw_mode == 'CALC':
#             self.use_user_bandwidth = False
#             self.bandwidths = []
#         else:
#             raise ValueError(f"Invalid InterfaceBandwidth '{bw_mode}'. Must be USER or CALC")

#         # 5. Batch architecture settings with defaults
#         self.array_rows = arch.getint('ArrayRows', 8)
#         self.array_cols = arch.getint('ArrayCols', 8)
#         self.ifmap_kb = arch.getint('IfmapSramKb', 32)
#         self.filter_kb = arch.getint('FilterSramKb', 32)
#         self.ofmap_kb = arch.getint('OfmapSramKb', 32)
#         self.ifmap_offset = arch.getint('IfmapOffset', 0)
#         self.filter_offset = arch.getint('FilterOffset', 0)
#         self.ofmap_offset = arch.getint('OfmapOffset', 0)
        
#         # 6. Dataflow validation with early return
#         self.df = arch.get('Dataflow', 'os').lower()
#         if self.df not in self.valid_df_list:
#             self.df = 'os'  # Default to output stationary

#         # 7. Optional network presets
#         if 'network_presets' in config:
#             self.topofile = config['network_presets'].get('TopologyFile', '')

#         # 8. Mark configuration valid
#         self.valid_conf_flag = True

#     except KeyError as e:
#         raise KeyError(f"Missing required section/key: {e}") from None
#     except Exception as e:
#         raise ValueError(f"Configuration error: {e}") from None

    #
    # def update_from_list(self, conf_list):
    #     if not len(conf_list) > 11:
    #         print("ERROR: scale_config.update_from_list: "
    #               "Incompatible number of elements in the list")

    #     self.run_name = conf_list[0]
    #     self.array_rows = int(conf_list[1])
    #     self.array_cols = int(conf_list[2])
    #     self.ifmap_sz_kb = int(conf_list[3])
    #     self.filter_sz_kb = int(conf_list[4])
    #     self.ofmap_sz_kb = int(conf_list[5])
    #     self.ifmap_offset = int(conf_list[6])
    #     self.filter_offset = int(conf_list[7])
    #     self.ofmap_offset = int(conf_list[8])
    #     self.df = conf_list[9]
    #     bw_mode_string = str(conf_list[10])

    #     assert bw_mode_string in ['CALC', 'USER'], 'Invalid mode of operation'
    #     if bw_mode_string == "USER":
    #         assert not len(conf_list) < 12, 'The user bandwidth needs to be provided'
    #         self.bandwidths = conf_list[11]
    #         self.use_user_bandwidth = True
    #     elif bw_mode_string == 'CALC':
    #         self.use_user_bandwidth = False

    #     if len(conf_list) == 15:
    #         self.topofile = conf_list[14]

    #     self.valid_conf_flag = True

    # def update_from_list(self, conf_list):
    #     """
    #     Updates configuration parameters from a provided list.
    #     Supports both basic and extended configuration formats.
        
    #     Parameters:
    #         conf_list (list): Configuration parameters in prescribed order
            
    #     Raises:
    #         AssertionError: For invalid bandwidth mode or insufficient parameters
    #     """
    #     # --- Input Validation ---
    #     if len(conf_list) < 11:
    #         print("Error: Configuration list requires at least 11 elements")
    #         return
        
    #     # --- Basic Parameters ---
    #     self.run_name = str(conf_list[0])
    #     self.array_rows = int(conf_list[1])
    #     self.array_cols = int(conf_list[2])
    #     self.ifmap_sz_kb = int(conf_list[3])
    #     self.filter_sz_kb = int(conf_list[4])
    #     self.ofmap_sz_kb = int(conf_list[5])
    #     self.ifmap_offset = int(conf_list[6])
    #     self.filter_offset = int(conf_list[7])
    #     self.ofmap_offset = int(conf_list[8])
    #     self.df = str(conf_list[9]).lower()
    #     bw_mode_string = str(conf_list[10]).upper()
        
    #     # --- Bandwidth Mode Handling ---
    #     assert bw_mode_string in ['CALC', 'USER'], \
    #         f"Invalid bandwidth mode '{bw_mode_string}'. Must be 'CALC' or 'USER'"
        
    #     if bw_mode_string == 'USER':
    #         assert len(conf_list) >= 12, "USER mode requires bandwidth values"
    #         self.bandwidths = [float(x) for x in str(conf_list[11]).split(',')]
    #         self.use_user_bandwidth = True
    #     else:  # CALC mode
    #         self.use_user_bandwidth = False
        
    #     # --- Topology File (Optional) ---
    #     if len(conf_list) == 15:
    #         self.topofile = str(conf_list[14])
        
    #     # --- Final Validation ---
    #     self.valid_conf_flag = True

    # ICL 
    def update_from_list(self, conf_list: list) -> None:
        """Update configuration parameters from provided list
        
        Args:
            conf_list: Configuration list containing:
                [run_name, array_rows, array_cols, 
                ifmap_sz_kb, filter_sz_kb, ofmap_sz_kb,
                ifmap_offset, filter_offset, ofmap_offset,
                df, bw_mode, (bandwidths), (topofile)]
        """
        # --- Input Validation ---
        if len(conf_list) < 11:
            print("Error: Incompatible number of elements in configuration list")
            return

        # --- Core Parameter Assignment ---  
        self.run_name = conf_list[0]
        self.array_rows = int(conf_list[1])
        self.array_cols = int(conf_list[2])
        self.ifmap_sz_kb = int(conf_list[3])
        self.filter_sz_kb = int(conf_list[4])
        self.ofmap_sz_kb = int(conf_list[5])
        self.ifmap_offset = int(conf_list[6])
        self.filter_offset = int(conf_list[7])
        self.ofmap_offset = int(conf_list[8])
        self.df = int(conf_list[9])
        bw_mode_string = conf_list[10].upper()

        # --- Bandwidth Mode Handling ---
        assert bw_mode_string in ('CALC', 'USER'), \
            "Bandwidth mode must be either 'CALC' or 'USER'"

        if bw_mode_string == 'USER':
            assert len(conf_list) >= 12, \
                "USER mode requires bandwidth values in position 11"
            self.bandwidths = conf_list[11]
            self.use_user_bandwidth = True
        else:  # CALC mode
            self.use_user_bandwidth = False

        # --- Optional Topology File ---
        if len(conf_list) == 15:
            self.topofile = conf_list[14]

        # --- Final Validation ---
        self.valid_conf_flag = True

    #
    def write_conf_file(self, conf_file_out):
        if not self.valid_conf_flag:
            print('ERROR: scale_config.write_conf_file: No valid config loaded')
            return

        config = cp.ConfigParser()

        section = 'general'
        config.add_section(section)
        config.set(section, 'run_name', str(self.run_name))

        section = 'architecture_presets'
        config.add_section(section)
        config.set(section, 'ArrayHeight', str(self.array_rows))
        config.set(section, 'ArrayWidth', str(self.array_cols))

        config.set(section, 'ifmapsramszkB', str(self.ifmap_sz_kb))
        config.set(section, 'filtersramszkB', str(self.filter_sz_kb))
        config.set(section, 'ofmapsramszkB', str(self.ofmap_sz_kb))

        config.set(section, 'IfmapOffset', str(self.ifmap_offset))
        config.set(section, 'FilterOffset', str(self.filter_offset))
        config.set(section, 'OfmapOffset', str(self.ofmap_offset))

        config.set(section, 'Dataflow', str(self.df))
        config.set(section, 'Bandwidth', ','.join([str(x) for x in self.bandwidths]))

        section = 'network_presets'
        config.add_section(section)
        topofile = '"' + self.topofile + '"'
        config.set(section, 'TopologyCsvLoc', str(topofile))

        with open(conf_file_out, 'w') as configfile:
            config.write(configfile)

    def set_arr_dims(self, rows=1, cols=1):
        self.array_rows = rows
        self.array_cols = cols

    #
    def set_dataflow(self, dataflow='os'):
        self.df = dataflow

    #
    def set_buffer_sizes_kb(self, ifmap_size_kb=1, filter_size_kb=1, ofmap_size_kb=1):
        self.ifmap_sz_kb = ifmap_size_kb
        self.filter_sz_kb = filter_size_kb
        self.ofmap_sz_kb = ofmap_size_kb

    #
    def set_topology_file(self, topofile=''):
        self.topofile = topofile

    #
    def set_offsets(self,
                    ifmap_offset=0,
                    filter_offset=10000000,
                    ofmap_offset=20000000
                    ):
        self.ifmap_offset = ifmap_offset
        self.filter_offset = filter_offset
        self.ifmap_offset = ofmap_offset
        self.valid_conf_flag = True

    #
    def force_valid(self):
        self.valid_conf_flag = True

    #
    def set_bw_mode_to_calc(self):
        self.use_user_bandwidth = False

    #
    def use_user_dram_bandwidth(self):
        if not self.valid_conf_flag:
            me = 'scale_config.' + 'use_user_dram_bandwidth()'
            message = 'ERROR: ' + me + ': Configuration is not valid'
            print(message)
            return

        return self.use_user_bandwidth

    #
    def get_conf_as_list(self):
        out_list = []

        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_conf_as_list: Configuration is not valid")
            return

        out_list.append(str(self.run_name))

        out_list.append(str(self.array_rows))
        out_list.append(str(self.array_cols))

        out_list.append(str(self.ifmap_sz_kb))
        out_list.append(str(self.filter_sz_kb))
        out_list.append(str(self.ofmap_sz_kb))

        out_list.append(str(self.ifmap_offset))
        out_list.append(str(self.filter_offset))
        out_list.append(str(self.ofmap_offset))

        out_list.append(str(self.df))
        out_list.append(str(self.topofile))
       
        return out_list

    def get_run_name(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_run_name() : Config data is not valid")
            return

        return self.run_name

    def get_topology_path(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_topology_path() : Config data is not valid")
            return
        return self.topofile

    def get_topology_name(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_topology_name() : Config data is not valid")
            return

        name = self.topofile.split('/')[-1].strip()
        name = name.split('.')[0]

        return name

    def get_dataflow(self):
        if self.valid_conf_flag:
            return self.df

    def get_array_dims(self):
        if self.valid_conf_flag:
            return self.array_rows, self.array_cols

    def get_mem_sizes(self):
        me = 'scale_config.' + 'get_mem_sizes()'

        if not self.valid_conf_flag:
            message = 'ERROR: ' + me
            message += 'Config is not valid. Not returning any values'
            return

        return self.ifmap_sz_kb, self.filter_sz_kb, self.ofmap_sz_kb

    def get_offsets(self):
        if self.valid_conf_flag:
            return self.ifmap_offset, self.filter_offset, self.ofmap_offset

    def get_bandwidths_as_string(self):
        if self.valid_conf_flag:
            return ','.join([str(x) for x in self.bandwidths])

    def get_bandwidths_as_list(self):
        if self.valid_conf_flag:
            return self.bandwidths

    def get_min_dram_bandwidth(self):
        if not self.use_user_dram_bandwidth():
            me = 'scale_config.' + 'get_min_dram_bandwidth()'
            message = 'ERROR: ' + me + ': No user bandwidth provided'
            print(message)
        else:
            return min(self.bandwidths)

    # FIX ISSUE #14
    @staticmethod
    def get_default_conf_as_list():
        dummy_obj = scale_config()
        dummy_obj.force_valid()
        out_list = dummy_obj.get_conf_as_list()
        return out_list
