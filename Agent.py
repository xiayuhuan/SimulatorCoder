import os
import re
import time
import requests
from typing import Optional, Tuple, NamedTuple
import subprocess

class TestResult(NamedTuple):
    success: bool
    error_message: str

class CodeCleaner:
    @staticmethod
    def clean_generated_code(raw_code: str) -> str:
        """
        Ultimate code cleaning solution to ensure pure, correctly indented Python code
        """
        # Remove all code block markers
        code = re.sub(r'```(python)?', '', raw_code)
        
        # Extract all non-empty lines
        lines = [line for line in code.split('\n') if line.strip()]
        
        if not lines:
            raise ValueError("Generated code is empty")

        # Find the first line starting with def or class
        start_idx = next((i for i, line in enumerate(lines) 
                        if re.match(r'^\s*(def|class)\s+', line)), None)
        
        if start_idx is None:
            raise ValueError("No valid class or function definition found")

        # Calculate base indentation
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        # Rebuild code while maintaining relative indentation
        cleaned_lines = []
        for line in lines[start_idx:]:
            original_indent = len(line) - len(line.lstrip())
            adjusted_indent = max(0, original_indent - base_indent)
            cleaned_lines.append(' ' * adjusted_indent + line.lstrip())
        
        cleaned_code = '\n'.join(cleaned_lines).strip()
        
        # Final validation
        if not re.search(r'^\s*(def|class)\s+', cleaned_code, re.MULTILINE):
            raise ValueError("Cleaned code does not contain valid class or function definitions")
            
        return cleaned_code

class CodeManager:
    def __init__(self, simulator_path: str, backup_dir: str = "backups"):
        self.simulator_path = simulator_path
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def _create_backup(self, filename: str) -> str:
        """Create a backup file and return the backup path"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"{filename}.bak.{timestamp}")
        
        src_path = os.path.join(self.simulator_path, filename)
        with open(src_path, 'r', encoding='utf-8') as src, open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
            
        return backup_path

    def _restore_backup(self, backup_path: str, target_file: str):
        """Restore file from backup"""
        target_path = os.path.join(self.simulator_path, target_file)
        with open(backup_path, 'r', encoding='utf-8') as src, open(target_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())

    def replace_class_method(self, filename: str, class_name: str, method_name: str, new_code: str) -> Tuple[bool, str]:
        """
        Safely replace a method in a class while preserving original indentation structure
        Returns (success, error_message)
        """
        filepath = os.path.join(self.simulator_path, filename)
        if not os.path.exists(filepath):
            return False, f"File {filepath} does not exist"
        
        try:
            # Create backup
            backup_path = self._create_backup(filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # 1. Locate class definition
            class_start, class_indent = self._find_class_def(original_lines, class_name)
            if class_start == -1:
                return False, f"Class {class_name} not found"
            
            # 2. Locate method definition
            method_start, method_end = self._find_method_def(
                original_lines[class_start:], 
                method_name,
                class_indent
            )
            if method_start == -1:
                return False, f"Method {method_name} not found"

            method_start += class_start
            method_end += class_start
            
            # 3. Calculate original method indentation
            original_indent = len(original_lines[method_start]) - len(original_lines[method_start].lstrip())
            
            # 4. Adjust new code indentation
            indented_code = self._adjust_indentation(new_code, original_indent)
            
            # 5. Perform replacement
            new_content = (
                original_lines[:method_start] +
                [indented_code + '\n'] +
                original_lines[method_end + 1:]
            )
            
            # 6. Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_content)
            
            return True, ""
            
        except Exception as e:
            # Restore backup
            if 'backup_path' in locals():
                self._restore_backup(backup_path, filename)
            return False, f"Replacement failed: {str(e)}"

    def _find_class_def(self, lines: list, class_name: str) -> Tuple[int, int]:
        """Locate class definition, returns (line_number, indent_level)"""
        for i, line in enumerate(lines):
            if re.match(r'^\s*class\s+' + class_name + r'\b', line):
                indent = len(line) - len(line.lstrip())
                return i, indent
        return -1, 0

    def _find_method_def(self, lines: list, method_name: str, class_indent: int) -> Tuple[int, int]:
        """Locate method definition, returns (start_line, end_line)"""
        start = -1
        end = -1
        method_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            line_indent = len(line) - len(line.lstrip())
            
            # Check if outside class scope
            if line_indent <= class_indent and i > 0:
                break
                
            # Find method definition
            if re.match(r'^\s*def\s+' + method_name + r'\b', line):
                start = i
                method_indent = line_indent
                continue
                
            # Find method end
            if start != -1 and end == -1:
                if line_indent <= method_indent and stripped:
                    end = i - 1
                    
        if start != -1 and end == -1:
            end = len(lines) - 1
            
        return start, end

    def _adjust_indentation(self, code: str, indent_level: int) -> str:
        """Adjust code indentation"""
        indent = ' ' * indent_level
        lines = code.split('\n')
        return '\n'.join(
            indent + line if line.strip() else ''
            for line in lines
        )

class SimulatorTester:
    def __init__(self, simulator_path: str, test_csv: str, config_file: str):
        self.simulator_path = simulator_path
        self.test_csv = test_csv
        self.config_file = config_file
    
    def run_test(self) -> TestResult:
        """Run scalesim test"""
        original_dir = os.getcwd()
        os.chdir(self.simulator_path)
        
        try:
            process = subprocess.Popen(
                ["python", "scale.py", "-t", self.test_csv, "-c", self.config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                return TestResult(True, stdout)
            else:
                error_msg = stderr if stderr else stdout
                return TestResult(False, error_msg)
        except Exception as e:
            return TestResult(False, str(e))
        finally:
            os.chdir(original_dir)

class AutoCodeAgent:
    def __init__(self, config: dict):
        self.config = config
        self.code_cleaner = CodeCleaner()
        self.code_manager = CodeManager(config['simulator_path'])
        self.simulator_tester = SimulatorTester(
            config['simulator_path'],
            config['test_csv'],
            config['config_file']
        )
        self.max_iterations = config.get('max_iterations', 10)
        self.llm_api_key = config['deepseek_api_key']
        self.llm_base_url = config.get('deepseek_base_url', "https://api.deepseek.com/v1")

    def run(self, prompt_file: str, target_file: str, class_name: str, method_name: str):
        """Main execution method"""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read()
        except Exception as e:
            print(f"Failed to read prompt file: {str(e)}")
            return

        context = ""
        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Generate code
            print("Generating code...")
            raw_code = self._call_llm_api(prompt, context)
            if not raw_code:
                print("Code generation failed")
                continue
                
            # Clean code
            try:
                clean_code = self.code_cleaner.clean_generated_code(raw_code)
                print(f"Generated code:\n{clean_code}")
            except ValueError as e:
                context = self._build_error_context("Code format error", str(e), clean_code if 'clean_code' in locals() else raw_code)
                print(context)
                continue
            
            # Replace code
            print(f"Replacing method {class_name}.{method_name} in {target_file}...")
            success, error_msg = self.code_manager.replace_class_method(
                target_file, class_name, method_name, clean_code
            )
            if not success:
                context = self._build_error_context("Code replacement failed", error_msg, clean_code)
                print(context)
                continue
            
            # Run test
            print("Running simulator test...")
            test_result = self.simulator_tester.run_test()
            if test_result.success:
                print("Test passed!")
                print("Output:", test_result.error_message)
                print("Code update successful!")
                return
            else:
                context = self._build_error_context("Test failed", test_result.error_message, clean_code)
                print(context)
        
        print(f"Reached maximum iterations ({self.max_iterations}), no valid solution found")

    def _call_llm_api(self, prompt: str, context: str) -> Optional[str]:
        """Call DeepSeek API to generate code"""
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        # Build more detailed prompt
        full_prompt = f"""Please generate Python code based on the following requirements and error feedback:
        
Original requirements:
{prompt}

Error feedback and context:
{context}

Please strictly follow these requirements:
1. Only generate method implementation code
2. Do not include class definitions or other methods
3. Do not include any explanations or Markdown markers
4. Maintain the same indentation style as the original code
5. Ensure the method signature exactly matches the original code
6. Pay special attention to the issues mentioned in the error feedback"""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.3,  # Reduce randomness for better accuracy
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API call failed: {str(e)}")
            return None

    def _build_error_context(self, error_type: str, error_message: str, generated_code: str = "") -> str:
        """Build detailed error feedback context"""
        context = f"""Previous attempt failed, please correct the code based on the following information:

Error type: {error_type}
Error details:
{error_message}

"""
        if generated_code:
            context += f"""Previously generated code:
{generated_code}

Problem analysis:
1. Check if method signature is correct
2. Check if indentation is correct
3. Ensure all variables are properly defined
4. Check for syntax errors
"""
        return context

if __name__ == "__main__":
    # Configuration parameters
    config = {
        'deepseek_api_key': 'xxx',  # Replace with your DeepSeek API key
        'simulator_path': 'path/scalesim',   # Simulator code path
        'test_csv': 'path/scalesi/topologies/conv_nets/test.csv',  # Test CSV file path
        'config_file': 'path/scalesim/configs/scale.cfg',      # Config file path
        # 'gcc_path': 'D:\LeStoreDownload\mingw\mingw64\mingw64\bin\gcc.exe',   # GCC path
        'max_iterations': 5   # Maximum iterations
    }
    
    # Create and run agent
    agent = AutoCodeAgent(config)
    
    # Execute code generation and replacement
    agent.run(
        prompt_file="path/scalesim/prompt template/xxx.txt",         # Prompt file
        target_file="path/scalesim/scale_config.py",    # Target file
        class_name="scale_config",        # Class name
        method_name="read_conf_file"           # Method name
    )