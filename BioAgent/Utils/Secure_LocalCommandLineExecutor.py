# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

# NOTE: This file extends AG2 LocalCommandLineExecutor with minimal changes:
# 1. Added pip_install_dir parameter for controlling where packages install
# 2. Enhanced sanitize_command with additional security patterns
# 3. Added pip_filter to skip already-installed packages
# 4. Improved timeout handling with process group cleanup (Unix only)
# 5. Added output truncation to prevent context window overflow

import logging
import os
import re
import signal
import subprocess
import sys
from hashlib import md5
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from autogen.code_utils import PYTHON_VARIANTS, TIMEOUT_MSG, WIN32, _cmd
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.coding.base import CodeBlock, CommandLineCodeResult
from autogen.coding.utils import _get_file_name_from_content, silence_pip

__all__ = ("SecureLocalCommandLineExecutor",)


class SecureLocalCommandLineExecutor(LocalCommandLineCodeExecutor):
    """
    Extended LocalCommandLineExecutor with security enhancements for LLM-generated code.
    
    Changes from base class:
    - pip_install_dir: Control where pip packages are installed
    - Enhanced command sanitization (blocks curl|sh, sudo, etc.)
    - Pip package filtering to skip already-installed packages
    - Better subprocess cleanup on timeout
    - Output truncation to prevent context window overflow
    """

    def __init__(
        self,
        timeout: int = 60,
        virtual_env_context: SimpleNamespace | None = None,
        work_dir: Path | str = Path(),
        functions: list = [],
        functions_module: str = "functions",
        execution_policies: dict[str, bool] | None = None,
        # ========== CHANGE 1: New parameters ==========
        pip_install_dir: Path | str | None = None,  # Where pip packages install
        inputs_dir: Path | str | None = None,       # Optional: reference to read-only inputs
        enable_pip_filter: bool = True,              # Whether to filter pip installs
        max_output_chars: int = 5000,                # Maximum output length to prevent context overflow
        # ==============================================
    ):
        """
        Initialize SecureLocalCommandLineExecutor.
        
        Args:
            (standard LocalCommandLineExecutor args...)
            
            pip_install_dir: Directory for pip package installations. 
                           Defaults to work_dir if not specified.
                           This allows separating code execution from package storage.
            
            inputs_dir: Optional path to read-only input data directory.
                       For reference only - doesn't affect execution.
            
            enable_pip_filter: If True, filters out already-installed packages from
                             shell script pip install commands to avoid redundant installs.
            
            max_output_chars: Maximum characters in output. If exceeded, shows first and last
                            portions with truncation message. Prevents context window overflow.
                            Default: 5000 (shows first 2500 + last 2500 chars)
        """
        # Initialize parent class normally
        super().__init__(
            timeout=timeout,
            virtual_env_context=virtual_env_context,
            work_dir=work_dir,
            functions=functions,
            functions_module=functions_module,
            execution_policies=execution_policies,
        )
        
        # ========== CHANGE 1: Setup pip_install_dir ==========
        # If not specified, use work_dir (same as default behavior)
        if pip_install_dir is None:
            pip_install_dir = work_dir
        
        self.pip_install_dir = Path(pip_install_dir) if isinstance(pip_install_dir, str) else pip_install_dir
        self.pip_install_dir.mkdir(parents=True, exist_ok=True)
        # =====================================================
        
        # Store inputs_dir for reference (optional)
        self.inputs_dir = Path(inputs_dir) if inputs_dir else None
        if self.inputs_dir:
            self.inputs_dir.mkdir(parents=True, exist_ok=True)
        
        # ========== CHANGE 2: Pip filtering setup ==========
        # Get list of installed packages to avoid reinstalling
        self._enable_pip_filter = enable_pip_filter
        self._installed_packages: set[str] | None = None
        if enable_pip_filter:
            self._installed_packages = self._get_installed_packages()
        # ===================================================
        
        # ========== CHANGE 3: Output truncation setup ==========
        # Store max output length to prevent context window overflow
        self._max_output_chars = max_output_chars
        # =======================================================
        
        self.logger = logging.getLogger(__name__)

    # ========== CHANGE 4: New method to get installed packages ==========
    def _get_installed_packages(self) -> set[str]:
        """
        Query pip to get set of currently installed package names.
        Used by pip_filter to avoid reinstalling packages.
        
        Returns:
            Set of lowercase package names (e.g., {'numpy', 'pandas', 'matplotlib'})
        """
        try:
            py_executable = self._virtual_env_context.env_exe if self._virtual_env_context else sys.executable
            result = subprocess.run(
                [py_executable, "-m", "pip", "list", "--format=freeze"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                packages = set()
                for line in result.stdout.splitlines():
                    if "==" in line:
                        # Extract package name (before version)
                        pkg_name = line.split("==")[0].lower()
                        packages.add(pkg_name)
                return packages
        except Exception as e:
            self.logger.warning(f"Could not retrieve package list: {e}")
        return set()
    # ====================================================================

    # ========== CHANGE 4: Enhanced sanitize_command ==========
    @staticmethod
    def sanitize_command(lang: str, code: str) -> None:
        """
        Enhanced security check - extends base class with additional dangerous patterns.
        
        Original patterns (from LocalCommandLineExecutor):
        - rm -rf
        - mv to /dev/null
        - dd command
        - Writing to disk blocks
        - Fork bombs
        
        NEW patterns added:
        - curl/wget piped to shell (curl | sh)
        - sudo commands
        - chmod 777
        - mkfs (filesystem creation)
        - chown (ownership changes)
        - Inline python execution (python -c)
        - Non-PyPI pip installs (git+, file://, VCS, etc.)
        - Alternative package managers (conda, apt, yum)
        """
        dangerous_patterns = [
            # Original patterns from base class:
            (r"\brm\s+-rf\b", "Use of 'rm -rf' command is not allowed."),
            (r"\bmv\b.*?\s+/dev/null", "Moving files to /dev/null is not allowed."),
            (r"\bdd\b", "Use of 'dd' command is not allowed."),
            (r">\s*/dev/sd[a-z][1-9]?", "Overwriting disk blocks directly is not allowed."),
            (r":\(\)\{\s*:\|\:&\s*\};:", "Fork bombs are not allowed."),
            
            # NEW: Additional security patterns
            (r"\bcurl\b.*\|.*\bsh\b", "Downloading and executing code with curl | sh is not allowed."),
            (r"\bwget\b.*\|.*\bsh\b", "Downloading and executing code with wget | sh is not allowed."),
            (r"\bpython\s+-c\s+", "Running inline Python code is not allowed."),
            (r"\bsudo\b", "Use of 'sudo' command is not allowed."),
            (r"\bmkfs\.", "Filesystem creation (mkfs) is not allowed."),
            (r"\bchown\b", "Changing ownership with chown is not allowed."),
            (r"\bchmod\s+[^\n;]*777\b", "Making files globally writable with chmod 777 is not allowed."),
        ]
        
        # Check dangerous patterns for shell scripts
        if lang in ["bash", "shell", "sh"]:
            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    raise ValueError(f"Potentially dangerous command detected: {message}")
        
        # NEW: Validate pip installs for all languages
        SecureLocalCommandLineExecutor._validate_pip_installs(code)

    @staticmethod
    def _validate_pip_installs(code: str) -> None:
        """
        NEW method: Ensure pip installs only come from PyPI.
        
        Blocks:
        - pip install from URLs (file://, http://, https://)
        - pip install from VCS (git+, svn+, hg+, bzr+)
        - pip install with dangerous flags (--index-url, --find-links with URLs)
        - Alternative package managers (conda, apt-get, yum, etc.)
        
        This prevents LLM from installing malicious packages from untrusted sources.
        """
        pip_install_pattern = r"pip\s+install\s+([^\n;]+)"
        
        for line in code.splitlines():
            pip_match = re.search(pip_install_pattern, line)
            if pip_match:
                install_args = pip_match.group(1)
                args = install_args.split()
                
                # Check for flags that could point to non-PyPI sources
                dangerous_flags = [
                    "--find-links", "--index-url", "--extra-index-url",
                    "--trusted-host", "--no-index",
                ]
                for i, arg in enumerate(args):
                    if arg in dangerous_flags:
                        if i + 1 < len(args):
                            next_arg = args[i + 1]
                            # Block if flag points to URL
                            if re.match(r"^(file|https?|ftp)://", next_arg):
                                raise ValueError(
                                    f"Non-PyPI pip install flag detected and blocked: '{arg} {next_arg}'"
                                )
                            # Block if flag points to file path
                            if next_arg.startswith(("/", "./", "../")):
                                raise ValueError(
                                    f"Non-PyPI pip install flag with file path blocked: '{arg} {next_arg}'"
                                )
                
                # Check package names/URLs
                for arg in args:
                    if arg.startswith("-"):
                        continue  # Skip flags
                    
                    # Block VCS protocols (e.g., package@git+https://...)
                    if re.search(r"@(git\+|svn\+|hg\+|bzr\+)", arg, re.IGNORECASE):
                        raise ValueError(f"VCS-based pip install detected and blocked: '{arg}'")
                    
                    # Block direct URLs
                    if re.match(r"^(file|https?|ftp)://", arg):
                        raise ValueError(f"URL-based pip install detected and blocked: '{arg}'")
                    
                    # Block git+ prefix
                    if arg.lower().startswith("git+"):
                        raise ValueError(f"Git-based pip install detected and blocked: '{arg}'")
            
            # Block alternative package managers
            if re.search(r"\b(conda|apt-get|apt|yum|dnf|apk)\s+install\b", line):
                raise ValueError(f"Non-PyPI package manager install blocked: '{line.strip()}'")
    # ==========================================================

    # ========== CHANGE 5: New method to filter pip installs ==========
    def pip_filter(self, shell_script: str, installed_packages: set[str]) -> tuple[str, bool]:
        """
        NEW method: Filter pip install commands to remove already-installed packages.
        
        This improves performance by skipping redundant installations when LLM
        generates code that installs packages already present in the environment.
        
        Example:
            Input:  "pip install numpy pandas scipy"
            If numpy and pandas are already installed:
            Output: "pip install scipy"
        
        Args:
            shell_script: Shell script code (bash/sh)
            installed_packages: Set of lowercase package names already installed
        
        Returns:
            (filtered_script, should_skip)
            - filtered_script: Script with filtered pip install lines
            - should_skip: True if script has no meaningful commands left
        """
        def _filter_one_install_line(install_line: str) -> tuple[str | None, bool]:
            """Filter a single pip install line, keeping only uninstalled packages."""
            tokens = install_line.strip().split()
            result_tokens = []
            has_packages = False
            
            for token in tokens:
                if token.startswith("-"):
                    # Keep flags (e.g., -q, --upgrade)
                    result_tokens.append(token)
                else:
                    # Extract package name (before version specifier like ==, >=, etc.)
                    pkg = re.split(r"[=<>!]", token)[0].lower()
                    
                    # Only keep if not already installed
                    if pkg not in installed_packages:
                        result_tokens.append(token)
                        has_packages = True
            
            if has_packages:
                return " ".join(result_tokens), True
            else:
                return None, False
        
        pattern = r"pip\s+install\s+([^\n;]+)"
        new_lines = []
        has_meaningful_content = False
        
        for line in shell_script.splitlines():
            match = re.search(pattern, line)
            
            if match:
                # This line contains pip install
                pkgs_str = match.group(1)
                filtered_install, has_packages = _filter_one_install_line(pkgs_str)
                
                if has_packages:
                    # Some packages still need installation
                    new_lines.append(f"pip install {filtered_install}")
                    has_meaningful_content = True
                # If all packages already installed, skip this line entirely
            else:
                # Not a pip install line - keep as is
                new_lines.append(line)
                # Check if this line is meaningful (not empty, not just comment)
                if line.strip() and not line.strip().startswith("#"):
                    has_meaningful_content = True
        
        filtered_script = "\n".join(new_lines)
        should_skip = not has_meaningful_content
        
        return filtered_script, should_skip
    # =================================================================

    # ========== CHANGE 6: New method to truncate output ==========
    def _truncate_output(self, output: str) -> str:
        """
        NEW method: Truncate output to prevent context window overflow.
        
        If output exceeds max_output_chars, shows:
        - First half of max_output_chars (e.g., first 2500 chars)
        - Truncation message with character counts
        - Last half of max_output_chars (e.g., last 2500 chars)
        
        This ensures agents get both the beginning (often containing errors/warnings)
        and the end (final results) while staying within context limits.
        
        Args:
            output: Raw output string from code execution
        
        Returns:
            Truncated output if needed, otherwise original output
        """
        if len(output) <= self._max_output_chars:
            # Output is within limit, return as-is
            return output
        
        # Calculate how much to show from start and end
        half_limit = self._max_output_chars // 2
        
        # Get first and last portions
        first_part = output[:half_limit]
        last_part = output[-half_limit:]
        
        # Calculate how many characters were truncated
        truncated_count = len(output) - self._max_output_chars
        
        # Create truncation message
        truncation_msg = (
            f"\n\n... [Output truncated: {truncated_count} characters hidden] ...\n"
            f"... [Total output length: {len(output)} characters] ...\n"
            f"... [Showing first {half_limit} and last {half_limit} characters] ...\n\n"
        )
        
        # Combine parts
        return first_part + truncation_msg + last_part
    # ================================================================

    # ========== CHANGE 7: Override execute method with enhancements ==========
    def _execute_code_dont_check_setup(self, code_blocks: list[CodeBlock]) -> CommandLineCodeResult:
        """
        Execute code blocks - OVERRIDES parent to add:
        1. Enhanced security validation
        2. Pip package filtering for shell scripts
        3. Better subprocess cleanup on timeout (Unix only)
        4. Output truncation to prevent context window overflow
        
        Most logic is identical to parent class, with changes marked below.
        """
        logs_all = ""
        file_names = []
        
        for code_block in code_blocks:
            lang, code = code_block.language, code_block.code
            lang = lang.lower()

            # CHANGE: Use enhanced sanitize_command (has additional security checks)
            try:
                SecureLocalCommandLineExecutor.sanitize_command(lang, code)
            except ValueError as e:
                return CommandLineCodeResult(exit_code=1, output=f"Security violation: {str(e)}\n")
            
            code = silence_pip(code, lang)

            if lang in PYTHON_VARIANTS:
                lang = "python"

            if WIN32 and lang in ["sh", "shell"]:
                lang = "ps1"

            if lang not in self.SUPPORTED_LANGUAGES:
                exitcode = 1
                logs_all += "\n" + f"unknown language {lang}"
                break

            execute_code = self.execution_policies.get(lang, False)
            
            # CHANGE: Filter pip installs for shell scripts to skip already-installed packages
            if self._enable_pip_filter and lang in ["bash", "shell", "sh"] and self._installed_packages:
                filtered_code, skip = self.pip_filter(code, self._installed_packages)
                if skip:
                    # All packages already installed or no meaningful commands
                    logs_all += "Skipped: All packages already installed\n"
                    continue
                code = filtered_code
            
            try:
                filename = _get_file_name_from_content(code, self._work_dir)
            except ValueError:
                return CommandLineCodeResult(exit_code=1, output="Filename is not in the workspace")

            if filename is None:
                code_hash = md5(code.encode()).hexdigest()
                filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"
            
            written_file = (self._work_dir / filename).resolve()
            with written_file.open("w", encoding="utf-8") as f:
                f.write(code)
            file_names.append(written_file)

            if not execute_code:
                logs_all += f"Code saved to {written_file!s}\n"
                exitcode = 0
                continue

            program = _cmd(lang)
            # Fix for containers with only python3
            if program == "python":
                program = "python3"
            cmd = [program, str(written_file.absolute())]
            env = os.environ.copy()
            
            if self._virtual_env_context:
                virtual_env_abs_path = os.path.abspath(self._virtual_env_context.bin_path)
                path_with_virtualenv = rf"{virtual_env_abs_path}{os.pathsep}{env['PATH']}"
                env["PATH"] = path_with_virtualenv
                if WIN32:
                    activation_script = os.path.join(virtual_env_abs_path, "activate.bat")
                    cmd = [activation_script, "&&", *cmd]
            
            # CHANGE: Add pip_install_dir to PYTHONPATH if it differs from work_dir
            # This ensures pip packages installed to pip_install_dir are importable
            if self.pip_install_dir != self._work_dir:
                python_path = env.get("PYTHONPATH", "")
                if python_path:
                    env["PYTHONPATH"] = f"{self.pip_install_dir}{os.pathsep}{python_path}"
                else:
                    env["PYTHONPATH"] = str(self.pip_install_dir)
                env["PIP_TARGET"] = str(self.pip_install_dir)

            # CHANGE: Enhanced subprocess handling with process group cleanup (Unix only)
            # On Windows, use original subprocess.run approach
            # On Unix, use Popen with process groups for better timeout cleanup
            proc = None
            try:
                if WIN32:
                    # Windows: use standard subprocess.run (same as parent class)
                    result = subprocess.run(
                        cmd,
                        cwd=self._work_dir,
                        capture_output=True,
                        text=True,
                        timeout=float(self._timeout),
                        env=env,
                        encoding="utf-8",
                    )
                    logs_all += result.stderr + result.stdout
                    exitcode = result.returncode
                else:
                    # Unix: use process groups for better cleanup
                    # os.setsid() creates new process group so we can kill all child processes
                    proc = subprocess.Popen(
                        cmd,
                        cwd=self._work_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                        encoding="utf-8",
                        preexec_fn=os.setsid,  # Create new process group
                    )
                    
                    # Wait for completion with timeout
                    stdout, stderr = proc.communicate(timeout=float(self._timeout))
                    logs_all += stderr + stdout
                    exitcode = proc.returncode
                    
            except subprocess.TimeoutExpired:
                # CHANGE: Kill entire process group on timeout (Unix only)
                # This ensures child processes are also terminated
                if proc and not WIN32:
                    try:
                        # Try graceful termination first (SIGTERM)
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        proc.wait(timeout=5)
                    except:
                        try:
                            # Force kill if graceful didn't work (SIGKILL)
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except:
                            pass
                
                logs_all += "\n" + TIMEOUT_MSG
                exitcode = 124  # Standard timeout exit code
                break
            
            finally:
                # CHANGE: Ensure process is cleaned up
                if proc and not WIN32:
                    try:
                        if proc.poll() is None:  # Process still running
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except:
                        pass

            if exitcode != 0:
                break

        # CHANGE: Truncate output before returning to prevent context window overflow
        logs_all = self._truncate_output(logs_all)
        
        code_file = str(file_names[0]) if len(file_names) > 0 else None
        return CommandLineCodeResult(exit_code=exitcode, output=logs_all, code_file=code_file)
    # =========================================================================