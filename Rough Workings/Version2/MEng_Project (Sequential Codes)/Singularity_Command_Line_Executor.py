# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import atexit
import logging
import uuid
from hashlib import md5
from pathlib import Path
from time import sleep
from types import TracebackType
from typing import Any, ClassVar
import os

#import docker
#from docker.errors import ImageNotFound
from typing_extensions import Self

# Absolute imports (replacement)
from autogen.code_utils import TIMEOUT_MSG, _cmd
from autogen. doc_utils import export_module
from autogen.coding.base import CodeBlock, CodeExecutor, CodeExtractor, CommandLineCodeResult
from autogen.coding.markdown_code_extractor import MarkdownCodeExtractor
from autogen.coding.utils import _get_file_name_from_content, silence_pip

# Added Imports:
import subprocess
import re


# Removed Wait for Ready function as Singularity does not need it due to Sync blcoks.
# Note: This system will try to emulate the way docker works and will have to create 
#Singularity containers that persist. By default, they don't persist and so running 2 code 
# blocks will always result in some error as the previous code setup will be lost.


__all__ = ("SingularityCommandLineCodeExecutor",)


@export_module("autogen.coding")
class SingularityCommandLineCodeExecutor(CodeExecutor):
    DEFAULT_EXECUTION_POLICY: ClassVar[dict[str, bool]] = {
        "bash": True,
        "shell": True,
        "sh": True,
        "pwsh": True,
        "powershell": True,
        "ps1": True,
        "python": True,
        "javascript": False,
        "html": False,
        "css": False,
    }
    LANGUAGE_ALIASES: ClassVar[dict[str, str]] = {"py": "python", "js": "javascript"}

    def __init__(
        self,
        image: str = "python:3-slim", # Defaults to python:3-slim image if another is not provided.
        container_name: str | None = None,
        timeout: int = 1000, # About 16 mins.
        work_dir: Path | str | None = None, # work_dir is where code files and execution results are saved. These are on host directory.
        bind_dir: Path | str | None = None, # bind_dir is mainly used for nested containers. It defalts to work_dir.
        setup_dir: Path | str | None= None, # This is where we can store the Singularity image files, rather than in home directory.
        inputs_dir: Path | str | None=None, # If the code needs to read any files to execute code, due to --containall and --no-home, inputs directory must be explicitly bound.
        #auto_remove deleted.
        pip_install_dir:Path | str | None=None, # Directory to use instad of writable-tmpfs for pip installs. Useful if memory constraints are an issue.
        stop_container: bool = True,
        execution_policies: dict[str, bool] | None = None,
        *,
        instance_startup_kwargs: dict[str, Any] | None = None,):
        """(Experimental) A code executor class that executes code through
        a command line environment in a Docker container.

        The executor first saves each code block in a file in the working
        directory, and then executes the code file in the container.
        The executor executes the code blocks in the order they are received.
        Currently, the executor only supports Python and shell scripts.
        For Python code, use the language "python" for the code block.
        For shell scripts, use the language "bash", "shell", or "sh" for the code
        block.

        Args:
            image: Docker image to use for code execution. Defaults to "python:3-slim". This is a Docker Image, but Singularity
            can automatically convert to .SIF file. If a .SIF file is already present, this will be used instead

            container_name: Name of the Singularity container which is created. If None, will autogenerate a name. Defaults to None.

            timeout: The timeout for code execution. Defaults to 1000.

            work_dir: The working directory for the code execution. Defaults to Path(".").

            bind_dir: The directory that will be bound to the code executor container. Useful for cases where you want to spawn
                the container from within a container. Defaults to work_dir.

            setup_dir: The directory where the Singularity image files are stored. Defaults to None.

            stop_container: If true, will automatically stop the
                container when stop is called, when the context manager exits or when
                the Python process exits with atext. Defaults to True.

            execution_policies: A dictionary mapping language names to boolean values that determine
                whether code in that language should be executed. True means code in that language
                will be executed, False means it will only be saved to a file. This overrides the
                default execution policies. Defaults to None.

            instance_startup_kwargs: Not Supported in this version of the Code Executor.


        Raises:
            ValueError: On argument error, or if the container fails to start.
        """
        #1# Sets bind_dir to work_dir if not provided elsewhere and sets path for work_dir.
        work_dir = work_dir if work_dir is not None else Path()

        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)
        work_dir.mkdir(exist_ok=True)

        if bind_dir is None:
            bind_dir = work_dir
        elif isinstance(bind_dir, str):
            bind_dir = Path(bind_dir)
        if setup_dir is None:
            setup_dir = work_dir
        if isinstance(setup_dir, str):
            setup_dir = Path(setup_dir)
        setup_dir.mkdir(exist_ok=True)
        if inputs_dir is None:
            inputs_dir = work_dir
        if isinstance(inputs_dir, str):
            inputs_dir = Path(inputs_dir)
        inputs_dir.mkdir(exist_ok=True)

        #++++++++ Alternative to Writable TMPFS if memory constraints are an issue: ++++++++
        if pip_install_dir is not None:
            if isinstance(pip_install_dir, str):
                pip_install_dir = Path(pip_install_dir)
            pip_install_dir.mkdir(exist_ok=True)
            pip_cache_dir = pip_install_dir /".cache"
            pip_cache_dir.mkdir(exist_ok=True)
            temp_dir = pip_install_dir /"tmp"
            temp_dir.mkdir(exist_ok=True)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #---------------------------------------------------------------------------------

        #2# Need to pull the correct image using Singularity conventions. This checks for a singularity image
        # in the setup files. If it can't be found, it tries to pull form docker and convert. If code used 
        # again, then the image is now in setup files in the home directory (Separate from other files).
        self.image_new=image.replace(":", "-") # Files can't have :
        self.image_new=self.image_new.replace("/", "-") # Files can't have / as this is reserved for creating a nested directory.
        self.singularity_image = f"{self.image_new}.sif"
        self.docker_image = f"docker://{image}"

        singularity_path = setup_dir/self.singularity_image
        singularity_path = setup_dir / self.singularity_image

        if not singularity_path.exists():
            logging.info(f"Singularity image not found.  Pulling {self.docker_image}...")
            result = subprocess.run(
                ["singularity", "pull", str(singularity_path), self.docker_image],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise ValueError(f"Failed to pull image {self. docker_image}: {result.stderr}")

        self._image_path = singularity_path
        #---------------------------------------------------------------------------------

        if container_name is None:
            container_name = f"autogen-code-exec-{uuid.uuid4()}"
        self._instance_name = container_name

        #3# Need to start a PERSISTENT Singularity container. Need to isolate from the home directory, and only
        # allow access to work_dir and bind_dir. Need to allow a temporary overlay for pip installs, as .sif
        # file is read only.

        
        # --containall: Full isolation (PID, IPC, environment, filesystem)
        # --no-home: Don't mount user's home directory
        # --writable-tmpfs: Allow pip install (temporary, lost on stop)
        # --bind: Only expose work_dir as /workspace

        self.inputs_dir=inputs_dir
        inputs_path=self.inputs_dir
        instance_start_cmd = [
            "singularity", "instance", "start",
            "--nv",  # Enable NVIDIA GPU support
            "--containall",
            #"--no-home",
            "--bind", f"{pip_install_dir.resolve()}:/home/fakeuser",  # Alternative to writable-tmpfs for pip installs. Needs to be writable to allow agents to install packages on the go.
            "--bind", f"{temp_dir.resolve()}:/tmp:rw",  # Temporary directory inside container
            "--home", "/home/fakeuser",
            #"--writable-tmpfs", # Memory limits are possible, depending on sessiondir max size. Requires root to modify.
            "--bind", f"{bind_dir.resolve()}:/workspace",
            "--bind", f"{inputs_path.resolve()}:/inputs:ro", # Read only to prevent warping data.
            # "--env", "SINGULARITYENV_PYTHONPATH=/pip_install",
            "--env", "SINGULARITYENV_TMPDIR=/tmp",
            "--env", "SINGULARITYENV_PIP_CACHE_DIR=/home/fakeuser/.cache",
            #"--env", "SINGULARITYENV_PIP_TARGET=/home/fakeuser",
            "--env", "SINGULARITYENV_HOME=/home/fakeuser",  # This prevents user installation fallback
            #"--env", "SINGULARITYENV_PYTHONUSERBASE=/home/fakeuser",
            #"--env", "SINGULARITYENV_USER=fakeuser",
            str(self._image_path),
            self._instance_name
        ]

        result = subprocess.run(
            instance_start_cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise ValueError(f"Failed to start Singularity instance: {result.stderr}")
        #---------------------------------------------------------------------------------

        #4# Cleanup function that can be called to stop the container from running.    #4# Cleanup function to stop instance on exit
        def cleanup() -> None:
            try:
                subprocess.run(
                    ["singularity", "instance", "stop", self._instance_name],
                    capture_output=True,
                    text=True
                )
            except Exception:
                pass
            atexit.unregister(cleanup)

        if stop_container:
            atexit.register(cleanup)

        self._cleanup = cleanup
        #---------------------------------------------------------------------------------

        #5# Verify instance is running and store instance variables
        # Check if the instance is running
        list_result = subprocess.run(
            ["singularity", "instance", "list"],
            capture_output=True,
            text=True
        )

        if container_name not in list_result. stdout:
            raise ValueError(f"Failed to start Singularity instance from image {image}.")
        else:
            logging.info(f"Singularity instance {self._instance_name} is running.")

        self._timeout = timeout
        self._work_dir: Path = work_dir
        self._bind_dir: Path = bind_dir
        self._container_name = container_name
        self. execution_policies = self.DEFAULT_EXECUTION_POLICY.copy()
        if execution_policies is not None:
            self. execution_policies.update(execution_policies)
        #---------------------------------------------------------------------------------

    @property
    def timeout(self) -> int:
        """(Experimental) The timeout for code execution."""
        return self._timeout

    @property
    def work_dir(self) -> Path:
        """(Experimental) The working directory for the code execution."""
        return self._work_dir

    @property
    def bind_dir(self) -> Path:
        """(Experimental) The binding directory for the code execution container."""
        return self._bind_dir

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) Export a code extractor that can be used by an agent."""
        return MarkdownCodeExtractor()
    # No changes needed for previous section.

    # New Function ---- Intercept Function: If the user encounters memory issues with using --tmpfs,
    # then, this function will get the pip list and then it can be used with a filter function later
    # to get rid of packages that are not needed and reduce risk of running out of memory. If memory
    # is not an issue, then this won't get in the way.
    def get_pip_list(self):
        result = subprocess.run(
            ['singularity', 'exec', f'instance://{self._container_name}', 'python', '-m', 'pip', 'list'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Could not retrieve pip list: {result.stderr}")
        pkgs = set()
        # Parse output, skip header lines
        for line in result.stdout.splitlines():
            if line.strip() and not line.lower().startswith(('package', 'version', '-----')):
                pkgs.add(line.split()[0].lower())
        return pkgs
    # New Function ---- Works wth the pip list function to remove any installations already
    # in image from the shell blocks before executing code.
    def pip_filter(self, shell_script: str, installed_packages: set[str]) -> tuple[str, bool]:
        """
        Modifies the pip install lines in the given shell script,
        removing packages already present in installed_packages. 
        
        Returns:
            tuple[str, bool]: A tuple containing:
                - The filtered shell script (str)
                - skip (bool): True if the entire script should be skipped 
                (no meaningful commands remain), False otherwise
        """
        
        def _filter_one_install_line(install_line):
            """
            Filters a single pip install line, removing already-installed packages. 
            
            Args:
                install_line: The part after 'pip install', e.g., '-qqq numpy pandas'
            
            Returns:
                tuple: (filtered_line or None, has_packages)
                    - filtered_line: The filtered arguments, or None if no packages remain
                    - has_packages: True if actual packages (not just flags) remain
            """
            tokens = install_line.strip().split()
            result_tokens = []
            has_packages = False
            
            for token in tokens:
                if token.startswith('-'):
                    result_tokens.append(token)
                else:
                    pkg = re.split(r'[=<>! ]', token)[0].lower()
                    
                    if pkg not in installed_packages:
                        result_tokens.append(token)
                        has_packages = True
            
            if has_packages:
                return ' '.join(result_tokens), True
            else:
                return None, False

        pattern = r'(pip\s+install\s+([^\n;]+))'
        
        new_lines = []
        has_any_meaningful_content = False  # Tracks if the script has ANY meaningful commands
        
        for line in shell_script.splitlines():
            m = re.match(pattern, line)
            
            if m:
                # This line is a pip install command
                pkgs_str = m.group(2)
                filtered_install, has_packages = _filter_one_install_line(pkgs_str)
                
                if has_packages:
                    # There are still packages to install
                    new_lines.append(f"pip install {filtered_install}")
                    has_any_meaningful_content = True
                # If no packages remain, skip this line entirely
            else:
                # Not a pip install line - keep it and mark as meaningful if non-empty
                new_lines.append(line)
                if line.strip() and not line.strip().startswith('#'):
                    # Line is not empty and not a comment, so it's meaningful
                    has_any_meaningful_content = True
        
        filtered_script = '\n'.join(new_lines)
        
        # Skip only if there's no meaningful content at all
        skip = not has_any_meaningful_content
        
        return filtered_script, skip
    # New Function ---- From the LocalCoomandLineExecutor, filter out any potentially harmful terminal commands, that limit access
    # to certain features of the system. This is incorporated inside the Singularity Container for additional security.
    def sanitize_command(self,lang: str, code: str) -> None:
        """Sanitize the code block to prevent dangerous commands.
        This approach acknowledges that while Docker or similar
        containerization/sandboxing technologies provide a robust layer of security,
        not all users may have Docker installed or may choose not to use it.
        Therefore, having a baseline level of protection helps mitigate risks for users who,
        either out of choice or necessity, run code outside of a sandboxed environment.
        """
        dangerous_patterns = [
            (r"\brm\s+-rf\b", "Use of 'rm -rf' command is not allowed."),
            (r"\bmv\b.*?\s+/dev/null", "Moving files to /dev/null is not allowed."),
            (r"\bdd\b", "Use of 'dd' command is not allowed."),
            (r">\s*/dev/sd[a-z][1-9]?", "Overwriting disk blocks directly is not allowed."),
            (r":\(\)\{\s*:\|\:&\s*\};:", "Fork bombs are not allowed."),
            # Added Patterns for extra security:
            (r"\bcurl\b.*\|.*\bsh\b", "Downloading and executing code with curl | sh is not allowed."),
            (r"\bwget\b.*\|.*\bsh\b", "Downloading and executing code with wget | sh is not allowed."),
            (r"\bpython\s+-c\s+", "Running inline Python code is not allowed."),
            (r"\bsudo\b", "Use of 'sudo' command is not allowed."),
            (r"\bmkfs\.", "Filesystem creation (mkfs) is not allowed."),
            (r"\bchown\b", "Changing ownership with chown is not allowed."),
            (r"\bchmod\s+[^\n;]*777\b", "Making files globally writable with chmod 777 is not allowed."),
            ]
        if lang in ["bash", "shell", "sh"]:
            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    raise ValueError(f"Potentially dangerous command detected: {message}")

            # Only allow pip installs from PyPI (block URLs, VCS, file paths, other package managers)
        pip_install_pattern = r"pip\s+install\s+([^\n;]+)"
        for line in code.splitlines():
            pip_match = re.search(pip_install_pattern, line)
            if pip_match:
                install_args = pip_match.group(1)
                args = install_args.split()
                
                # Check for dangerous flags that could bypass restrictions
                dangerous_flags = ['--find-links', '--index-url', '--extra-index-url', 
                                  '--trusted-host', '--no-index', '--prefer-binary']
                for i, arg in enumerate(args):
                    if arg in dangerous_flags:
                        # Check the next argument (flag value)
                        if i + 1 < len(args):
                            next_arg = args[i + 1]
                            # Block file://, http://, https://, ftp:// URLs in flag values
                            if re.match(r"^(file|https?|ftp)://", next_arg):
                                raise ValueError(
                                    f"Non-PyPI pip install flag detected and blocked: '{arg} {next_arg}'"
                                )
                            # Block file paths in flag values
                            if next_arg.startswith("/") or next_arg.startswith("./") or next_arg.startswith("../"):
                                raise ValueError(
                                    f"Non-PyPI pip install flag with file path blocked: '{arg} {next_arg}'"
                                )
                
                # Check package arguments (skip flags)
                for arg in args:
                    # Accept flags/options for pip install
                    if arg.startswith('-'):
                        continue
                    
                    # Check for VCS protocols in the argument (handles package@git+https://...)
                    if re.search(r"@(git\+|svn\+|hg\+|bzr\+)", arg, re.IGNORECASE):
                        raise ValueError(
                            f"VCS-based pip install detected and blocked: '{arg}'"
                        )
                    
                    # Reject URLs (including file://)
                    if re.match(r"^(file|https?|ftp)://", arg):
                        raise ValueError(
                            f"URL-based pip install detected and blocked: '{arg}'"
                        )
                    
                    # Reject git+ at start (standalone git+ URL)
                    if arg.lower().startswith("git+"):
                        raise ValueError(
                            f"Git-based pip install detected and blocked: '{arg}'"
                        )
                    
                    # Reject file paths
                    #if arg.startswith("/") or arg.startswith("./") or arg.startswith("../"):
                       # raise ValueError(
                        #    f"File path pip install detected and blocked: '{arg}'"
                        #)
        
            # Block alternative package managers
            if re.search(r"\b(conda|apt-get|apt|yum|dnf|apk)\s+install\b", line):
                raise ValueError(
                    f"Non-PyPI package manager install blocked: '{line.strip()}'"
                )

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> CommandLineCodeResult:
        """(Experimental) Execute the code blocks and return the result. 

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CommandlineCodeResult: The result of the code execution. 
        """
        if len(code_blocks) == 0:
            raise ValueError("No code blocks to execute.")

        outputs = []
        files = []
        last_exit_code = 0

        for code_block in code_blocks:
            lang = self.LANGUAGE_ALIASES.get(code_block.language.lower(), code_block.language.lower())
            if lang not in self.DEFAULT_EXECUTION_POLICY:
                outputs.append(f"Unsupported language {lang}\n")
                last_exit_code = 1
                break

            execute_code = self.execution_policies.get(lang, False)
            #code = silence_pip(code_block.code, lang)
            code = code_block.code

            # Check if there is a filename comment
            try:
                filename = _get_file_name_from_content(code, self._work_dir)
            except ValueError:
                outputs.append("Filename is not in the workspace")
                last_exit_code = 1
                break

            if not filename:
                filename = f"tmp_code_{md5(code.encode()).hexdigest()}.{lang}"

            code_path = self._work_dir / filename
            with code_path.open("w", encoding="utf-8") as fout:
                fout.write(code)
            files.append(code_path)

            if not execute_code:
                outputs.append(f"Code saved to {code_path!s}\n")
                continue

            #---- New Functions Used: Intercept Executions:

            #1 Check code for any attempts at harmful commands.
            self.sanitize_command(lang, code)

            #2 Run pip/package check inside the container
            #print("Checking container packages before executing shell code...")
            package_list = self.get_pip_list()  # Write this method as shown above
            print("Container installed packages:\n", package_list)  # or other logic
            if lang in ["bash", "shell", "sh"]:
                #3 Filter out any packages that are already installed in the container.
                New_code,skip=self.pip_filter(code, package_list)
                if skip:
                    continue

                with code_path.open("w", encoding="utf-8") as fout:
                    fout.write(New_code)
                print(New_code)

            #6# Executes the command in an already running Singularity contianer instance. Uses the CLI rather
            # than Python SDK like Docker.
            
            command = [
                "singularity", "exec",
                f"instance://{self._container_name}",
                #"timeout", str(self._timeout),
                _cmd(lang), f"/workspace/{filename}",

            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )

            exit_code = result.returncode
            output = result.stdout + result.stderr
            #---------------------------------------------------------------------------------

            if exit_code == 124:
                output += "\n" + TIMEOUT_MSG
            outputs.append(output)

            last_exit_code = exit_code
            if exit_code != 0:
                break

        code_file = str(files[0]) if files else None
        return CommandLineCodeResult(exit_code=last_exit_code, output="".join(outputs), code_file=code_file)

    def restart(self) -> None:
        """(Experimental) Restart the code executor."""
        #7# Stops current instance. Adapted to Singularity.
        subprocess.run(
            ["singularity", "instance", "stop", self._container_name],
            capture_output=True,
            text=True
        )

        instance_start_cmd = [
            "singularity", "instance", "start",
            "--nv",  # Enable NVIDIA GPU support
            "--containall",
            "--no-home",
            "--bind", f"{pip_install_dir.resolve()}:/pip_install",  # Alternative to writable-tmpfs for pip installs. Needs to be writable to allow agents to install packages on the go.
            #"--writable-tmpfs", # Memory limits are possible, depending on sessiondir max size. Requires root to modify.
            "--bind", f"{bind_dir.resolve()}:/workspace",
            "--bind", f"{inputs_path.resolve()}:/inputs:ro", # Read only to prevent warping data.
            "--env", "PYTHONPATH=/pip_install",
            "--env", "TMPDIR=/pip_install",
            "--env", "PIP_CACHE_DIR=/pip_install",
            "--env", "PIP_TARGET=/pip_install",
            "--env", "HOME=/pip_install",  # This prevents user installation fallback
            str(self._image_path),
            self._instance_name
        ]

        result = subprocess.run(
            instance_start_cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise ValueError(f"Failed to restart Singularity instance: {result.stderr}")
        #---------------------------------------------------------------------------------

    def stop(self) -> None:
        """(Experimental) Stop the code executor."""
        self._cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

# Tests Completed:
# 1. Installed Docker Image with existing installs - successfully used in python code blocks.
#2. pip installations for packages, such as ffmpeg into writabl;e tmpfs, and used in python code.
#   Tested within the Anaconda Docker image.
#3. Tested the filter for the removal of existing python packages.

# Not Tested:
# 1. Not tested security features such as filters, but code in place in case, but unlikely to be required.
# 2. Not tested the robustness of the filters and if they eventually stop the AG2 agent from running.