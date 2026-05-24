import subprocess
import sys
import os
import time
import requests
import threading
import atexit
import signal
from typing import Optional, Dict
from dataclasses import dataclass
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/app/.env")

logger = logging.getLogger(__name__)

class LLM_Manager():
    _instance = None
    _lock = threading.Lock() # Prevents multiple instances of the class from being created. This instance can remember what servers are set up.
    def __new__(cls,*args,**kwargs):
        if cls._instance is None:
            with cls._lock:
                cls._instance = super(LLM_Manager,cls).__new__(cls) # Note: When constructor has run cls._instance._init is the same as saying self.init.
                # super() calls parent, which os object, which allocates memory and creates an instance of LLM_Manager, but constructor function does NOT
                # run just yet.
                cls._instance._init= False # Constructor has NOT been run yet in above line.
        return cls._instance
    def __init__(self,LLM_Type: str):
        if self._init:

            self.LLM_Type= LLM_Type
            self.host= "127.0.0.1" # For a local server.
            self.api_key= os.getenv("API_KEY") # For the API key.
            if self.LLM_Type=="Reasoning":
                self.model= os.getenv("LLM_Model_Reasoning")
            elif self.LLM_Type=="Coding":
                self.model= os.getenv("LLM_Model_Coding")
            elif self.LLM_Type=="VL":
                self.model= os.getenv("LLM_Model_Visual")
            elif self.LLM_Type=="Judge":
                self.model = os.getenv("LLM_Model_Judge")
            else:
                raise ValueError(f"Invalid LLM Type: {self.LLM_Type}")

            return
        self.process= None
        self.started_at= None
        self._init= True # This says that the instance has been initialised now.
        self.port=8002 # Default port for a local server.
        self.LLM_Type= LLM_Type
        self.host= "127.0.0.1" # For a local server.
        self.api_key= os.getenv("API_KEY") # For the API key.
        # File Path For Logging:
        self.parent_dir = Path("/workspace")
        self.logging_dir = self.parent_dir / "logs"
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        #----
        if self.LLM_Type=="Reasoning":
            self.model= os.getenv("LLM_Model_Reasoning")
        elif self.LLM_Type=="Coding":
            self.model= os.getenv("LLM_Model_Coding")
        elif self.LLM_Type=="VL":
            self.model= os.getenv("LLM_Model_Visual")
        elif self.LLM_Type=="Judge":
                self.model = os.getenv("LLM_Model_Judge")
        else:
            raise ValueError(f"Invalid LLM Type: {self.LLM_Type}")
    
    def start_server(self):
        if self.LLM_Type=="Reasoning":
            cmd = ["vllm", "serve", self.model,
            "-dp", "1",
            "--dtype", "bfloat16", # Use for Ampere GPU or higher.
            #"--quantization", "bitsandbytes", # Use for Turing GPU ONLY as bfloat16 not compatble with Turing. (4-bit quantisation significantly increases compute time and reduces model effectiveness).
            #"--enable-expert-parallel",
            #"--language-model-only",
            "--reasoning-parser", "qwen3",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--generation-config", "vllm",
            "--host", self.host,
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--gpu-memory-utilization", "0.80",
            "--max-num-seqs", "2",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "16384",
            "--max-model-len", "90k",
            #"--chat-template", "/app/Config/tool_chat_template_gemma4.jinja"
            ]
        elif self.LLM_Type=="Coding":
            cmd = ["vllm", "serve", self.model,
            "--quantization", "bitsandbytes",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--generation-config", "vllm",
            "--host", self.host,
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--max-num-seqs", "4",
            "--max-model-len", "24000",
            "--gpu-memory-utilization", "0.85",]
        elif self.LLM_Type=="VL":
            cmd = ["vllm", "serve", self.model,
            "--quantization", "bitsandbytes",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            #"--chat-template", "vllm/examples/tool_chat_template_hermes.jinja",
            "--generation-config", "vllm",
            "--host", self.host,
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--gpu-memory-utilization", "0.8",
            "--max-num-seqs", "4",
            "--max-model-len", "20000"
            ]
        elif self.LLM_Type=="Judge":
            cmd = ["vllm", "serve", self.model,
            #"-dp", "1",
            "--dtype", "bfloat16", # Use for Ampere GPU or higher.
            #"--quantization", "bitsandbytes", # Use for Turing GPU ONLY as bfloat16 not compatble with Turing. (4-bit quantisation significantly increases compute time and reduces model effectiveness).
            "--tensor-parallel-size", "2",
            #"--enable-expert-parallel",
            "--language-model-only",
            "--reasoning-parser", "qwen3",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--generation-config", "vllm",
            "--host", self.host,
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "1",
            "--max-model-len", "150k" # use 30k for bgger GPU
            ]
        print(f"Starting {self.LLM_Type} server on {self.host}:{self.port}...")
        # Start process
        log_file = open(f"{self.logging_dir}/vllm_server_{self.LLM_Type.lower()}.log", "w")
        self.process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
        self.started_at = time.time()
        
        # Wait for server to be ready
        logger.info("Waiting for server to be ready")
        start_time = time.time()
        wait_timeout = 5000
        
        while time.time() - start_time < wait_timeout:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(" ✓ Ready!")
                    return True
            except (requests.ConnectionError, requests.Timeout, requests.RequestException):
                pass
                
            logger.debug(".",extra={"end":"", "flush":True})
            time.sleep(1)
            
        logger.error(" ✗ Failed to start!")
        return False
    def stop_server(self):
        if self.process:
            logger.info("Stopping server....")
            self.process.terminate()  # Sends shutdown signal
            self.process.wait() # Waits for it to finish
            logger.info("Server stopped")
            self.process = None
            if hasattr(self, "Current_Usage"):
                del self.Current_Usage
            return True
    def Manage_VLLM(self):

        if self.process and hasattr(self, 'Current_Usage'): # If Current Usage doesn't exist yet, the block is skipped, but variable created here.
        # If a process is not running then this block gets skipped.
            if self.Current_Usage != self.LLM_Type:
                logger.info("Switching LLM")
                self.stop_server()
            else:
                logger.info("Correct LLM is already running")
                return True # Exits the function f correct LLM is running already.
        #====================================Test
        out = ""
        try:
            out = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory,process_name","--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        if out:
            print("GPU processes detected:\n" + out)
        #========================================
        if self.start_server():
            self.Current_Usage = self.LLM_Type
            logger.info(f"Server started with model: {self.model}")
            return True
        else:
            logger.error("Failed to start server")
            return False # Exits the function if the server fails to start.


            
            
            


