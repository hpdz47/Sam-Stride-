"""
Configuration package - automatically loads .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find .env file in project root (parent of Config/)
PROJECT_ROOT = Path(__file__).parent.parent
env_file = PROJECT_ROOT / ".env"

# Load .env file
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded .env from: {env_file}")
else:
    print(f"⚠️ .env file not found at: {env_file}")