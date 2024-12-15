import json
import os

# Get the current script directory
current_dir = os.path.dirname(__file__)

# Load characters from JSON file
char_path = os.path.join(current_dir, 'chars.json')
try:
    with open(char_path, 'r') as json_file:
        chars = json.load(json_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find 'vocab.json' at {char_path}")
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON in 'vocab.json' at {char_path}")

# Load parameters from JSON file
config_path = os.path.join(current_dir, 'config.json')
try:
    with open(config_path, 'r') as json_file:
        CONFIG = json.load(json_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find 'config.json' at {config_path}")
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON in 'config.json' at {config_path}")

# Access data paths safely
DATA_PATHS = CONFIG.get('data_paths', {})

# Debug print

