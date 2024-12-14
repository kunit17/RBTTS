import json
import os

# Get the current script directory
current_dir = os.path.dirname(__file__)

# Load vocabulary from JSON file
vocab_path = os.path.join(current_dir, 'vocab.json')
try:
    with open(vocab_path, 'r') as json_file:
        vocab = json.load(json_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find 'vocab.json' at {vocab_path}")
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON in 'vocab.json' at {vocab_path}")

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

