"""
Common configuration for MLX LoRA demo.
"""

# Model configuration
BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
DEFAULT_ADAPTER = "adapters/nhl-stanley-cups-demo"
BEST_ADAPTER = "adapters/stanley-cup-best-2500"

# Training parameters
TRAINING_PARAMS = {
    "rank": 16,
    "dropout": 0.1,
    "scale": 20.0,
    "iterations": 2500,
    "batch_size": 2,
    "learning_rate": 1.5e-5
}

# Generation defaults
DEFAULT_MAX_TOKENS = 50
DEFAULT_TEMPERATURE = 0.1

# Data paths
TRAIN_DATA_PATH = "data/nhl_stanley_cups"