import os

HYBRID_REWARD_ARCHITECTURE_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(HYBRID_REWARD_ARCHITECTURE_ROOT, os.pardir, "models")
RUNS_SAVE_PATH = os.path.join(HYBRID_REWARD_ARCHITECTURE_ROOT, os.pardir, "runs")
WANDB_PATH = os.path.join(HYBRID_REWARD_ARCHITECTURE_ROOT, os.pardir, "wandb")
CONFIG_PATH = os.path.join(HYBRID_REWARD_ARCHITECTURE_ROOT, os.pardir, "config")
