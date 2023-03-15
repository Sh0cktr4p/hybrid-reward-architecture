import os

hybrid_reward_architecture_root = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(hybrid_reward_architecture_root, os.pardir, "models")
runs_save_path = os.path.join(hybrid_reward_architecture_root, os.pardir, "runs")
wandb_path = os.path.join(hybrid_reward_architecture_root, os.pardir, "wandb")
