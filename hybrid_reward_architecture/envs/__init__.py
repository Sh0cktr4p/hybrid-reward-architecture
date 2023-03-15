from gym import register

from hybrid_reward_architecture.envs.hra_ant import HRAAnt  # noqa: F401
from hybrid_reward_architecture.envs.hra_humanoid import HRAHumanoid  # noqa: F401
from hybrid_reward_architecture.envs.hra_half_cheetah import HRAHalfCheetah  # noqa: F401

register(
    id="HRAAnt-v3",
    entry_point="hybrid_reward_architecture.envs.hra_ant:HRAAnt",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="HRAHumanoid-v3",
    entry_point="hybrid_reward_architecture.envs.hra_humanoid:HRAHumanoid",
    max_episode_steps=1000,
)

register(
    id="HRAHalfCheetah-v3",
    entry_point="hybrid_reward_architecture.envs.hra_half_cheetah:HRAHalfCheetah",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
