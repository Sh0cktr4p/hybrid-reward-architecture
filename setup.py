from setuptools import setup, find_packages

setup(
    name="hybrid_reward_architecture",
    version="0.0.3",
    packages=[package for package in find_packages() if package.startswith("hybrid_reward_architecture")],
    install_requires=[
        "gym[all]==0.21.0",
        "stable-baselines3==1.7.0",
    ]
)
