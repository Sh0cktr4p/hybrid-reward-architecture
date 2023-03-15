from setuptools import setup, find_packages

setup(
    name="hybrid_reward_architecture",
    version="0.0.3",
    packages=[package for package in find_packages() if package.startswith("hybrid_reward_architecture")],
)
