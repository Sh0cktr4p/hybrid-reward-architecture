from setuptools import setup, find_packages

setup(
    name="hybrid_reward_architecture",
    version="0.0.1",
    packages=[package for package in find_packages() if package.startswith("human_robot_gym")],
)
