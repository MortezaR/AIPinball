from setuptools import setup, find_packages

setup(
    name="pinball_gym_env",  # name of your package
    version="0.1.0",
    author="MortezaR",
    description="A custom OpenAI Gym environment that plays pinball using Visual Pinball",
    packages=find_packages(),  # finds Python packages automatically
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.22.0",
    ],
    entry_points={
        "gymnasium.envs": [
            "PinballEnv-v0 = pinball_env.pinball_env:PinballEnv",  # id = module:class
        ],
    },
)
