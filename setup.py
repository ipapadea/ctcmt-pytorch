"""Setup for the clean CTCMT CTTA package."""
from setuptools import find_packages, setup

setup(
    name="ctcmt",
    version="0.2.0",
    description=(
        "Clean-PyTorch CTTA layer for CTCMT experiments. Model, checkpoints, "
        "datasets and evaluators are reused verbatim from the detectron2 "
        "project at CTCMT/detectron2."
    ),
    author="CTCMT",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest"],
    },
)
