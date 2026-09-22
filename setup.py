"""T2Image Phase-A research kernel (editable install)."""

from setuptools import setup, find_packages

setup(
    name="t2image",
    version="0.1.0-phase-a",
    description="T2Image Phase-A: text-to-image generation research kernel",
    author="t2image-team",
    packages=find_packages(where=".", exclude=["tests*", "docs*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    extras_require={
        "dev": ["pytest", "black", "isort", "ruff"],
        "eval": ["clean-fid==0.1.35", "torch-fidelity==0.3.0",
                 "open-clip-torch==2.26.1", "hpsv2==1.2.0", "pickscore==0.2.1"],
        "data": ["img2dataset==1.45.0"],
    },
)
