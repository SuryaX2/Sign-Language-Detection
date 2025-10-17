"""
Setup script for Sign Language Detection project.
Handles package installation and configuration.
"""

from setuptools import setup, find_packages
import os


# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Sign Language Detection using Transformer Model"


# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirement.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="sign-language-detection",
    version="1.0.0",
    author="Surya",
    author_email="sekharsurya111@gmail.com",
    description="Real-time sign language detection using Transformer model and MediaPipe",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/suryax2/sign-language-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pylint>=2.12.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sign-preprocess=src.data_preprocessing:main",
            "sign-train=src.train_model:main",
            "sign-evaluate=src.evaluate_model:main",
            "sign-demo=src.live_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.py"],
    },
    zip_safe=False,
    keywords=[
        "sign-language",
        "detection",
        "transformer",
        "mediapipe",
        "computer-vision",
        "deep-learning",
        "machine-learning",
        "real-time",
        "gesture-recognition",
    ],
    project_urls={
        "Source": "https://github.com/suryax2/sign-language-detection",
    },
)
