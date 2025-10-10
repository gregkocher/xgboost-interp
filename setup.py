"""
Setup script for xgboost-interp package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="xgboost-interp",
    version="0.1.0",
    author="Greg Kocher",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for interpreting and analyzing XGBoost models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xgboost-interp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "interactive": [
            "plotly>=5.0",
            "networkx>=2.5",
        ],
        "ale": [
            "pyALE>=0.2",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="xgboost, interpretability, machine learning, feature importance, partial dependence",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/xgboost-interp/issues",
        "Source": "https://github.com/yourusername/xgboost-interp",
        "Documentation": "https://xgboost-interp.readthedocs.io/",
    },
)
