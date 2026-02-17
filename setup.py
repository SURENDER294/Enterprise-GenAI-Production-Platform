"""Package setup configuration for AI Engineer Portfolio Projects."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    # Filter out comment lines and empty lines
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ai-engineer-portfolio",
    version="1.0.0",
    author="Surender Reddy",
    author_email="surender29.methukupalli@gmail.com",
    description="End-to-end AI Engineering portfolio with LLM fine-tuning, RAG, Agentic AI, CV, and MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SURENDER294/AI-Engineer-Portfolio-Projects",
    project_urls={
        "Bug Tracker": "https://github.com/SURENDER294/AI-Engineer-Portfolio-Projects/issues",
        "Documentation": "https://github.com/SURENDER294/AI-Engineer-Portfolio-Projects#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"src": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.9.0",
            "pre-commit>=3.7.0",
        ],
        "gpu": [
            "torch>=2.2.1+cu121",
            "torchvision>=0.17.1+cu121",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-api=src.main:app",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
