"""
MiRAGE - Setup Configuration

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="mirage-benchmark",
    version="1.0.2",
    author="MiRAGE Authors",
    author_email="contact@example.com",
    description="A Multiagent Framework for Generating Multimodal Multihop QA Datasets for RAG Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChandanKSahu/MiRAGE",
    
    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "transformers>=4.44.0",
        "huggingface_hub>=0.16.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "sentence-transformers>=2.2.0",
        "bertopic>=0.16.0",
        "umap-learn>=0.5.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "gpu": [
            "faiss-gpu>=1.7.0",
            "bitsandbytes>=0.43.0",
            "accelerate>=0.20.0",
        ],
        "pdf": [
            "docling>=0.1.0",
            "pypdfium2>=4.0.0",
        ],
        "eval": [
            "ragas>=0.1.0",
            "datasets>=2.0.0",
            "langchain-google-genai>=1.0.0",
            "langchain-openai>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "twine>=4.0.0",
            "build>=0.10.0",
        ],
        "all": [
            # GPU
            "faiss-gpu>=1.7.0",
            "bitsandbytes>=0.43.0",
            "accelerate>=0.20.0",
            # PDF
            "docling>=0.1.0",
            "pypdfium2>=4.0.0",
            # Eval
            "ragas>=0.1.0",
            "datasets>=2.0.0",
            "langchain-google-genai>=1.0.0",
            "langchain-openai>=0.1.0",
            # Dev
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
        ],
    },
    
    # Entry points for CLI
    entry_points={
        "console_scripts": [
            "mirage=mirage.cli:main",
            "mirage-preflight=mirage.utils.preflight:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="rag multimodal qa dataset generation llm vlm evaluation benchmark",
    
    # Include package data
    include_package_data=True,
)
