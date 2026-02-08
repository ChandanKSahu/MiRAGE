"""
MiRAGE - Setup Configuration

Install with: pip install -e .
Or from PyPI: pip install mirage-benchmark
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="mirage-benchmark",
    version="1.2.7",
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
    
    # Core dependencies - everything needed for `runmirage` to work out of the box
    install_requires=[
        # === Core ML/DL ===
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.10.0",
        "Pillow>=10.0.0",

        # === Transformers & Embeddings ===
        "transformers>=4.44.0",
        "huggingface_hub>=0.23.2,<1.0",
        "sentence-transformers>=2.2.0",
        "safetensors>=0.4.0",
        "tokenizers>=0.20.0",

        # === Quantization & Acceleration ===
        "bitsandbytes>=0.43.0",
        "accelerate>=0.20.0",
        "peft>=0.11.0",

        # === Multimodal Embeddings ===
        "colpali-engine>=0.3.0",

        # === PDF & Document Processing ===
        "docling>=2.0.0",
        "docling-core>=2.0.0",
        "pypdfium2>=4.0.0",
        "easyocr>=1.7.0",
        "opencv-python-headless>=4.8.0",

        # === Vector Search & Retrieval ===
        "faiss-cpu>=1.7.0",

        # === Topic Modeling & Visualization ===
        "bertopic>=0.16.0",
        "umap-learn>=0.5.0",
        "datamapplot>=0.4.0",
        "plotly>=5.0.0",
        "matplotlib>=3.5.0",

        # === Data & Config ===
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",

        # === Networking ===
        "requests>=2.28.0",
        "aiohttp>=3.8.0",

        # === Gemini API ===
        "google-generativeai>=0.5.0",
    ],
    
    # Optional dependencies
    extras_require={
        "eval": [
            "ragas>=0.1.0",
            "datasets>=2.0.0",
            "langchain>=0.1.0",
            "langchain-core>=0.1.0",
            "langchain-community>=0.0.10",
            "langchain-google-genai>=1.0.0",
            "langchain-openai>=0.1.0",
            "tiktoken>=0.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "twine>=4.0.0",
            "build>=0.10.0",
        ],
        "all": [
            # Eval
            "ragas>=0.1.0",
            "datasets>=2.0.0",
            "langchain>=0.1.0",
            "langchain-core>=0.1.0",
            "langchain-community>=0.0.10",
            "langchain-google-genai>=1.0.0",
            "langchain-openai>=0.1.0",
            "tiktoken>=0.5.0",
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
            "run_mirage=mirage.run_mirage:main",
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
