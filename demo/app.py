#!/usr/bin/env python3
"""
MiRAGE Demo - Hugging Face Spaces Entry Point

This is the main entry point for Hugging Face Spaces deployment.
It imports and launches the Gradio app.

For Hugging Face Spaces, this file should be named 'app.py'.
"""

from gradio_app import create_demo

# Create and launch the demo
demo = create_demo()

# Launch with Spaces-compatible settings
if __name__ == "__main__":
    demo.launch()
