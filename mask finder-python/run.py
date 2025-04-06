#!/usr/bin/env python3
"""
Run script for the Face Mask Detection System
This script provides a command-line interface to run different components
of the Face Mask Detection System.
"""

import os
import argparse
import subprocess
import sys

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import tensorflow
        import opencv_python
        import numpy
        import pandas
        import sklearn
        import PIL
        import matplotlib
        return True
    except ImportError as e:
        print(f"Missing requirement: {e}")
        print("Please install all requirements with: pip install -r requirements.txt")
        return False

def create_directories():
    """Create required directories if they don't exist"""
    os.makedirs("data/with_mask", exist_ok=True)
    os.makedirs("data/without_mask", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("downloads", exist_ok=True)

def download_data():
    """Download sample dataset"""
    try:
        print("Downloading sample dataset...")
        subprocess.run([sys.executable, "download_sample_dataset.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error downloading dataset. Please check your internet connection.")
        return False
    return True

def train_model():
    """Train the face mask detection model"""
    print("Training the face mask detection model...")
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error training the model.")
        return False
    return True

def run_app():
    """Run the Streamlit web application"""
    print("Starting the Streamlit web application...")
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError:
        print("Error running the Streamlit application.")
        return False
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    return True

def main():
    """Main function to run the Face Mask Detection System"""
    parser = argparse.ArgumentParser(description="Face Mask Detection System")
    parser.add_argument("--download", action="store_true", help="Download sample dataset")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--run", action="store_true", help="Run the web application")
    parser.add_argument("--all", action="store_true", help="Download data, train model, and run app")

    args = parser.parse_args()

    # If no arguments provided, show help
    if not (args.download or args.train or args.run or args.all):
        parser.print_help()
        return

    # Check requirements
    if not check_requirements():
        return

    # Create directories
    create_directories()

    # Process arguments
    if args.all or args.download:
        if not download_data():
            return

    if args.all or args.train:
        if not train_model():
            return

    if args.all or args.run:
        run_app()

if __name__ == "__main__":
    main()
