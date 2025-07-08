#!/usr/bin/env python3
"""
RAG system setup and execution script
"""

import os
import subprocess
import sys
import argparse

def check_dependencies():
    """Check required dependencies"""
    try:
        import torch
        import faiss
        import sentence_transformers
        print("✓ All dependencies are installed.")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"✓ Number of GPUs: {gpu_count}")
            
            if gpu_memory < 30:
                print("⚠️  Warning: GPU memory is less than 30GB. 32B model may be difficult to run.")
                print("   Recommend using 14B or 7B model.")
            
            return True
        else:
            print("✗ GPU not available.")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    dirs = ['models', 'documents', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Directory created: {dir_name}/")

def install_dependencies():
    """Install dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def run_download_models():
    """Run model download"""
    print("Starting model download...")
    try:
        subprocess.run([sys.executable, 'download_models.py'], check=True)
        print("✓ Model download complete")
        return True
    except subprocess.CalledProcessError:
        print("✗ Model download failed")
        return False

def run_build_index():
    """Run index building"""
    print("Starting index building...")
    try:
        subprocess.run([sys.executable, '-m', 'rag.build_index'], check=True)
        print("✓ Index building complete")
        return True
    except subprocess.CalledProcessError:
        print("✗ Index building failed")
        return False

def run_interactive():
    """Run interactive mode"""
    print("Starting RAG system interactive mode...")
    try:
        subprocess.run([sys.executable, '-m', 'query.query_rag', '--interactive'], check=True)
    except subprocess.CalledProcessError:
        print("✗ RAG system execution failed")
    except KeyboardInterrupt:
        print("\nProgram terminated.")

def main():
    parser = argparse.ArgumentParser(description="RAG system setup and execution")
    parser.add_argument("--check", action="store_true", help="Check system only")
    parser.add_argument("--setup", action="store_true", help="Perform initial setup")
    parser.add_argument("--download", action="store_true", help="Download models")
    parser.add_argument("--build-index", action="store_true", help="Build index")
    parser.add_argument("--run", action="store_true", help="Run interactive mode")
    parser.add_argument("--all", action="store_true", help="Perform all steps")
    
    args = parser.parse_args()
    
    print("=== RAG System Setup and Execution ===")
    print()
    
    # Check system
    print("1. System Environment Check")
    deps_ok = check_dependencies()
    gpu_ok = check_gpu()
    
    if args.check:
        return
    
    if not deps_ok and not args.setup:
        print("Please install dependencies first: python setup.py --setup")
        return
    
    # Initial setup
    if args.setup or args.all:
        print("\n2. Initial Setup")
        setup_directories()
        if not deps_ok:
            install_dependencies()
    
    # Model download
    if args.download or args.all:
        print("\n3. Model Download")
        if not run_download_models():
            return
    
    # Index building
    if args.build_index or args.all:
        print("\n4. Index Building")
        if not os.path.exists('documents') or not os.listdir('documents'):
            print("✗ No documents found in documents/ directory.")
            print("  Please place documents in documents/ directory and try again.")
            return
        
        if not run_build_index():
            return
    
    # Run interactive mode
    if args.run or args.all:
        print("\n5. RAG System Execution")
        run_interactive()
    
    if not any([args.check, args.setup, args.download, args.build_index, args.run, args.all]):
        print("Usage:")
        print("  python setup.py --check        # Check system")
        print("  python setup.py --setup        # Initial setup")
        print("  python setup.py --download     # Download models")
        print("  python setup.py --build-index  # Build index")
        print("  python setup.py --run          # Run interactive mode")
        print("  python setup.py --all          # Perform all steps")

if __name__ == "__main__":
    main() 