#!/usr/bin/env python3
"""
Model download script for RAG system
"""

import os
import argparse
from huggingface_hub import snapshot_download
import yaml

def load_config(config_path: str = "llm/config.yaml"):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_llm_model(model_name: str, model_path: str):
    """Download LLM model"""
    print(f"Downloading LLM model: {model_name}")
    print(f"Save path: {model_path}")
    
    # Create directory
    os.makedirs(model_path, exist_ok=True)
    
    # Download model
    snapshot_download(
        repo_id=model_name,
        local_dir=model_path
    )
    
    print(f"LLM model download complete: {model_path}")

def download_embedding_model(model_name: str, model_path: str):
    """Download embedding model"""
    print(f"Downloading embedding model: {model_name}")
    print(f"Save path: {model_path}")
    
    # Create directory
    os.makedirs(model_path, exist_ok=True)
    
    # Download model
    snapshot_download(
        repo_id=model_name,
        local_dir=model_path
    )
    
    print(f"Embedding model download complete: {model_path}")

def estimate_model_size(model_name: str) -> str:
    """Estimate model size"""
    size_estimates = {
        # DeepSeek-R1 models
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "~65GB (FP16), requires ~64GB VRAM",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "~28GB (FP16), requires ~28GB VRAM",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "~14GB (FP16), requires ~14GB VRAM",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "~3GB (FP16), requires ~3GB VRAM",
        
        # AWQ quantized versions
        "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ": "~6GB (AWQ 4-bit), requires ~6GB VRAM",
        "Valdemardi/DeepSeek-R1-Distill-Qwen-14B-AWQ": "~3.5GB (AWQ 4-bit), requires ~3.5GB VRAM",
        "Valdemardi/DeepSeek-R1-Distill-Qwen-7B-AWQ": "~1.8GB (AWQ 4-bit), requires ~1.8GB VRAM",
        
        # Other models
        "Qwen/Qwen2.5-32B-Instruct": "~65GB (FP16), ~20GB (4bit)",
        "Qwen/Qwen2.5-32B-Instruct-AWQ": "~6GB (AWQ 4-bit quantized)",
        "Qwen/Qwen2.5-14B-Instruct": "~28GB (FP16), ~8GB (4bit)",
        "Qwen/Qwen2.5-7B-Instruct": "~14GB (FP16), ~4GB (4bit)",
        "BAAI/bge-m3": "~2.3GB",
        "BAAI/bge-large-en-v1.5": "~1.3GB",
        "sentence-transformers/all-MiniLM-L6-v2": "~90MB"
    }
    
    return size_estimates.get(model_name, "Size information not available")

def main():
    parser = argparse.ArgumentParser(description="Download models for RAG system")
    parser.add_argument("--config", default="llm/config.yaml", help="Configuration file path")
    parser.add_argument("--llm-only", action="store_true", help="Download LLM model only")
    parser.add_argument("--embedding-only", action="store_true", help="Download embedding model only")
    parser.add_argument("--dry-run", action="store_true", help="Show information only without actual download")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    llm_config = config['llm']
    embedding_config = config['embedding']
    
    print("=== RAG System Model Download ===")
    print(f"Configuration file: {args.config}")
    print()
    
    # Print model information
    print("Models to download:")
    if not args.embedding_only:
        llm_size = estimate_model_size(llm_config['model_name'])
        print(f"  LLM: {llm_config['model_name']}")
        print(f"       Path: {llm_config['model_path']}")
        print(f"       Estimated size: {llm_size}")
    
    if not args.llm_only:
        embedding_size = estimate_model_size(embedding_config['model_name'])
        print(f"  Embedding: {embedding_config['model_name']}")
        print(f"            Path: {embedding_config['model_path']}")
        print(f"            Estimated size: {embedding_size}")
    
    print()
    
    # VRAM usage prediction
    print("=== Memory Usage Prediction ===")
    model_name_lower = llm_config['model_name'].lower()
    quantization_config = llm_config.get('quantization', {})
    
    if "awq" in model_name_lower:
        if "32b" in model_name_lower:
            print("LLM (DeepSeek-R1-32B-AWQ): ~6GB total, ~3GB per GPU with tensor_parallel_size=2")
            print("✅ 16GB GPU x2 환경에 적합 (하지만 메모리 활용도 낮음)")
        elif "14b" in model_name_lower:
            print("LLM (DeepSeek-R1-14B-AWQ): ~3.5GB total, single GPU 가능")
            print("✅ 16GB GPU 단일 사용으로 충분")
        elif "7b" in model_name_lower:
            print("LLM (DeepSeek-R1-7B-AWQ): ~1.8GB total, single GPU 가능")
            print("✅ 매우 여유있는 메모리 사용")
    elif "32b" in model_name_lower:
        if quantization_config.get('enabled', False):
            method = quantization_config.get('method', 'unknown')
            bits = quantization_config.get('bits', 16)
            
            if method == 'bitsandbytes' and bits == 4:
                print("LLM (DeepSeek-R1-32B + 4bit BitsAndBytes): ~28GB total, ~14GB per GPU")
                print("✅ 16GB GPU x2 환경에 최적 (최고 성능, 28GB 활용)")
            elif method == 'fp8' or bits == 8:
                print("LLM (DeepSeek-R1-32B + 8bit FP8): ~32GB total, ~16GB per GPU")
                print("⚠️  16GB GPU x2 환경에서 한계 (최대 성능, 32GB 활용)")
            else:
                print("LLM (DeepSeek-R1-32B + quantization): 메모리 사용량 불명확")
        else:
            print("LLM (DeepSeek-R1-32B): ~64GB total, ~32GB per GPU")
            print("❌ 16GB GPU x2 환경에는 메모리 부족")
    elif "14b" in model_name_lower:
        print("LLM (DeepSeek-R1-14B): ~28GB total, ~14GB per GPU with tensor_parallel_size=2")
        print("✅ 16GB GPU x2 환경에 최적 (안정적, 28GB 활용)")
    elif "7b" in model_name_lower:
        print("LLM (DeepSeek-R1-7B): ~14GB total, single GPU 가능")
        print("✅ 16GB GPU 단일 사용으로 충분")
    elif "1.5b" in model_name_lower:
        print("LLM (DeepSeek-R1-1.5B): ~3GB total")
        print("✅ CPU 실행도 가능")
    
    print("Embedding model: CPU usage (RAM ~2-4GB)")
    print("FAISS index: CPU usage (RAM ~hundreds of MB)")
    print()
    
    if args.dry_run:
        print("Dry run mode: No actual download will be performed.")
        return
    
    # User confirmation
    response = input("Do you want to start the download? (y/N): ").lower().strip()
    if response != 'y':
        print("Download cancelled.")
        return
    
    try:
        # Download LLM model
        if not args.embedding_only:
            download_llm_model(llm_config['model_name'], llm_config['model_path'])
        
        # Download embedding model
        if not args.llm_only:
            download_embedding_model(embedding_config['model_name'], embedding_config['model_path'])
        
        print("\n=== All Model Downloads Complete ===")
        print("Next steps:")
        print("1. Place documents in documents/ directory")
        print("2. Build index: python -m rag.build_index")
        print("3. Run RAG system: python -m query.query_rag --interactive")
        
    except Exception as e:
        print(f"Download error: {e}")
        print("Please check your network connection and try again.")

if __name__ == "__main__":
    main() 