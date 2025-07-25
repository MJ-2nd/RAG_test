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

def download_model(model_name: str, model_path: str):
    """Download model"""
    print(f"Downloading model: {model_name}")
    print(f"Save path: {model_path}")
    
    # Create directory
    os.makedirs(model_path, exist_ok=True)
    
    # Download model
    snapshot_download(
        repo_id=model_name,
        local_dir=model_path
    )
    
    print(f"Model download complete: {model_path}")

def estimate_model_size(model_name: str) -> str:
    """Estimate model size"""
    size_estimates = {
        # 🥇 1순위: DeepSeek R1 14B (최고 성능, 32GB 최적 활용)
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "~28GB (FP16), requires ~28GB VRAM 🥇 최고 성능, 코드 특화, Tool-calling 지원",
        
        # 🥈 2순위: Qwen2.5 32B AWQ (고성능, 양자화로 32GB 활용)
        "Qwen/Qwen2.5-32B-Instruct-AWQ": "~16GB (AWQ 4-bit), requires ~16GB VRAM 🥈 고성능, 다국어, 부분적 Tool-calling",
        
        # 🥉 3순위: DeepSeek R1 32B AWQ (코드 특화, 양자화)
        "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ": "~16GB (AWQ 4-bit), requires ~16GB VRAM 🥉 코드 특화, Tool-calling 지원",
        
        # 🏅 4순위: Qwen2.5 14B (다국어, 부분적 tool-calling)
        "Qwen/Qwen2.5-14B-Instruct": "~28GB (FP16), requires ~28GB VRAM 🏅 다국어, 부분적 Tool-calling",
        
        # 🏅 5순위: Llama 3.1 8B (안정적, 제한적 tool-calling)
        "meta-llama/Llama-3.1-8B-Instruct": "~16GB (FP16), requires ~16GB VRAM 🏅 안정적, 제한적 Tool-calling",
        
        # 🏅 6순위: Mistral 7B (MoE 아키텍처, 제한적 tool-calling)
        "mistralai/Mistral-7B-Instruct-v0.3": "~14GB (FP16), requires ~14GB VRAM 🏅 MoE, 제한적 Tool-calling",
        
        # 🏅 7순위: SmolLM3 3B (Tool-calling 특화, 효율적)
        "HuggingFaceTB/SmolLM3-3B-Instruct": "~6GB (FP16), requires ~6GB VRAM 🏅 Tool-calling 특화, 효율적",
        
        # 기타 모델들
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "~14GB (FP16), requires ~14GB VRAM ✅ Tool-calling 지원",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "~65GB (FP16), requires 64GB+ VRAM ❌ 32GB 부족",
        "Qwen/Qwen2.5-7B-Instruct": "~14GB (FP16), requires ~14GB VRAM ⚠️ 부분적 tool-calling",
        "Qwen/Qwen2.5-14B-Instruct-AWQ": "~6GB (AWQ 4-bit), requires ~6GB VRAM ⚠️ 부분적 tool-calling",
        "meta-llama/Llama-3.1-70B-Instruct": "~140GB (FP16), requires 분산 처리 ❌ 32GB 부족",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "~90GB (MoE), requires 64GB+ VRAM ❌ 32GB 부족",
        "moonshotai/Kimi-K2-Instruct": "~65-90GB (1T MoE), requires 64GB+ VRAM ❌ 32GB 부족",
        "moonshotai/Kimi-K2-Base": "~65-90GB (1T MoE), requires 64GB+ VRAM ❌ 32GB 부족",
        "Valdemardi/DeepSeek-R1-Distill-Qwen-14B-AWQ": "~8GB (AWQ 4-bit), requires ~8GB VRAM ✅ 32GB에 안전",
        "Valdemardi/DeepSeek-R1-Distill-Qwen-7B-AWQ": "~4GB (AWQ 4-bit), requires ~4GB VRAM ✅ 매우 안전",
        
        # Embedding models
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
    
    # Model type detection and memory prediction (성능 순)
    if "deepseek" in model_name_lower and "14b" in model_name_lower and "awq" not in model_name_lower:
        print("🥇 LLM (DeepSeek R1-14B): ~28GB total, ~14GB per GPU with tensor_parallel_size=2")
        print("✅ 32GB VRAM 최적 활용 (87.5% 사용)")
        print("🔧 코드 특화, Tool-calling 지원")
        print("🎯 32GB 환경에서 최고 성능 모델")
    elif "qwen" in model_name_lower and "32b" in model_name_lower and "awq" in model_name_lower:
        print("🥈 LLM (Qwen2.5-32B-AWQ): ~16GB total, ~8GB per GPU with tensor_parallel_size=2")
        print("✅ 32GB VRAM 효율적 활용 (50% 사용)")
        print("🌐 다국어 지원, 부분적 Tool-calling")
        print("🎯 양자화로 32B 성능을 16GB로 압축")
    elif "deepseek" in model_name_lower and "32b" in model_name_lower and "awq" in model_name_lower:
        print("🥉 LLM (DeepSeek R1-32B-AWQ): ~16GB total, ~8GB per GPU with tensor_parallel_size=2")
        print("✅ 32GB VRAM 효율적 활용 (50% 사용)")
        print("🔧 코드 특화, Tool-calling 지원")
        print("🎯 32B 성능을 양자화로 16GB로 압축")
    elif "qwen" in model_name_lower and "14b" in model_name_lower and "awq" not in model_name_lower:
        print("🏅 LLM (Qwen2.5-14B): ~28GB total, ~14GB per GPU with tensor_parallel_size=2")
        print("✅ 32GB VRAM 최적 활용 (87.5% 사용)")
        print("🌐 다국어 지원, 부분적 Tool-calling")
        print("🎯 다국어 성능에 특화")
    elif "llama" in model_name_lower and "8b" in model_name_lower:
        print("🏅 LLM (Llama-3.1-8B): ~16GB total, ~8GB per GPU with tensor_parallel_size=2")
        print("✅ 32GB VRAM 효율적 활용 (50% 사용)")
        print("🦙 Meta's 안정적 모델, 제한적 Tool-calling")
        print("🎯 안정성과 성능의 균형")
    elif "mistral" in model_name_lower and "7b" in model_name_lower and "mixtral" not in model_name_lower:
        print("🏅 LLM (Mistral-7B): ~14GB total, ~7GB per GPU with tensor_parallel_size=2")
        print("✅ 32GB VRAM 효율적 활용 (43.75% 사용)")
        print("🔧 MoE 아키텍처, 제한적 Tool-calling")
        print("🎯 MoE의 효율성과 성능")
    elif "smollm3" in model_name_lower:
        print("🏅 LLM (SmolLM3-3B): ~6GB total, single GPU 가능")
        print("✅ 32GB VRAM 여유로운 활용 (18.75% 사용)")
        print("🔧 Tool-calling 특화 파인튜닝")
        print("🎯 Tool-calling에 최적화된 효율적 모델")
    elif "deepseek" in model_name_lower:
        if "32b" in model_name_lower:
            print("LLM (DeepSeek R1-32B): ~65GB total, ~32GB per GPU")
            print("❌ 32GB VRAM 환경에는 부족")
        elif "7b" in model_name_lower:
            print("LLM (DeepSeek R1-7B): ~14GB total, single GPU 가능")
            print("✅ 32GB VRAM 환경에 안전")
        print("🔧 Code-specialized with tool-calling")
    elif "qwen" in model_name_lower:
        if "32b" in model_name_lower:
            if "awq" in model_name_lower:
                print("LLM (Qwen2.5-32B-AWQ): ~16GB total, ~8GB per GPU")
                print("✅ 32GB VRAM 환경에 적합 (AWQ 양자화)")
            else:
                print("LLM (Qwen2.5-32B): ~65GB total, ~32GB per GPU")
                print("❌ 32GB VRAM 환경에는 부족")
        elif "14b" in model_name_lower:
            if "awq" in model_name_lower:
                print("LLM (Qwen2.5-14B-AWQ): ~6GB total, single GPU 가능")
                print("✅ 32GB VRAM 환경에 최적 (AWQ 양자화)")
            else:
                print("LLM (Qwen2.5-14B): ~28GB total, ~14GB per GPU")
                print("✅ 32GB VRAM 환경에 적합")
        elif "7b" in model_name_lower:
            print("LLM (Qwen2.5-7B): ~14GB total, single GPU 가능")
            print("✅ 32GB VRAM 환경에 안전")
        print("🌐 Multilingual support, ⚠️ 부분적 tool-calling")
    elif "kimi" in model_name_lower or "moonshotai" in model_name_lower:
        print("LLM (Kimi K2): ~65-90GB total (1T MoE)")
        print("❌ 32GB VRAM 환경에는 부족 (64GB+ 필요)")
        print("💡 대안: DeepSeek R1-14B 사용 권장")
    elif "llama" in model_name_lower:
        if "70b" in model_name_lower:
            print("LLM (Llama-70B): ~140GB total, 분산 처리 필수")
            print("❌ 32GB 환경에서는 양자화 없이 불가능")
        print("🦙 Meta's open-source LLM, ⚠️ 제한적 tool-calling")
    elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
        if "mixtral" in model_name_lower:
            print("LLM (Mixtral MoE): ~90GB total, ~45GB per GPU")
            print("❌ 32GB 환경에서는 양자화 필요")
            print("🔧 MoE architecture")
        else:
            print("LLM (Mistral-7B): ~14GB total, single GPU 가능")
            print("✅ 32GB VRAM 환경에 안전")
        print("⚠️ 제한적 tool-calling")
    elif "awq" in model_name_lower:
        if "32b" in model_name_lower:
            print("LLM (32B-AWQ): ~16GB total, ~8GB per GPU with tensor_parallel_size=2")
            print("✅ 32GB VRAM 환경에 적합 (AWQ 양자화)")
        elif "14b" in model_name_lower:
            print("LLM (14B-AWQ): ~6-8GB total, single GPU 가능")
            print("✅ 32GB VRAM 환경에 적합 (AWQ 양자화)")
        elif "7b" in model_name_lower:
            print("LLM (7B-AWQ): ~3-4GB total, single GPU 가능")
            print("✅ 매우 여유있는 메모리 사용")
    elif "32b" in model_name_lower:
        if quantization_config.get('enabled', False):
            method = quantization_config.get('method', 'unknown')
            bits = quantization_config.get('bits', 16)
            
            if method == 'bitsandbytes' and bits == 4:
                print("LLM (32B + 4bit BitsAndBytes): ~28GB total, ~14GB per GPU")
                print("✅ 32GB VRAM 환경에 적합 (양자화)")
            elif method == 'fp8' or bits == 8:
                print("LLM (32B + 8bit FP8): ~32GB total, ~16GB per GPU")
                print("⚠️  32GB VRAM 환경에서 한계")
            else:
                print("LLM (32B + quantization): 메모리 사용량 불명확")
        else:
            print("LLM (32B): ~65GB total, ~32GB per GPU")
            print("❌ 32GB VRAM 환경에는 부족")
    elif "14b" in model_name_lower:
        print("LLM (14B): ~28GB total, ~14GB per GPU with tensor_parallel_size=2")
        print("✅ 32GB VRAM 환경에 적합")
    elif "7b" in model_name_lower:
        print("LLM (7B): ~14GB total, single GPU 가능")
        print("✅ 32GB VRAM 환경에 안전")
    elif "3b" in model_name_lower:
        print("LLM (3B): ~6GB total, single GPU 가능")
        print("✅ 32GB VRAM 환경에 매우 안전")
    elif "1.5b" in model_name_lower:
        print("LLM (1.5B): ~3GB total")
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
            download_model(llm_config['model_name'], llm_config['model_path'])
        
        # Download embedding model
        if not args.llm_only:
            download_model(embedding_config['model_name'], embedding_config['model_path'])
        
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