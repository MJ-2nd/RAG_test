#!/usr/bin/env python3
"""
RAG system setup and execution script with universal LLM support
"""

import os
import subprocess
import sys
import argparse


def check_system():
    """Check system requirements for various LLM models"""
    print("=== System Requirements Check ===")
    
    # Python version check
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 9):
        print("✗ Python 3.9+ required for modern LLM models")
        return False
    print("✓ Python version OK")
    
    # GPU check
    gpu_ok = check_gpu()
    
    # Package check
    required_packages = [
        'torch', 'transformers', 'fastapi', 'uvicorn', 
        'huggingface_hub', 'pydantic', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return gpu_ok

def detect_model_requirements():
    """Detect model type and requirements from config"""
    try:
        import yaml
        with open('llm/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_name = config['llm']['model_name'].lower()
        
        # Model type detection
        if "kimi" in model_name or "moonshotai" in model_name:
            return "kimi", "MoE architecture with tool-calling", "~16GB"
        elif "qwen" in model_name:
            return "qwen", "High-performance Chinese LLM", "varies"
        elif "deepseek" in model_name:
            return "deepseek", "Code-specialized LLM", "varies"
        elif "smollm" in model_name:
            return "smollm", "Efficient small LLM with tool-calling", "~6GB"
        elif "llama" in model_name:
            return "llama", "Meta's open LLM", "varies"
        elif "mistral" in model_name or "mixtral" in model_name:
            return "mistral", "MoE architecture", "varies"
        else:
            return "generic", "Generic LLM model", "unknown"
            
    except Exception:
        return "unknown", "Could not detect model type", "unknown"

def check_gpu():
    """Check GPU availability for various LLM models"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            total_memory = 0
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                total_memory += gpu_memory
                print(f"✓ GPU {i}: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            
            print(f"✓ Total VRAM: {total_memory:.1f}GB across {gpu_count} GPU(s)")
            
            # Model requirements check
            model_type, model_desc, memory_req = detect_model_requirements()
            print(f"✓ Detected model type: {model_type} ({model_desc})")
            
            # Memory recommendations based on total VRAM
            if total_memory >= 30:
                print("✅ Excellent! Supports large models (32B+, MoE models)")
                if gpu_count >= 2:
                    print("✅ Dual GPU detected - Optimal for tensor parallelism")
                return True
            elif total_memory >= 16:
                print("✅ Good! Supports medium-large models (up to 32B)")
                print("   Recommended: Kimi K2, SmolLM3, Qwen2.5-14B")
                return True
            elif total_memory >= 8:
                print("⚠️  Moderate VRAM. Suitable for smaller models")
                print("   Recommended: SmolLM3-3B, Qwen2.5-7B, quantized models")
                return True
            elif total_memory >= 4:
                print("⚠️  Limited VRAM. Use small models or heavy quantization")
                print("   Consider CPU inference for larger models")
                return True
            else:
                print("⚠️  Very limited VRAM. CPU inference recommended")
                return True
        else:
            print("✗ GPU not available. CPU inference only")
            print("⚠️  LLM performance will be significantly slower on CPU")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def setup_directories():
    """Create necessary directories for LLM system"""
    dirs = ['models', 'documents', 'tools', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Directory created: {dir_name}/")
    
    # Create sample documents if none exist
    if not os.path.exists('documents') or not os.listdir('documents'):
        model_type, _, _ = detect_model_requirements()
        sample_content = f"""# RAG System with Tool-calling - Sample Document

## 현재 사용 중인 모델
이 시스템은 현재 **{model_type}** 타입의 모델을 사용하고 있습니다.

## Tool-calling 기능
다음과 같은 도구들을 자동으로 실행할 수 있습니다:
- 수학 계산 (calculate)
- 문서 검색 (search_documents)
- 파일 탐색 (search_files)
- 시간 조회 (get_current_time)
- 사용자 정의 도구 (MCP)

## 지원되는 모델들
- **Kimi K2**: MoE 아키텍처, Native tool-calling
- **Qwen 시리즈**: 고성능 중국어/영어 LLM
- **DeepSeek**: 코드 특화 LLM
- **SmolLM3**: 효율적인 소형 LLM
- **Llama**: Meta의 오픈소스 LLM
- **Mistral/Mixtral**: MoE 아키텍처

이 샘플 문서는 RAG 시스템의 동작을 테스트하기 위한 것입니다.
"""
        with open('documents/sample_rag_system.txt', 'w', encoding='utf-8') as f:
            f.write(sample_content)
        print("✓ Sample document created: documents/sample_rag_system.txt")

def run_download_models():
    """Run model download"""
    model_type, model_desc, memory_req = detect_model_requirements()
    print(f"Starting model download...")
    print(f"📦 Model type: {model_type} ({model_desc})")
    print(f"💾 Estimated memory: {memory_req}")
    print("⚠️  Large models may take significant time to download.")
    try:
        subprocess.run([sys.executable, 'download_models.py'], check=True)
        print(f"✓ {model_type} model download complete")
        return True
    except subprocess.CalledProcessError:
        print("✗ Model download failed")
        print("💡 Try: python download_models.py --dry-run to check requirements")
        return False

def run_build_index():
    """Run index building"""
    print("Starting RAG index building...")
    try:
        subprocess.run([sys.executable, '-m', 'rag.build_index'], check=True)
        print("✓ RAG index building complete")
        return True
    except subprocess.CalledProcessError:
        print("✗ Index building failed")
        return False

def run_server():
    """Run FastAPI server with tool-calling"""
    model_type, model_desc, _ = detect_model_requirements()
    print(f"Starting LLM server with tool-calling...")
    print(f"🤖 Model type: {model_type}")
    print(f"📝 Description: {model_desc}")
    print("🔧 Tool-calling features:")
    print("   - Document search via RAG")
    print("   - Mathematical calculations") 
    print("   - File system exploration")
    print("   - Current time queries")
    print("   - Custom MCP tools")
    print()
    print("🌐 Access the API at: http://localhost:8000/docs")
    print("💬 Interactive chat available at: http://localhost:8000")
    print()
    try:
        subprocess.run([sys.executable, '-m', 'llm.app'], check=True)
    except subprocess.CalledProcessError:
        print("✗ LLM server failed to start")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")

def run_interactive():
    """Run interactive mode"""
    print("Starting RAG system interactive mode...")
    print("💬 Available commands:")
    print("   - Ask questions about your documents")
    print("   - Use /stats to see index statistics")
    print("   - Use /context on/off to toggle context")
    print("   - Type 'quit' to exit")
    print()
    try:
        subprocess.run([sys.executable, '-m', 'query.query_rag', '--interactive'], check=True)
    except subprocess.CalledProcessError:
        print("✗ RAG system execution failed")
    except KeyboardInterrupt:
        print("\n🛑 Program terminated.")

def test_tool_calling():
    """Test tool-calling functionality"""
    print("Testing LLM tool-calling functionality...")
    
    # Check if tools directory exists and has tools
    if os.path.exists('tools') and os.listdir('tools'):
        tool_files = [f for f in os.listdir('tools') if f.endswith('.json')]
        print(f"✓ Found {len(tool_files)} tool definitions:")
        for tool_file in tool_files:
            tool_name = tool_file.replace('.json', '')
            print(f"  - {tool_name}")
    else:
        print("✗ No tools found in tools/ directory")
        return False
    
    # Test API server (if running)
    try:
        import requests
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ LLM server is running")
            print(f"  - Model: {data.get('model_name', 'Unknown')}")
            print(f"  - Tool-calling: {data.get('tool_calling_enabled', False)}")
            print(f"  - MCP support: {data.get('mcp_enabled', False)}")
            return True
        else:
            print("⚠️  LLM server not responding properly")
            return False
    except Exception:
        print("⚠️  LLM server not running")
        print("   Start with: python setup.py --server")
        return False

def main():
    parser = argparse.ArgumentParser(description="RAG system with universal LLM support")
    parser.add_argument("--check", action="store_true", help="Check system requirements")
    parser.add_argument("--setup", action="store_true", help="Perform initial setup")
    parser.add_argument("--download", action="store_true", help="Download LLM model")
    parser.add_argument("--build-index", action="store_true", help="Build RAG index")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--server", action="store_true", help="Run FastAPI server with tool-calling")
    parser.add_argument("--test-tools", action="store_true", help="Test tool-calling functionality")
    parser.add_argument("--all", action="store_true", help="Perform all setup steps")
    
    args = parser.parse_args()
    
    print("🚀 RAG System with Universal LLM Support - Tool-calling & MCP")
    print("=" * 60)
    
    # Check system
    if args.check or args.all:
        print("\n1. System Requirements Check")
        if not check_system():
            print("\n❌ System check failed. Please resolve issues before continuing.")
            return
    
    # Initial setup
    if args.setup or args.all:
        print("\n2. Initial Setup")
        setup_directories()
    
    # Model download
    if args.download or args.all:
        print("\n3. LLM Model Download")
        if not run_download_models():
            print("\n❌ Model download failed. Please check your internet connection.")
            return
    
    # Index building
    if args.build_index or args.all:
        print("\n4. RAG Index Building")
        if not os.path.exists('documents') or not os.listdir('documents'):
            print("✗ No documents found in documents/ directory.")
            print("  Sample document was created. Add your own documents and try again.")
            return
        
        if not run_build_index():
            print("\n❌ Index building failed.")
            return
    
    # Test tool-calling
    if args.test_tools:
        print("\n🔧 Tool-calling Test")
        test_tool_calling()
        return
    
    # Run server
    if args.server:
        print("\n🌐 Starting FastAPI Server")
        run_server()
        return
    
    # Run interactive mode
    if args.interactive or (args.all and not args.server):
        print("\n💬 RAG System Interactive Mode")
        run_interactive()
        return
    
    # Show usage if no arguments
    if not any(vars(args).values()):
        print("\n📖 Usage Examples:")
        print("  python setup.py --check           # Check system requirements")
        print("  python setup.py --setup           # Initial setup")
        print("  python setup.py --download        # Download LLM model")
        print("  python setup.py --build-index     # Build RAG index")
        print("  python setup.py --interactive     # Run interactive chat")
        print("  python setup.py --server          # Run FastAPI server")
        print("  python setup.py --test-tools      # Test tool-calling")
        print("  python setup.py --all             # Complete setup + interactive")
        print()
        print("🔧 For FastAPI server with tool-calling:")
        print("  1. python setup.py --all")
        print("  2. python setup.py --server")
        print("  3. Open http://localhost:8000/docs in browser")
        print()
        print("💡 Quick start: python setup.py --all")
        print()
        print("🤖 Supported LLM types:")
        print("  - Kimi K2 (MoE, tool-calling)")
        print("  - Qwen series (multilingual)")
        print("  - DeepSeek (code-specialized)")
        print("  - SmolLM3 (efficient)")
        print("  - Llama (Meta)")
        print("  - Mistral/Mixtral (MoE)")

if __name__ == "__main__":
    main() 