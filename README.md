# RAG System - 32GB VRAM Optimized with Dual GPU Support

This project is a RAG (Retrieval-Augmented Generation) system optimized for 32GB VRAM environment with dual GPU model parallelism.

## System Architecture

### Model Selection Strategy
- **LLM**: Qwen2.5-32B-Instruct (4bit quantization, ~20-24GB VRAM across 2 GPUs)
- **Embedding**: BGE-M3 (CPU usage, ~2GB RAM)
- **Retrieval**: FAISS (CPU usage)

### Resource Allocation
- **VRAM**: LLM exclusive (~24GB total across 2 GPUs)
- **CPU/RAM**: Embedding model + retrieval system
- **Storage**: Model files (~70GB), index files (~hundreds of MB)

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Additional packages (if needed)
pip install huggingface-hub
```

### 2. Model Download
```bash
# Download all models
python download_models.py

# Download LLM only
python download_models.py --llm-only

# Download embedding model only
python download_models.py --embedding-only

# Check information before download
python download_models.py --dry-run
```

### 3. Document Preparation
```bash
# Place documents in documents/ directory
cp your_documents.txt documents/
```

### 4. Index Building
```bash
# Basic index building
python -m rag.build_index

# Custom settings
python -m rag.build_index --doc_dir documents --index_path models/my_index
```

### 5. RAG System Execution
```bash
# Interactive mode
python -m query.query_rag --interactive

# Single query
python -m query.query_rag --query "your question"

# Run without context
python -m query.query_rag --query "your question" --no-context
```

## Detailed Configuration

### Model Configuration (`llm/config.yaml`)

```yaml
llm:
  model_name: "Qwen/Qwen2.5-32B-Instruct"
  quantization:
    enabled: true
    bits: 4  # 4bit quantization for VRAM saving
  vllm:
    tensor_parallel_size: 2  # Use 2 GPUs for model parallelism
    gpu_memory_utilization: 0.85  # GPU memory utilization

embedding:
  model_name: "BAAI/bge-m3"
  device: "cpu"  # Run on CPU
  batch_size: 32

rag:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5
  similarity_threshold: 0.7
```

### Alternative Model Configurations

#### Smaller VRAM Usage (14B model)
```yaml
llm:
  model_name: "Qwen/Qwen2.5-14B-Instruct"
  # Uses ~28GB VRAM with FP16 (across 2 GPUs)
```

#### Faster Processing (7B model)
```yaml
llm:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  # Uses ~14GB VRAM with FP16 (across 2 GPUs)
```

## Usage

### Interactive Mode
```bash
python -m query.query_rag --interactive
```

Available commands in interactive mode:
- `/stats`: Show index statistics
- `/context on/off`: Toggle context usage
- `/topk <number>`: Set number of documents to retrieve
- `quit` or `exit`: Exit

### FastAPI Server
```bash
# Run LLM server
python -m llm.app

# Access http://localhost:8000/docs in browser
```

API endpoints:
- `POST /generate`: Text generation
- `GET /health`: Health check

## Performance Optimization

### VRAM Saving Tips
1. **Use 4bit quantization**: ~75% memory usage reduction
2. **CPU embedding model**: Dedicated VRAM for LLM
3. **Adjust context length**: Control memory with `max_model_len` setting
4. **Dual GPU setup**: Distribute model across 2 GPUs

### Retrieval Performance Enhancement
1. **Adjust chunk size**: Tune `chunk_size` according to document characteristics
2. **Embedding batch size**: Adjust `batch_size` for CPU performance
3. **FAISS index type**: Automatically selected based on document count

### Response Quality Improvement
1. **Adjust similarity threshold**: Set `similarity_threshold`
2. **Optimize document count**: Tune `top_k` value
3. **Custom prompt templates**: Improve LLM output quality

## Troubleshooting

### Out of Memory Error
```bash
# Reduce GPU memory utilization
# Set gpu_memory_utilization to 0.8 or lower in config.yaml
```

### Model Loading Failure
```bash
# Re-download model files
python download_models.py --llm-only

# Set HuggingFace token (if needed)
huggingface-cli login
```

### Poor Retrieval Performance
```bash
# Rebuild index
python -m rag.build_index --doc_dir documents --index_path models/new_index
```

## System Requirements

### Hardware
- **GPU**: 2x GPUs with total 32GB VRAM (e.g., 2x RTX 4090, 2x RTX 3090, etc.)
- **CPU**: Multi-core processor (for embedding processing)
- **RAM**: Minimum 16GB, recommended 32GB+
- **Storage**: Minimum 100GB free space

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU usage)
- PyTorch 2.0+
- VLLM 0.2.0+

## Dual GPU Configuration

This system is configured to use 2 GPUs for model parallelism:
- **tensor_parallel_size: 2** - Distributes the 32B model across 2 GPUs
- **Automatic load balancing** - VLLM handles GPU memory distribution
- **Reduced per-GPU memory usage** - Each GPU uses ~12GB instead of 24GB

### GPU Requirements
- 2 GPUs with at least 16GB VRAM each
- GPUs should be of similar performance for optimal load balancing
- NVLink connection recommended but not required

## License

This project follows the MIT License.

## Contributing

Bug reports, feature requests, and pull requests are welcome.

## Support

Please create an issue if you have questions or problems.