# RAG System with Advanced Tool-calling & MCP Support

이 프로젝트는 **다양한 최신 LLM 모델**을 지원하는 고급 RAG (Retrieval-Augmented Generation) 시스템입니다. **Tool-calling** 및 **MCP (Model Context Protocol)** 를 완전 지원하며, 32GB VRAM 환경에 최적화되어 있습니다.

## 🚀 주요 특징

- **다양한 LLM 지원**: Kimi K2, Qwen, DeepSeek, SmolLM3, Llama, Mistral 등
- **Tool-calling 지원**: 계산, 검색, 파일 탐색 등 도구 자동 실행
- **MCP 지원**: Model Context Protocol로 확장 가능한 에이전트 구조
- **32GB VRAM 최적화**: 양자화 및 분산 처리로 효율적인 메모리 사용
- **듀얼 GPU 지원**: Tensor Parallelism으로 성능 최적화
- **모델별 최적화**: 각 모델 타입에 맞는 채팅 템플릿과 설정

## 🔧 시스템 아키텍처

### 지원 모델 (2025년 최신)
- **Kimi K2**: 1T MoE, 32B active, Native tool-calling (~16GB)
- **DeepSeek R1**: 코드 특화, Tool-calling 지원 (7B~32B)
- **SmolLM3**: 효율적인 소형 모델, Tool-calling 지원 (~6GB)
- **Qwen 시리즈**: 다국어 지원, 부분적 tool-calling (7B~32B)
- **Llama**: Meta의 오픈소스 모델 (8B~70B)
- **Mistral/Mixtral**: MoE 아키텍처 (7B~8x7B)

### 리소스 할당
- **VRAM**: 모델 전용 (8GB~32GB, 모델 크기에 따라)
- **CPU/RAM**: 임베딩 모델 + 검색 시스템 + 도구 실행
- **Storage**: 모델 파일 (6GB~140GB), 인덱스 파일

## ⚡ 빠른 시작

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 추가 패키지 (필요시)
pip install huggingface-hub transformers>=4.45.0
```

### 2. 모델 선택 및 다운로드
```bash
# config.yaml에서 원하는 모델 선택
# 예: Kimi K2, DeepSeek R1, SmolLM3 등

# 모든 모델 다운로드
python download_models.py

# 특정 모델만 다운로드
python download_models.py --llm-only

# 다운로드 전 정보 확인
python download_models.py --dry-run
```

### 3. 문서 준비
```bash
# documents/ 디렉토리에 문서 배치
cp your_documents.txt documents/
```

### 4. 인덱스 구축
```bash
# 기본 인덱스 생성
python -m rag.build_index

# 사용자 정의 설정
python -m rag.build_index --doc_dir documents --index_path models/my_index
```

### 5. RAG 시스템 실행

#### 대화형 모드
```bash
python -m query.query_rag --interactive
```

#### FastAPI 서버 (Tool-calling 지원)
```bash
# LLM 서버 실행
python -m llm.app

# 브라우저에서 http://localhost:8000/docs 접속
```

## 🛠️ Tool-calling 사용법

### 기본 제공 도구들

1. **문서 검색** (`search_documents`)
   ```json
   {
     "name": "search_documents",
     "arguments": {
       "query": "인공지능 개발 방법",
       "top_k": 5
     }
   }
   ```

2. **수학 계산** (`calculate`)
   ```json
   {
     "name": "calculate", 
     "arguments": {
       "expression": "2 + 3 * 4"
     }
   }
   ```

3. **현재 시간** (`get_current_time`)
   ```json
   {
     "name": "get_current_time",
     "arguments": {}
   }
   ```

4. **파일 검색** (`search_files`)
   ```json
   {
     "name": "search_files",
     "arguments": {
       "pattern": "*.py",
       "directory": "."
     }
   }
   ```

### 도구 사용 예시

**질문**: "현재 시간을 알려주고, 2024년부터 몇 년이 지났는지 계산해줘"

**LLM 자동 응답** (모델에 따라 형식이 다를 수 있음):
```
<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call>

<tool_call>
{"name": "calculate", "arguments": {"expression": "2025 - 2024"}}
</tool_call>

현재 시간은 2025-01-XX이고, 2024년부터 1년이 지났습니다.
```

## 🔧 상세 설정

### 모델 설정 (`llm/config.yaml`)

```yaml
llm:
  # 현재 선택된 모델 (쉽게 변경 가능)
  model_name: "moonshotai/Kimi-K2-Instruct"  # 또는 다른 모델
  model_path: "./models/kimi-k2-instruct"
  
  # Tool-calling 설정 (모델별 자동 최적화)
  tool_calling:
    enabled: true
    format: "json"  # 모델에 따라 자동 선택
    max_tools_per_call: 5
    parallel_tools: true
    
  # GPU 설정 (모델 크기에 따라 자동 조정)
  vllm:
    tensor_parallel_size: 2      # GPU 수
    max_model_len: 32768         # 컨텍스트 길이
    gpu_memory_utilization: 0.85
    
  # Generation 설정 (Tool-calling 최적화)
  generation:
    max_tokens: 2048
    temperature: 0.3  # Tool-calling에는 낮은 temperature 권장
    top_p: 0.9

# MCP 설정
mcp:
  enabled: true
  protocol_version: "2025.1"
  tools_registry: "./tools/"

# RAG 설정 (Tool-calling 연동)
rag:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5
  similarity_threshold: 0.3
  enable_tool_retrieval: true
```

### 모델별 권장 설정

#### 고성능 모델 (32GB+ VRAM)
```yaml
llm:
  model_name: "moonshotai/Kimi-K2-Instruct"        # MoE, ~16GB
  # model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # ~65GB
```

#### 중간 성능 모델 (16GB+ VRAM)
```yaml
llm:
  model_name: "Qwen/Qwen2.5-14B-Instruct-AWQ"     # ~6GB
  # model_name: "HuggingFaceTB/SmolLM3-3B-Instruct"     # ~6GB
```

#### 효율적 모델 (8GB+ VRAM)
```yaml
llm:
  model_name: "Qwen/Qwen2.5-7B-Instruct"          # ~14GB
  # model_name: "mistralai/Mistral-7B-Instruct-v0.3"   # ~14GB
```

## 🌐 API 사용법

### FastAPI 엔드포인트

#### 1. 텍스트 생성 (Tool-calling 포함)
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "현재 시간을 알려주고 2+3을 계산해줘",
    "tools": [
      {
        "name": "get_current_time",
        "description": "Get current time",
        "parameters": {"type": "object", "properties": {}}
      },
      {
        "name": "calculate", 
        "description": "Perform calculations",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string"}
          }
        }
      }
    ]
  }'
```

#### 2. 사용 가능한 도구 목록
```bash
curl "http://localhost:8000/tools"
```

#### 3. 상태 확인
```bash
curl "http://localhost:8000/health"
```

## 🎯 성능 최적화

### VRAM 절약 팁
1. **모델 선택**: 용도에 맞는 적절한 크기 선택
2. **양자화 사용**: AWQ, BitsAndBytes 4bit 양자화
3. **듀얼 GPU 분산**: Tensor Parallelism으로 메모리 분산
4. **CPU 임베딩**: VRAM을 LLM 전용으로 활용

### Tool-calling 성능 향상
1. **낮은 Temperature**: Tool-calling에는 0.3 권장
2. **병렬 도구 실행**: `parallel_tools: true`로 속도 향상
3. **모델별 최적화**: 각 모델에 맞는 채팅 템플릿 자동 선택

## 🔧 사용자 정의 도구 추가

### 1. 도구 정의 파일 생성 (`tools/my_tool.json`)
```json
{
  "name": "my_custom_tool",
  "description": "내 사용자 정의 도구",
  "parameters": {
    "type": "object",
    "properties": {
      "input": {"type": "string", "description": "입력 값"}
    },
    "required": ["input"]
  }
}
```

### 2. 도구 실행 로직 추가 (`llm/app.py`)
```python
async def execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
    if tool_call.name == "my_custom_tool":
        input_value = tool_call.arguments.get('input')
        # 사용자 정의 로직 구현
        result = my_custom_logic(input_value)
        return {"result": result}
```

## 🐛 문제 해결

### 메모리 부족 오류
```bash
# GPU 메모리 사용률 줄이기
# config.yaml에서 gpu_memory_utilization을 0.8 이하로 설정

# 더 작은 모델 사용
# SmolLM3-3B 또는 Qwen2.5-7B로 변경
```

### 모델 로딩 실패
```bash
# 모델 파일 재다운로드
python download_models.py --llm-only

# HuggingFace 토큰 설정 (필요시)
huggingface-cli login
```

### Tool-calling 오류
```bash
# 도구 레지스트리 확인
ls tools/
python -c "import json; print(json.load(open('tools/calculate.json')))"

# 모델이 tool-calling을 지원하는지 확인
curl http://localhost:8000/health
```

## 💻 시스템 요구사항

### 하드웨어
- **GPU**: 8GB~32GB VRAM (모델에 따라)
  - **소형 모델**: 8GB (SmolLM3, Qwen-7B)
  - **중형 모델**: 16GB (Qwen-14B, DeepSeek-14B)
  - **대형 모델**: 32GB (Kimi K2, DeepSeek-32B)
- **CPU**: 멀티코어 프로세서 (임베딩 처리용)
- **RAM**: 최소 16GB, 권장 32GB+
- **Storage**: 최소 100GB 여유 공간

### 소프트웨어
- Python 3.9+
- CUDA 12.0+ (GPU 사용시)
- PyTorch 2.1+
- Transformers 4.45.0+

## 🚀 지원 모델 비교

| 모델 | 크기 | VRAM | Tool-calling | 특징 |
|------|------|------|--------------|------|
| **Kimi K2** | 1T MoE (32B active) | ~16GB | ✅ Native | MoE, 최신 |
| **DeepSeek R1** | 7B~32B | ~14GB~65GB | ✅ Good | 코드 특화 |
| **SmolLM3** | 3B | ~6GB | ✅ Good | 효율적 |
| **Qwen 2.5** | 7B~32B | ~14GB~65GB | ⚠️ Partial | 다국어 |
| **Llama 3.1** | 8B~70B | ~16GB~140GB | ⚠️ Limited | Meta |
| **Mistral** | 7B~8x7B | ~14GB~90GB | ⚠️ Limited | MoE |

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🤝 기여하기

버그 리포트, 기능 요청, Pull Request를 환영합니다.

## 💬 지원

질문이나 문제가 있으시면 이슈를 생성해 주세요.