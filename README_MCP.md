# MCP (Model Context Protocol) Implementation

이 프로젝트는 **MCP (Model Context Protocol)**을 구현하여 LLM이 프롬프트 기반으로 함수를 호출할 수 있도록 하는 시스템입니다.

## 🎯 주요 특징

- **프롬프트 기반 함수 호출**: JSON 파일이 아닌 프롬프트에 함수 정의 포함
- **다양한 형식 지원**: XML, JSON 블록, 함수 호출 등 다양한 형식 파싱
- **클린코드 원칙**: 모듈화, 확장가능성, 가독성 철저히 준수
- **현재 시간 함수**: 실제 예시 함수로 현재 시간 반환 기능 구현

## 🏗️ 아키텍처

```
mcp/
├── __init__.py          # 패키지 초기화
├── functions.py         # 함수 정의 및 레지스트리
├── parser.py           # LLM 응답 파싱
├── prompt_builder.py   # MCP 프롬프트 생성
└── handler.py          # 함수 실행 핸들러

llm/
└── app.py              # MCP 통합된 LLM 서버

test_mcp.py             # 테스트 스크립트
```

## 🔧 구현된 함수들

### 1. `get_current_time()`
- **설명**: 현재 날짜와 시간을 ISO 형식으로 반환
- **매개변수**: 없음
- **반환값**: `"2025-01-27T15:30:45.123456"`

### 2. `get_current_date()`
- **설명**: 현재 날짜를 YYYY-MM-DD 형식으로 반환
- **매개변수**: 없음
- **반환값**: `"2025-01-27"`

### 3. `get_current_time_formatted(format_str)`
- **설명**: 사용자 정의 형식으로 현재 시간 반환
- **매개변수**: 
  - `format_str` (선택): 시간 형식 문자열 (기본값: `"%Y-%m-%d %H:%M:%S"`)
- **반환값**: `"2025-01-27 15:30:45"`

## 🚀 사용법

### 1. 서버 실행
```bash
python -m llm.app
```

### 2. API 호출 예시

#### 현재 시간 요청
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the current time?",
    "max_tokens": 512,
    "temperature": 0.3
  }'
```

#### 응답 예시
```json
{
  "response": "The current time is 2025-01-27T15:30:45.123456",
  "function_calls": [
    {
      "name": "get_current_time",
      "arguments": {},
      "confidence": 1.0
    }
  ],
  "function_results": [
    {
      "function_name": "get_current_time",
      "arguments": {},
      "confidence": 1.0,
      "execution_result": {
        "result": "2025-01-27T15:30:45.123456"
      },
      "success": true
    }
  ],
  "generation_info": {
    "max_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.9,
    "has_functions": true
  }
}
```

### 3. 사용 가능한 함수 확인
```bash
curl "http://localhost:8000/functions"
```

### 4. 서버 상태 확인
```bash
curl "http://localhost:8000/health"
```

## 🧪 테스트

### 자동 테스트 실행
```bash
python test_mcp.py
```

### 테스트 결과 예시
```
🚀 Starting MCP Tests
==================================================
✅ Health check passed
   Model: Qwen/Qwen2.5-7B-Instruct
   Model type: qwen
   MCP enabled: true
   Available functions: 3

✅ Functions list retrieved
   Total functions: 3
   Available functions:
     - get_current_time
     - get_current_date
     - get_current_time_formatted

==================================================
🧪 Testing Current Time Function

🔍 Testing query: 'What is the current time?'
✅ Query processed successfully
   Response: The current time is 2025-01-27T15:30:45.123456
   Function calls: 1
     1. get_current_time
        Arguments: {}
   Function results: 1
     1. get_current_time: ✅ 2025-01-27T15:30:45.123456
✅ Expected function 'get_current_time' was called
```

## 📋 지원하는 함수 호출 형식

### 1. XML 형식
```xml
<function_call>
{"name": "get_current_time", "arguments": {}}
</function_call>
```

### 2. JSON 코드 블록
```json
```json
{"name": "get_current_time", "arguments": {}}
```
```

### 3. Tool Call 형식
```xml
<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call>
```

### 4. 함수 호출 형식
```
get_current_time()
```

## 🔧 새로운 함수 추가하기

### 1. 함수 정의
`mcp/functions.py`에 새로운 함수를 추가:

```python
def my_custom_function(param1: str, param2: int = 10) -> str:
    """My custom function description"""
    return f"Processed {param1} with {param2}"

# 함수 등록
FunctionDefinition(
    name="my_custom_function",
    description="My custom function description",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "First parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Second parameter",
                "default": 10
            }
        },
        "required": ["param1"]
    },
    required_params=["param1"],
    function=my_custom_function
)
```

### 2. 자동 등록
`_register_default_functions()` 메서드에 추가하면 서버 시작 시 자동으로 등록됩니다.

## 🎨 클린코드 특징

### 1. **단일 책임 원칙**
- `functions.py`: 함수 정의 및 관리
- `parser.py`: 응답 파싱만 담당
- `prompt_builder.py`: 프롬프트 생성만 담당
- `handler.py`: 함수 실행만 담당

### 2. **개방-폐쇄 원칙**
- 새로운 함수 추가 시 기존 코드 수정 없이 확장 가능
- 새로운 파싱 형식 추가 시 기존 코드 영향 없음

### 3. **의존성 역전 원칙**
- 추상화된 인터페이스 사용
- 구체적인 구현에 의존하지 않음

### 4. **모듈화**
- 각 기능이 독립적인 모듈로 분리
- 명확한 인터페이스 정의

## 🔍 디버깅

### 로그 확인
```bash
# 서버 로그에서 MCP 관련 정보 확인
tail -f server.log | grep MCP
```

### 함수 실행 로그
```
INFO:mcp.functions:Registered MCP function: get_current_time
INFO:mcp.parser:Parsing function calls from text: <function_call>...
INFO:mcp.handler:Executing function call 1/1: get_current_time
INFO:mcp.functions:Executed function 'get_current_time' with result: 2025-01-27T15:30:45.123456
```

## 🚀 성능 최적화

### 1. 함수 캐싱
- 자주 사용되는 함수 결과 캐싱
- 메모리 효율적인 캐시 구현

### 2. 병렬 실행
- 여러 함수 호출 시 병렬 처리
- 비동기 함수 실행 지원

### 3. 타임아웃 설정
- 함수 실행 타임아웃 설정
- 무한 대기 방지

## 🔒 보안 고려사항

### 1. 함수 검증
- 함수 호출 전 매개변수 검증
- 허용되지 않은 함수 호출 차단

### 2. 입력 검증
- 사용자 입력 sanitization
- SQL 인젝션, XSS 등 공격 방지

### 3. 권한 관리
- 함수별 실행 권한 설정
- 역할 기반 접근 제어

## 📈 확장 계획

### 1. 추가 함수들
- 파일 시스템 접근
- 네트워크 요청
- 데이터베이스 쿼리
- 외부 API 호출

### 2. 고급 기능
- 함수 체이닝
- 조건부 함수 실행
- 에러 처리 및 복구

### 3. 모니터링
- 함수 실행 통계
- 성능 메트릭
- 에러 추적

## 🤝 기여하기

1. 새로운 함수 추가
2. 파싱 형식 개선
3. 성능 최적화
4. 문서 개선
5. 테스트 케이스 추가

## 📄 라이선스

MIT License 