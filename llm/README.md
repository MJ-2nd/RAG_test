# LLM Server - Clean Architecture

기존 `app.py`를 클린 코드 원칙에 따라 리팩토링한 LLM 서버입니다.

## 🏗️ 아키텍처

### 리팩토링 원칙
- **단일 책임 원칙 (SRP)**: 각 클래스는 하나의 책임만 가집니다
- **의존성 주입**: 인터페이스를 통한 느슨한 결합
- **모듈화**: 기능별로 명확히 분리된 모듈
- **확장 가능성**: 새로운 기능 추가가 용이
- **가독성**: 코드의 의도가 명확히 드러남

### 패키지 구조

```
llm/
├── constants.py              # 상수 정의
├── config/                   # 설정 관리
│   ├── __init__.py
│   └── config_manager.py     # ConfigManager
├── models/                   # 모델 관리
│   ├── __init__.py
│   └── model_manager.py      # ModelManager
├── tools/                    # 도구 관리
│   ├── __init__.py
│   ├── tool_manager.py       # ToolManager
│   └── tool_executor.py      # ToolExecutor
├── services/                 # 비즈니스 로직
│   ├── __init__.py
│   ├── prompt_formatter.py   # PromptFormatter
│   └── text_generator.py     # TextGenerator
├── core/                     # 코어 클래스
│   ├── __init__.py
│   └── llm_server.py         # LLMServer (메인)
├── api/                      # API 모델
│   ├── __init__.py
│   └── models.py             # Pydantic 모델
├── app_refactored.py         # 새로운 FastAPI 앱
└── app.py                    # 기존 파일 (참고용)
```

## 🔧 주요 클래스

### 1. ConfigManager
- 설정 파일 로딩 및 관리
- 각종 설정값에 대한 접근 인터페이스 제공

### 2. ModelManager  
- 모델 로딩 및 초기화
- GPU 설정 및 양자화 관리
- 모델 타입 감지

### 3. ToolManager
- 도구 레지스트리 관리
- 도구 호출 파싱
- 기본 도구 정의

### 4. ToolExecutor
- 실제 도구 실행
- 각 도구별 실행 로직 분리

### 5. PromptFormatter
- 모델 타입별 프롬프트 포맷팅
- 채팅 템플릿 적용
- 도구 섹션 생성

### 6. TextGenerator
- 텍스트 생성 로직
- Stop sequence 처리
- 도구 호출과 통합

### 7. LLMServer (Core)
- 모든 컴포넌트를 조합하는 메인 클래스
- 깔끔한 인터페이스 제공

## 🚀 사용법

### 기본 사용
```python
from llm import LLMServer

# 서버 초기화
server = LLMServer("llm/config.yaml")
server.initialize()

# 텍스트 생성
prompt = server.format_chat_prompt("안녕하세요")
result = server.generate(prompt)
print(result['response'])
```

### FastAPI 서버 실행
```python
# 리팩토링된 앱 실행
python -m llm.app_refactored

# 기존 방식과 동일한 API 엔드포인트 제공
# POST /generate
# GET /tools  
# GET /health
```

## 📈 개선사항

### Before (기존 app.py)
- ❌ 730줄의 거대한 파일
- ❌ LLMServer 클래스에 모든 기능 집중
- ❌ 하드코딩된 값들
- ❌ 긴 메서드들 (100줄 이상)
- ❌ 높은 결합도

### After (리팩토링)
- ✅ 기능별로 분리된 작은 모듈들
- ✅ 단일 책임을 가진 클래스들
- ✅ 상수로 분리된 설정값들
- ✅ 짧고 명확한 메서드들 (평균 10-20줄)
- ✅ 느슨한 결합과 의존성 주입

## 🔄 마이그레이션

기존 코드에서 새로운 구조로 전환:

```python
# Before
from llm.app import LLMServer

# After  
from llm import LLMServer
```

모든 기존 API는 동일하게 작동하므로 클라이언트 코드 변경 불필요.

## 🧪 테스트 용이성

각 컴포넌트가 독립적이므로 단위 테스트 작성이 쉬움:

```python
# 예시: ConfigManager 테스트
def test_config_manager():
    config_manager = ConfigManager("test_config.yaml")
    assert config_manager.model_name == "test-model"

# 예시: Mock을 이용한 TextGenerator 테스트  
def test_text_generator_with_mock():
    mock_model_manager = Mock()
    mock_tool_manager = Mock()
    
    generator = TextGenerator(config_manager, mock_model_manager, mock_tool_manager)
    # 테스트 로직...
```

## 🔮 확장 가능성

새로운 기능 추가가 매우 쉬움:

```python
# 새로운 도구 추가
class CustomToolExecutor(ToolExecutor):
    def _execute_custom_tool(self, tool_call):
        # 새로운 도구 로직
        pass

# 새로운 모델 타입 지원
class EnhancedModelManager(ModelManager):
    def detect_model_type(self, model_name):
        # 확장된 모델 타입 감지
        pass
```

이렇게 리팩토링된 구조는 유지보수성, 테스트 용이성, 확장성을 크게 향상시킵니다. 