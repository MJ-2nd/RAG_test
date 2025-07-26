"""
Constants for LLM server
"""

# Model Types
class ModelType:
    KIMI = "kimi"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    SMOLLM = "smollm"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GENERIC = "generic"

# Chat Templates
class ChatTemplate:
    CHATML = "chatml"
    LLAMA = "llama"
    MISTRAL = "mistral"

# Quantization Methods
class QuantizationMethod:
    AWQ = "awq"
    MOE_OPTIMIZED = "moe_optimized"
    BITSANDBYTES = "bitsandbytes"

# Tool Call Formats
class ToolCallFormat:
    JSON = "json"
    XML = "xml"

# Default Values
class Defaults:
    TIMEOUT = 30
    MAX_TOKENS = 2048
    TEMPERATURE = 0.3
    TOP_P = 0.9
    REPETITION_PENALTY = 1.05
    DEVICE_CACHE_DURATION = 5

# Tool Names
class ToolNames:
    SEARCH_DOCUMENTS = "search_documents"
    CALCULATE = "calculate"
    GET_CURRENT_TIME = "get_current_time"
    CONTROL_ANDROID_DEVICE = "control_android_device"

# Error Messages
class ErrorMessages:
    LLM_NOT_INITIALIZED = "LLM server not initialized."
    ADB_NOT_INSTALLED = "ADB is not installed or not accessible"
    NO_DEVICES_CONNECTED = "No Android devices connected"
    UNKNOWN_TOOL = "Unknown tool: {}"
    TOOL_EXECUTION_FAILED = "Tool execution failed: {}" 