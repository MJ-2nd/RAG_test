#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Test Script

This script tests the MCP implementation with various function calls.
"""

import requests
import json
import time
from typing import Dict, Any


class MCPTester:
    """Test MCP functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def test_health(self) -> bool:
        """Test server health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   Model: {data.get('model_name', 'Unknown')}")
                print(f"   Model type: {data.get('model_type', 'Unknown')}")
                print(f"   MCP enabled: {data.get('mcp_enabled', False)}")
                print(f"   Available functions: {data.get('available_functions', {}).get('total_functions', 0)}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_functions_list(self) -> bool:
        """Test functions list endpoint"""
        try:
            response = requests.get(f"{self.base_url}/functions", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Functions list retrieved")
                print(f"   Total functions: {data.get('functions', {}).get('total_functions', 0)}")
                print("   Available functions:")
                for func_name in data.get('functions', {}).get('function_names', []):
                    print(f"     - {func_name}")
                return True
            else:
                print(f"❌ Functions list failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Functions list error: {e}")
            return False
    
    def test_function_call(self, query: str, expected_function: str = None) -> bool:
        """Test function calling with a query"""
        try:
            print(f"\n🔍 Testing query: '{query}'")
            
            payload = {
                "query": query,
                "max_tokens": 512,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Query processed successfully")
                print(f"   Response: {data.get('response', '')[:100]}...")
                
                function_calls = data.get('function_calls', [])
                function_results = data.get('function_results', [])
                
                if function_calls:
                    print(f"   Function calls: {len(function_calls)}")
                    for i, func_call in enumerate(function_calls):
                        print(f"     {i+1}. {func_call.get('name', 'Unknown')}")
                        print(f"        Arguments: {func_call.get('arguments', {})}")
                
                if function_results:
                    print(f"   Function results: {len(function_results)}")
                    for i, result in enumerate(function_results):
                        func_name = result.get('function_name', 'Unknown')
                        success = result.get('success', False)
                        exec_result = result.get('execution_result', {})
                        
                        if success:
                            print(f"     {i+1}. {func_name}: ✅ {exec_result.get('result', 'No result')}")
                        else:
                            print(f"     {i+1}. {func_name}: ❌ {exec_result.get('error', 'Unknown error')}")
                
                if expected_function:
                    called_functions = [fc.get('name') for fc in function_calls]
                    if expected_function in called_functions:
                        print(f"✅ Expected function '{expected_function}' was called")
                        return True
                    else:
                        print(f"❌ Expected function '{expected_function}' was not called")
                        print(f"   Called functions: {called_functions}")
                        return False
                
                return True
            else:
                print(f"❌ Query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all MCP tests"""
        print("🚀 Starting MCP Tests")
        print("=" * 50)
        
        # Test 1: Health check
        if not self.test_health():
            print("❌ Health check failed, stopping tests")
            return
        
        # Test 2: Functions list
        if not self.test_functions_list():
            print("❌ Functions list failed, stopping tests")
            return
        
        # Test 3: Current time function
        print("\n" + "=" * 50)
        print("🧪 Testing Current Time Function")
        self.test_function_call(
            "What is the current time?",
            expected_function="get_current_time"
        )
        
        # Test 4: Current date function
        print("\n" + "=" * 50)
        print("🧪 Testing Current Date Function")
        self.test_function_call(
            "What is today's date?",
            expected_function="get_current_date"
        )
        
        # Test 5: Formatted time function
        print("\n" + "=" * 50)
        print("🧪 Testing Formatted Time Function")
        self.test_function_call(
            "Get the current time in a nice format",
            expected_function="get_current_time_formatted"
        )
        
        # Test 6: Multiple functions
        print("\n" + "=" * 50)
        print("🧪 Testing Multiple Functions")
        self.test_function_call(
            "Tell me the current time and date",
            expected_function="get_current_time"  # Should call at least one function
        )
        
        # Test 7: No function needed
        print("\n" + "=" * 50)
        print("🧪 Testing No Function Needed")
        self.test_function_call(
            "Hello, how are you today?",
            expected_function=None  # Should not call any functions
        )
        
        print("\n" + "=" * 50)
        print("✅ MCP Tests Completed")


def main():
    """Main function"""
    tester = MCPTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 