#!/usr/bin/env python3
"""
Multi-Step MCP Test Script

This script tests the multi-step MCP implementation with function chaining.
"""

import requests
import json
import time
from typing import Dict, Any


class MultiStepMCPTester:
    """Test multi-step MCP functionality"""
    
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
    
    def test_multi_step_query(self, query: str, expected_functions: list = None) -> bool:
        """Test multi-step query with function chaining"""
        try:
            print(f"\n🔍 Testing multi-step query: '{query}'")
            
            payload = {
                "query": query,
                "max_tokens": 512,
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=120  # Longer timeout for multi-step processing
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Multi-step query processed successfully")
                print(f"   Response: {data.get('response', '')[:200]}...")
                
                # Check processing mode
                processing_mode = data.get('generation_info', {}).get('processing_mode', 'unknown')
                print(f"   Processing mode: {processing_mode}")
                
                # Check iterations
                iterations = data.get('iterations', 0)
                if iterations > 0:
                    print(f"   Iterations: {iterations}")
                
                # Check conversation history
                conversation_history = data.get('conversation_history', [])
                if conversation_history:
                    print(f"   Conversation steps: {len(conversation_history)}")
                    for i, step in enumerate(conversation_history):
                        print(f"     Step {i+1}: {step.get('query', '')[:50]}...")
                        func_calls = step.get('function_calls', [])
                        if func_calls:
                            for func_call in func_calls:
                                print(f"       → Called: {func_call.get('name', 'Unknown')}")
                
                # Check function calls
                function_calls = data.get('function_calls', [])
                function_results = data.get('function_results', [])
                
                if function_calls:
                    print(f"   Total function calls: {len(function_calls)}")
                    for i, func_call in enumerate(function_calls):
                        print(f"     {i+1}. {func_call.get('name', 'Unknown')}")
                        print(f"        Arguments: {func_call.get('arguments', {})}")
                
                if function_results:
                    print(f"   Function results:")
                    for i, result in enumerate(function_results):
                        func_name = result.get('function_name', 'Unknown')
                        success = result.get('success', False)
                        exec_result = result.get('execution_result', {})
                        
                        if success:
                            print(f"     {i+1}. {func_name}: ✅ {exec_result.get('result', 'No result')}")
                        else:
                            print(f"     {i+1}. {func_name}: ❌ {exec_result.get('error', 'Unknown error')}")
                
                # Check expected functions
                if expected_functions:
                    called_functions = [fc.get('name') for fc in function_calls]
                    missing_functions = [f for f in expected_functions if f not in called_functions]
                    if missing_functions:
                        print(f"❌ Missing expected functions: {missing_functions}")
                        return False
                    else:
                        print(f"✅ All expected functions were called: {expected_functions}")
                
                return True
            else:
                print(f"❌ Query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all multi-step MCP tests"""
        print("🚀 Starting Multi-Step MCP Tests")
        print("=" * 60)
        
        # Test 1: Health check
        if not self.test_health():
            print("❌ Health check failed, stopping tests")
            return
        
        # Test 2: Functions list
        if not self.test_functions_list():
            print("❌ Functions list failed, stopping tests")
            return
        
        # Test 3: Simple time query (single step)
        print("\n" + "=" * 60)
        print("🧪 Testing Simple Time Query (Single Step)")
        self.test_multi_step_query(
            "What is the current time?",
            expected_functions=["get_current_time"]
        )
        
        # Test 4: Restaurant reservation (multi-step)
        print("\n" + "=" * 60)
        print("🧪 Testing Restaurant Reservation (Multi-Step)")
        self.test_multi_step_query(
            "I want to make a restaurant reservation for 7:30 PM tonight",
            expected_functions=["get_current_time", "check_restaurant_availability", "make_restaurant_reservation"]
        )
        
        # Test 5: Complex restaurant query
        print("\n" + "=" * 60)
        print("🧪 Testing Complex Restaurant Query")
        self.test_multi_step_query(
            "Can you check what restaurants are available at 8 PM and then make a reservation for 2 people at the Italian restaurant?",
            expected_functions=["check_restaurant_availability", "make_restaurant_reservation"]
        )
        
        # Test 6: Menu and reservation
        print("\n" + "=" * 60)
        print("🧪 Testing Menu and Reservation")
        self.test_multi_step_query(
            "Show me the menu for Korean BBQ and then make a reservation for 4 people at 6:30 PM",
            expected_functions=["get_restaurant_menu", "make_restaurant_reservation"]
        )
        
        # Test 7: No function needed
        print("\n" + "=" * 60)
        print("🧪 Testing No Function Needed")
        self.test_multi_step_query(
            "Hello, how are you today?",
            expected_functions=None
        )
        
        print("\n" + "=" * 60)
        print("✅ Multi-Step MCP Tests Completed")


def main():
    """Main function"""
    tester = MultiStepMCPTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 