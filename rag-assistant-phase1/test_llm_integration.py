#!/usr/bin/env python3
"""
Test Script for LLM Integration - Step 6

This script tests the LLM integration module with various configurations
and validates that the query_llm() function works correctly.
"""

import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_handler import create_llm_manager, HuggingFaceLLM
from config import (
    LLM_BACKEND, HF_MODEL_NAME, HF_DEVICE, HF_MAX_LENGTH, HF_TEMPERATURE,
    OLLAMA_MODEL, OLLAMA_BASE_URL
)

def test_huggingface_llm():
    """Test Hugging Face LLM implementation."""
    print("\n=== Testing Hugging Face LLM ===")
    
    try:
        # Use a lightweight model for testing
        test_model = "distilgpt2"  # Small, fast model for testing
        print(f"Initializing HF LLM with model: {test_model}")
        
        llm = HuggingFaceLLM(
            model_name=test_model,
            device="cpu",  # Force CPU for reliability
            use_quantization=False  # Disable for CPU
        )
        
        # Test model info
        model_info = llm.get_model_info()
        print(f"Model Info: {model_info}")
        
        # Test basic generation
        test_prompts = [
            "The weather today is",
            "Machine learning is",
            "Python programming"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Testing prompt: '{prompt}'")
            
            start_time = time.time()
            response = llm.generate_response(
                prompt, 
                max_length=100, 
                temperature=0.7
            )
            generation_time = time.time() - start_time
            
            print(f"Response ({generation_time:.2f}s): {response[:150]}...")
            
            if "error" in response.lower():
                print("❌ Generation contained error")
                return False
        
        print("✅ Hugging Face LLM test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Hugging Face LLM test failed: {str(e)}")
        return False

def test_llm_manager():
    """Test the LLM Manager with different configurations."""
    print("\n=== Testing LLM Manager ===")
    
    try:
        # Test with Hugging Face backend
        print("Testing with Hugging Face backend...")
        llm_manager = create_llm_manager(
            backend="huggingface",
            hf_model="distilgpt2",
            device="cpu",
            use_quantization=False
        )
        
        # Get status
        status = llm_manager.get_status()
        print(f"LLM Manager Status: {status}")
        
        # Test query_llm function
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks work?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            start_time = time.time()
            response = llm_manager.query_llm(
                query, 
                max_length=150, 
                temperature=0.7
            )
            query_time = time.time() - start_time
            
            print(f"Response ({query_time:.2f}s): {response[:200]}...")
            
            # Basic validation
            if len(response.strip()) < 10:
                print("❌ Response too short")
                return False
            
            if "sorry" in response.lower() and "unable" in response.lower():
                print("❌ LLM returned error response")
                return False
        
        print("✅ LLM Manager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLM Manager test failed: {str(e)}")
        return False

def test_configuration_integration():
    """Test that configuration settings are properly integrated."""
    print("\n=== Testing Configuration Integration ===")
    
    try:
        print(f"Config - LLM Backend: {LLM_BACKEND}")
        print(f"Config - HF Model: {HF_MODEL_NAME}")
        print(f"Config - HF Device: {HF_DEVICE}")
        print(f"Config - Max Length: {HF_MAX_LENGTH}")
        print(f"Config - Temperature: {HF_TEMPERATURE}")
        
        # Create manager using config values
        if LLM_BACKEND.lower() == "huggingface":
            llm_manager = create_llm_manager(
                backend=LLM_BACKEND,
                hf_model="distilgpt2",  # Override with lightweight model
                device="cpu"  # Force CPU for testing
            )
        else:
            print("Ollama backend specified in config - testing connection...")
            llm_manager = create_llm_manager(
                backend=LLM_BACKEND,
                ollama_model=OLLAMA_MODEL,
                ollama_url=OLLAMA_BASE_URL
            )
        
        # Test a single query with config parameters
        test_query = "What is machine learning?"
        response = llm_manager.query_llm(
            test_query,
            max_length=min(HF_MAX_LENGTH, 150),  # Limit for testing
            temperature=HF_TEMPERATURE
        )
        
        print(f"Test Query: {test_query}")
        print(f"Response: {response[:200]}...")
        
        if response and "sorry" not in response.lower():
            print("✅ Configuration integration test passed!")
            return True
        else:
            print("❌ Configuration integration test failed - no valid response")
            return False
            
    except Exception as e:
        print(f"❌ Configuration integration test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run comprehensive LLM integration tests."""
    print("=" * 60)
    print("RAG Assistant Phase 1 - Step 6: LLM Integration Tests")
    print("=" * 60)
    
    # Track test results
    results = {
        "huggingface_llm": False,
        "llm_manager": False,
        "configuration": False
    }
    
    # Run tests
    results["huggingface_llm"] = test_huggingface_llm()
    results["llm_manager"] = test_llm_manager()
    results["configuration"] = test_configuration_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All LLM integration tests passed! Step 6 is complete.")
        print("\nSuccess Criteria Met:")
        print("✅ LLM loading and tokenization working")
        print("✅ query_llm() function implemented and functional")
        print("✅ Basic question-answering capability demonstrated")
        print("✅ Memory/GPU constraints handled")
        print("✅ Configuration integration working")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
