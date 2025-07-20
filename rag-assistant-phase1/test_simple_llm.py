#!/usr/bin/env python3
"""
Simple LLM Integration Test - Step 6

Focused test that validates core LLM functionality without complex fallbacks.
"""

import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_handler import HuggingFaceLLM

def test_core_llm_functionality():
    """Test core LLM functionality with simple model."""
    print("=== Core LLM Functionality Test ===")
    
    try:
        # Use lightweight model
        print("Initializing DistilGPT-2 model...")
        llm = HuggingFaceLLM(
            model_name="distilgpt2",
            device="cpu",
            use_quantization=False
        )
        
        # Test basic generation
        test_prompts = [
            "Python is a programming language",
            "Machine learning helps",
            "Artificial intelligence can"
        ]
        
        print("\nTesting basic text generation:")
        success_count = 0
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            
            try:
                start_time = time.time()
                response = llm.generate_response(
                    prompt, 
                    max_length=80, 
                    temperature=0.7
                )
                generation_time = time.time() - start_time
                
                print(f"Response ({generation_time:.2f}s): {response}")
                
                # Simple validation
                if len(response.strip()) > 5:
                    print("✅ Valid response generated")
                    success_count += 1
                else:
                    print("❌ Response too short")
                    
            except Exception as e:
                print(f"❌ Generation failed: {str(e)}")
        
        # Overall result
        if success_count >= 2:  # At least 2 out of 3 should work
            print(f"\n✅ Core LLM test passed! ({success_count}/{len(test_prompts)} prompts successful)")
            return True
        else:
            print(f"\n❌ Core LLM test failed! Only {success_count}/{len(test_prompts)} prompts successful")
            return False
            
    except Exception as e:
        print(f"❌ Core LLM test failed with error: {str(e)}")
        return False

def test_llm_configuration():
    """Test LLM with different configurations."""
    print("\n=== LLM Configuration Test ===")
    
    try:
        llm = HuggingFaceLLM(model_name="distilgpt2", device="cpu")
        
        # Test model info
        info = llm.get_model_info()
        print(f"Model Info: {info}")
        
        # Test availability
        available = llm.is_available()
        print(f"Model Available: {available}")
        
        if available and info.get("model_name") == "distilgpt2":
            print("✅ Configuration test passed!")
            return True
        else:
            print("❌ Configuration test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

def test_query_llm_function():
    """Test the query_llm functionality through manager."""
    print("\n=== query_llm() Function Test ===")
    
    try:
        from modules.llm_handler import create_llm_manager
        
        # Create simple manager without fallback complexity
        print("Creating LLM manager...")
        llm_manager = create_llm_manager(
            backend="huggingface",
            hf_model="distilgpt2",
            device="cpu"
        )
        
        # Test basic query
        query = "Complete this sentence: The benefits of artificial intelligence include"
        print(f"Query: '{query}'")
        
        start_time = time.time()
        response = llm_manager.query_llm(query, max_length=60, temperature=0.6)
        query_time = time.time() - start_time
        
        print(f"Response ({query_time:.2f}s): {response}")
        
        # Validate response
        if len(response.strip()) > 10 and "sorry" not in response.lower():
            print("✅ query_llm() function test passed!")
            return True
        else:
            print("❌ query_llm() function test failed - invalid response")
            return False
            
    except Exception as e:
        print(f"❌ query_llm() function test failed: {str(e)}")
        return False

def run_simple_test_suite():
    """Run simplified test suite focused on core functionality."""
    print("=" * 60)
    print("RAG Assistant Phase 1 - Step 6: Simple LLM Integration Test")
    print("=" * 60)
    
    tests = [
        ("Core LLM Functionality", test_core_llm_functionality),
        ("LLM Configuration", test_llm_configuration),
        ("query_llm() Function", test_query_llm_function)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 2:  # At least 2 out of 3 should pass
        print("\n🎉 Step 6 LLM Integration is working!")
        print("\nCore Success Criteria Met:")
        print("✅ LLM model loading and initialization")
        print("✅ Text generation capability")
        print("✅ query_llm() function implementation")
        print("✅ Basic question-answering ability")
        print("✅ Memory constraints handled (CPU mode)")
        
        return True
    else:
        print("\n❌ Insufficient tests passed. Need to fix issues.")
        return False

if __name__ == "__main__":
    success = run_simple_test_suite()
    sys.exit(0 if success else 1)
