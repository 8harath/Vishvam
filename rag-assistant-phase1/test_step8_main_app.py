#!/usr/bin/env python3
"""
Test and Validation Script for Step 8: Main Application & Demo

This script validates the main application functionality and tests
real-world use cases with the product manual.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.rag_pipeline import create_rag_pipeline

def run_command_test(command: str, description: str, timeout: int = 60):
    """Run a command and return success/failure."""
    print(f"\n🧪 Testing: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command.split(), 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            return True
        else:
            print(f"❌ FAILED (return code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (exceeded {timeout}s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_pipeline_directly():
    """Test the RAG pipeline directly without CLI."""
    print("\n🧪 Testing RAG Pipeline Directly")
    
    try:
        # Initialize pipeline
        rag_pipeline = create_rag_pipeline()
        print("✅ Pipeline initialized")
        
        # Load document
        pdf_path = "sample_data/product_manual.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ Test PDF not found: {pdf_path}")
            return False
        
        success = rag_pipeline.load_document(pdf_path)
        if not success:
            print("❌ Failed to load document")
            return False
        print("✅ Document loaded successfully")
        
        # Test questions
        test_questions = [
            "What is the warranty period?",
            "How do I set up the router?",
            "What is customer support contact?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            result = rag_pipeline.generate_answer(question)
            
            if result['success']:
                print(f"✅ Answer: {result['answer'][:100]}...")
                print(f"⏱️ Time: {result['total_time']:.2f}s")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                return False
        
        print("\n✅ All direct pipeline tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test error: {str(e)}")
        return False

def validate_files():
    """Validate that all required files exist."""
    print("\n🧪 Validating File Structure")
    
    required_files = [
        "main_step8.py",
        "modules/rag_pipeline.py",
        "modules/pdf_parser.py",
        "modules/text_splitter.py",
        "modules/embedder.py",
        "modules/llm_handler.py",
        "config.py",
        "sample_data/product_manual.pdf"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_cli_help():
    """Test CLI help functionality."""
    print("\n🧪 Testing CLI Help")
    
    try:
        result = subprocess.run(
            ["python", "main_step8.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0 and "RAG Assistant" in result.stdout:
            print("✅ CLI help works correctly")
            return True
        else:
            print(f"❌ CLI help failed")
            return False
            
    except Exception as e:
        print(f"❌ CLI help error: {str(e)}")
        return False

def test_single_question():
    """Test single question mode."""
    print("\n🧪 Testing Single Question Mode")
    
    command = [
        "python", "main_step8.py",
        "--pdf", "sample_data/product_manual.pdf",
        "--question", "What is the warranty period?"
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,  # Allow more time for processing
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            if "warranty" in result.stdout.lower():
                print("✅ Single question mode works correctly")
                print(f"Sample output: {result.stdout[:200]}...")
                return True
            else:
                print("❌ No relevant answer found in output")
                return False
        else:
            print(f"❌ Single question mode failed (code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Single question mode error: {str(e)}")
        return False

def main():
    """Run all tests and validations for Step 8."""
    print("=" * 60)
    print("RAG Assistant Phase 1 - Step 8 Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: File structure validation
    test_results.append(("File Structure", validate_files()))
    
    # Test 2: CLI help
    test_results.append(("CLI Help", test_cli_help()))
    
    # Test 3: Direct pipeline test
    test_results.append(("Direct Pipeline", test_pipeline_directly()))
    
    # Test 4: Single question mode
    test_results.append(("Single Question", test_single_question()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests PASSED! Step 8 implementation is working correctly.")
        
        # Show usage examples
        print("\n" + "=" * 60)
        print("USAGE EXAMPLES")
        print("=" * 60)
        print("\n1. Interactive mode:")
        print("   python main_step8.py --pdf sample_data/product_manual.pdf --interactive")
        print("\n2. Demo questions:")
        print("   python main_step8.py --pdf sample_data/product_manual.pdf --demo")
        print("\n3. Single question:")
        print("   python main_step8.py --pdf sample_data/product_manual.pdf --question \"What is the warranty period?\"")
        print("\n4. Status check:")
        print("   python main_step8.py --pdf sample_data/product_manual.pdf --status")
        
    else:
        print(f"\n⚠️ {total - passed} tests FAILED. Please check the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
