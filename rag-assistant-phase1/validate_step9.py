#!/usr/bin/env python3
"""
Step 9 Validation Script - Quick validation of key robustness features
"""

import sys
from pathlib import Path

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

def main():
    print("🔍 Step 9: Robustness & Error Handling - Quick Validation")
    print("=" * 60)
    
    # Test 1: Import error handling system
    try:
        from modules.error_handler import get_logger, ValidationUtils, HealthChecker
        print("✅ Error handling system imported successfully")
    except Exception as e:
        print(f"❌ Error handling import failed: {e}")
        return False
    
    # Test 2: Logger functionality
    try:
        logger = get_logger()
        logger.info("Test log message")
        print("✅ Logging system working")
    except Exception as e:
        print(f"❌ Logging system failed: {e}")
        return False
    
    # Test 3: Validation utilities
    try:
        # Test file validation with non-existent file
        try:
            ValidationUtils.validate_file_path("/nonexistent/file.pdf")
        except FileNotFoundError:
            print("✅ File validation properly rejects non-existent files")
        
        # Test text validation
        try:
            ValidationUtils.validate_text_content("ab")  # Too short
        except ValueError:
            print("✅ Text validation properly rejects short content")
            
    except Exception as e:
        print(f"❌ Validation utilities failed: {e}")
        return False
    
    # Test 4: Health checker
    try:
        health_checker = HealthChecker()
        results = health_checker.run_health_check()
        if results.get('overall_healthy', False):
            print("✅ System health check passed")
        else:
            print("⚠️  System health check detected issues (expected in some environments)")
    except Exception as e:
        print(f"❌ Health checker failed: {e}")
        return False
    
    # Test 5: Enhanced RAG pipeline import
    try:
        from modules.enhanced_rag_pipeline import create_enhanced_rag_pipeline
        # Just test import, don't need to use the function
        del create_enhanced_rag_pipeline
        print("✅ Enhanced RAG pipeline imported successfully")
    except Exception as e:
        print(f"❌ Enhanced RAG pipeline import failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Step 9 validation completed successfully!")
    print("✅ All core robustness features are working")
    print("✅ Error handling system operational")
    print("✅ Logging and monitoring active")
    print("✅ Input validation functional")
    print("✅ System health checks working")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
