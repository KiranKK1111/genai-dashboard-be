#!/usr/bin/env python3
"""
Debug script to test if DataQueryResponse now includes debug metadata
for proper SQL extraction in ChatGPT-level session architecture.
"""

import asyncio
import sys
import json
from app import schemas

def test_data_query_response_debug():
    """Test if DataQueryResponse can include debug metadata"""
    
    print("🔍 Testing DataQueryResponse with debug metadata...")
    
    # Test creating DataQueryResponse with debug field
    try:
        response = schemas.DataQueryResponse(
            intent="test query",
            confidence=0.95,
            message="Test response",
            datasets=[],
            visualizations=[],
            layout=schemas.Layout(type="single", arrangement="single"),
            related_queries=[],
            debug={
                "normalized_user_request": "test query",
                "sql_executed": "SELECT * FROM customers WHERE id = 1",
                "row_count": 1,
                "complexity": "simple",
                "response_type": "test",
            },
        )
        
        print("✅ DataQueryResponse created successfully with debug field")
        
        # Test if debug field is accessible
        if hasattr(response, 'debug'):
            print("✅ debug attribute exists")
            print(f"   debug.sql_executed: {response.debug.get('sql_executed')}")
            print(f"   debug.row_count: {response.debug.get('row_count')}")
            
            # Test SQL extraction logic (similar to session_query_handler.py)
            debug_info = response.debug
            if isinstance(debug_info, dict):
                generated_sql = debug_info.get('sql_executed', None)
                result_count = debug_info.get('row_count', 0)
                
                print(f"   Extracted SQL: {generated_sql}")
                print(f"   Extracted count: {result_count}")
                
                if generated_sql:
                    print("✅ SQL extraction would work!")
                    return True
                else:
                    print("❌ SQL extraction would fail - no sql_executed")
                    return False
            else:
                print("❌ debug_info is not a dict")
                return False
        else:
            print("❌ debug attribute missing")
            return False
            
    except Exception as e:
        print(f"❌ Error creating DataQueryResponse: {e}")
        return False

def test_response_wrapper_extraction():
    """Test the complete response wrapper extraction logic"""
    
    print("\n🔍 Testing ResponseWrapper extraction logic...")
    
    try:
        # Create a DataQueryResponse with debug metadata
        data_response = schemas.DataQueryResponse(
            intent="test query",
            confidence=0.95,
            message="Test response",
            datasets=[],
            visualizations=[],
            layout=schemas.Layout(type="single", arrangement="single"),
            related_queries=[],
            debug={
                "sql_executed": "SELECT COUNT(*) FROM customers WHERE EXTRACT(MONTH FROM dob) = 3",
                "row_count": 2850,
                "complexity": "medium",
            },
        )
        
        # Wrap it in ResponseWrapper
        response_wrapper = schemas.ResponseWrapper(
            success=True,
            response=data_response,
            timestamp=1234567890,
            original_query="test query"
        )
        
        print("✅ ResponseWrapper created successfully")
        
        # Test extraction logic (from session_query_handler.py)
        if hasattr(response_wrapper.response, 'debug'):
            debug_info = response_wrapper.response.debug
            print(f"   Found debug info: {debug_info is not None}")
            
            if isinstance(debug_info, dict):
                generated_sql = debug_info.get('sql_executed', None)
                result_count = debug_info.get('row_count', 0)
                
                print(f"   SQL: {generated_sql}")
                print(f"   Count: {result_count}")
                
                if generated_sql:
                    print("✅ Complete extraction pipeline would work!")
                    return True
                    
        print("❌ Extraction failed")
        return False
        
    except Exception as e:
        print(f"❌ Error in extraction test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG: DataQueryResponse Debug Metadata Test")
    print("=" * 60)
    
    test1 = test_data_query_response_debug()
    test2 = test_response_wrapper_extraction()
    
    print(f"\n📊 Results:")
    print(f"   DataQueryResponse debug: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Complete extraction: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 ChatGPT-level session architecture should now work!")
        sys.exit(0)
    else:
        print("\n❌ Issues detected - SQL conversation storage may still fail")
        sys.exit(1)