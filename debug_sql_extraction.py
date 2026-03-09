#!/usr/bin/env python3
"""
Debug ChatGPT-Level SQL Session Architecture - Focus on Query Extraction

This tests just the core session architecture without LLM dependency.
"""

from app.services.session_state_manager import SessionStateManager
from app.services.followup_manager import FollowUpAnalyzer

def test_query_extraction():
    """Test previous query extraction from SQL conversation history."""
    
    print("🧪 Testing SQL Conversation History & Query Extraction")
    print("=" * 60)
    
    # Create a session manager
    session_manager = SessionStateManager("test_session", "test_user")
    
    # Simulate first query
    print("\n1️⃣ Adding First Query to Session")
    session_manager.add_sql_conversation_entry(
        user_query="How many male and female clients have birthdays in Mar?",
        generated_sql="SELECT COUNT(*) FROM genai.customers WHERE gender IN ('M', 'F') AND EXTRACT(MONTH FROM dob) = 3",
        result_count=2850,
        success=True
    )
    
    # Get conversation history
    sql_history = session_manager.get_sql_conversation_history()
    print(f"📝 SQL History:\n{sql_history}")
    print(f"📏 History Length: {len(sql_history)} characters")
    
    # Test query extraction
    print(f"\n2️⃣ Testing Previous Query Extraction")
    analyzer = FollowUpAnalyzer()
    previous_query = analyzer._extract_previous_query(sql_history)
    print(f"🔍 Extracted Previous Query: '{previous_query}'")
    
    # Test with multiple queries
    print(f"\n3️⃣ Adding Second Query")
    session_manager.add_sql_conversation_entry(
        user_query="list all those clients",
        generated_sql="SELECT * FROM genai.customers WHERE gender IN ('M', 'F') AND EXTRACT(MONTH FROM dob) = 3",
        result_count=2850,
        success=True
    )
    
    updated_history = session_manager.get_sql_conversation_history()
    print(f"📝 Updated SQL History:\n{updated_history}")
    
    # Extract previous query from updated history
    previous_query_2 = analyzer._extract_previous_query(updated_history)
    print(f"🔍 Previous Query from Updated History: '{previous_query_2}'")
    
    # Manual test of the conversation format
    print(f"\n4️⃣ Manual Format Test")
    test_history = """USER: How many male and female clients have birthdays in Mar?
SQL: SELECT COUNT(*) FROM genai.customers WHERE gender IN ('M', 'F') AND EXTRACT(MONTH FROM dob) = 3 | Results: 2850 rows"""
    
    manual_previous = analyzer._extract_previous_query(test_history)
    print(f"🧪 Manual Test Result: '{manual_previous}'")
    
    print(f"\n✅ Query Extraction Test Results:")
    print(f"   Session History Working: {'✓' if sql_history else '✗'}")
    print(f"   Query Extraction Working: {'✓' if previous_query else '✗'}")
    print(f"   Expected: 'How many male and female clients have birthdays in Mar?'")
    print(f"   Actual: '{previous_query}'")

if __name__ == "__main__":
    test_query_extraction()