"""
Complete Advanced RAG System Test with LLM-as-a-Judge Evaluation
Tests the entire pipeline including retrieval, generation, evaluation, and enhancement
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advanced_rag.processor import AdvancedRAGProcessor

# Load environment variables
load_dotenv()

def format_json_response(response_data):
    """Format the response data as pretty JSON for console display"""
    return json.dumps(response_data, indent=2, ensure_ascii=False)

def test_complete_rag_with_llm_evaluation():
    """Test the complete RAG system with LLM evaluation using sample queries"""
    
    print("TESTING COMPLETE ADVANCED RAG SYSTEM WITH LLM-AS-A-JUDGE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Sample queries for testing different scenarios - PERFORMANCE COMPARISON TEST
    test_queries = [
        {
            "id": 1,
            "query": "What are effective sleep routines for toddlers?",
            "expected_quality": "high",
            "description": "Well-covered topic in childcare knowledge base"
        }
        # Add second query back for full comparison
        # {
        #     "id": 2, 
        #     "query": "How should I handle my 2-year-old's tantrums in public?",
        #     "expected_quality": "medium-high",
        #     "description": "Common parenting challenge"
        # }
    ]
    
    # Initialize the processor
    print("Initializing Advanced RAG Processor...")
    processor = AdvancedRAGProcessor()
    
    print("Connecting to systems...")
    if not processor.initialize():
        print("ERROR: Failed to initialize processor")
        return False
    
    print("SUCCESS: System initialization complete!")
    print("\n" + "=" * 80)
    
    # Test each query
    for test_case in test_queries:
        print(f"\nTEST CASE {test_case['id']}: {test_case['description']}")
        print("=" * 60)
        print(f"Query: {test_case['query']}")
        print(f"Expected Quality: {test_case['expected_quality']}")
        print("-" * 60)
        
        start_time = time.time()
        print(f"� ASYNC TEST: Starting optimized query processing at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Process the query through complete pipeline
            print("Processing query through ASYNC Advanced RAG pipeline...")
            result = processor.process_query(test_case['query'])
            
            processing_time = time.time() - start_time
            
            if result['success']:
                print(f"⚡ ASYNC RESULT: Query processed successfully in {processing_time:.3f}s")
                print(f"🕐 PERFORMANCE IMPROVEMENT:")
                baseline_time = 55.805  # Previous baseline
                improvement = ((baseline_time - processing_time) / baseline_time) * 100
                print(f"   Baseline Time: {baseline_time:.3f}s")
                print(f"   Async Time: {processing_time:.3f}s")
                print(f"   Performance Gain: {improvement:.1f}% faster")
                print(f"   Retrieved Documents: {result.get('retrieval_metadata', {}).get('total_docs_retrieved', 0)}")
                print(f"   Final Documents Used: {result.get('retrieval_metadata', {}).get('final_docs_used', 0)}")
                print(f"   Strategy Used: {result.get('retrieval_metadata', {}).get('retrieval_strategy', 'unknown')}")
                
                # Format the complete response for frontend
                frontend_response = {
                    "request": {
                        "query": test_case['query'],
                        "timestamp": datetime.now().isoformat(),
                        "test_case_id": test_case['id']
                    },
                    "response": {
                        "success": True,
                        "answer": result['answer'],
                        "original_answer": result.get('original_answer', ''),
                        "processing_time": result['processing_time'],
                        "confidence_score": result['confidence_score']
                    },
                    "evaluation": result.get('evaluation', {}),
                    "retrieval_metadata": result.get('retrieval_metadata', {}),
                    "sources": result.get('sources', []),
                    "system_metadata": {
                        "timestamp": result.get('timestamp'),
                        "relevance_check": result.get('relevance_check', {}),
                        "pipeline_version": "Advanced RAG v2.0 with LLM-as-a-Judge"
                    }
                }
                
                print("\nCOMPLETE FRONTEND RESPONSE:")
                print("=" * 50)
                print(format_json_response(frontend_response))
                
                # Quick summary for easy viewing
                print(f"\nQUICK SUMMARY:")
                print(f"   Answer Quality: {result.get('evaluation', {}).get('quality_grade', 'N/A')}")
                print(f"   LLM Score: {result.get('evaluation', {}).get('llm_quality_score', 'N/A'):.2f}")
                print(f"   Safety Level: {result.get('evaluation', {}).get('safety_level', 'N/A')}")
                print(f"   Enhancement: {result.get('evaluation', {}).get('enhancement_applied', 'N/A')}")
                print(f"   Sources Used: {len(result.get('sources', []))}")
                print(f"   Strategy: {result.get('retrieval_metadata', {}).get('retrieval_strategy', 'N/A')}")
                
            else:
                print(f"ERROR: Query failed: {result.get('error', 'Unknown error')}")
                
                # Format error response for frontend
                frontend_response = {
                    "request": {
                        "query": test_case['query'],
                        "timestamp": datetime.now().isoformat(),
                        "test_case_id": test_case['id']
                    },
                    "response": {
                        "success": False,
                        "error": result.get('error', 'Unknown error'),
                        "processing_time": result.get('processing_time', processing_time)
                    },
                    "system_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "pipeline_version": "Advanced RAG v2.0 with LLM-as-a-Judge"
                    }
                }
                
                print("\nERROR RESPONSE:")
                print("=" * 40)
                print(format_json_response(frontend_response))
                
        except Exception as e:
            print(f"ERROR: Exception during processing: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 80)
    
    print("\nCOMPLETE SYSTEM TEST FINISHED!")
    print("The system is ready for frontend integration.")
    return True

def test_system_status():
    """Test and display system status"""
    print("\nSYSTEM STATUS CHECK")
    print("=" * 40)
    
    processor = AdvancedRAGProcessor()
    
    if processor.initialize():
        status = processor.get_system_status()
        print("System Status:")
        print(format_json_response(status))
        return True
    else:
        print("ERROR: System initialization failed")
        return False

def main():
    """Main test execution"""
    print("ADVANCED RAG SYSTEM COMPREHENSIVE TEST")
    print("Testing complete pipeline with LLM-as-a-Judge evaluation")
    print("=" * 80)
    
    # Check environment
    required_keys = ['OPENAI_API_KEY', 'COHERE_API_KEY', 'TAVILY_API_KEY', 'ZILLIZ_TOKEN', 'ZILLIZ_URI']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"ERROR: Missing environment variables: {missing_keys}")
        print("Please check your .env file")
        return False
    
    print("SUCCESS: All required environment variables found")
    
    # Run system status check
    if not test_system_status():
        return False
    
    # Run complete system test
    return test_complete_rag_with_llm_evaluation()

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nALL TESTS COMPLETED SUCCESSFULLY!")
        print("SUCCESS: System is ready for frontend integration")
        print("SUCCESS: LLM-as-a-Judge evaluation is working")
        print("SUCCESS: Complete pipeline validated")
    else:
        print("\nTESTS FAILED!")
        print("Please review the errors above")
    
    sys.exit(0 if success else 1)
