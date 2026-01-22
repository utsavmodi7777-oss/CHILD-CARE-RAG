"""
Enhanced RAG Test - Single Query Version
Modified to test just one query to verify document ID fix
"""

import sys
import os
from pathlib import Path
import json
import time

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def test_enhanced_rag_single():
    """Test enhanced RAG system with single query"""
    print("Enhanced Multi-Tier RAG System Test - Single Query")
    print("=" * 60)
    
    try:
        from advanced_rag.processor import AdvancedRAGProcessor
        
        # Initialize processor
        print("Initializing Advanced RAG Processor...")
        processor = AdvancedRAGProcessor()
        
        # Initialize all components
        print("Setting up database connections and models...")
        if not processor.initialize():
            print("Failed to initialize processor")
            return False
        print("System initialized successfully!")
        
        # Single test query
        test_query = "Can labeling my child negatively affect their behavior?"
        expected_tier = "high"
        
        print(f"\nTest Query: {test_query}")
        print("Expected Tier: high (core childcare topic)")
        print("-" * 60)
        
        # Process query
        start_time = time.time()
        result = processor.process_query(test_query)
        processing_time = time.time() - start_time
        
        if result.get('success', False):
            print(f"SUCCESS: Query processed in {processing_time:.2f}s")
            
            # Extract confidence information
            retrieval_metadata = result.get('retrieval_metadata', {})
            confidence_score = retrieval_metadata.get('confidence_score', 0)
            confidence_tier = retrieval_metadata.get('confidence_tier', 'unknown')
            strategy_used = retrieval_metadata.get('retrieval_strategy', 'unknown')
            
            print(f"Confidence Score: {confidence_score:.3f}")
            print(f"Confidence Tier: {confidence_tier}")
            print(f"Strategy Used: {strategy_used}")
            print(f"Expected Tier: {expected_tier}")
            
            # Check if tier matches expectation
            tier_match = confidence_tier == expected_tier
            print(f"Tier Match: {'✓' if tier_match else '✗'}")
            
            # Show answer preview
            answer = result.get('answer', '')
            if answer:
                # preview = answer[:150] + "..." if len(answer) > 150 else answer
                # print(f"\nAnswer Preview: {preview}")

                print(f"\n\nComplete result: {result}")

            # Evaluate document ID fix effectiveness
            if confidence_score > 0.7:
                print("\nHigh confidence indicates document ID fix is working!")
                fix_status = "WORKING"
            elif confidence_score > 0.5:
                print("\nModerate confidence suggests improvement from document ID fix")
                fix_status = "IMPROVED"
            else:
                print("\nDocument ID fix may need further optimization")
                fix_status = "NEEDS_WORK"
            
            # Save results
            test_results = {
                "query": test_query,
                "expected_tier": expected_tier,
                "actual_tier": confidence_tier,
                "confidence_score": confidence_score,
                "strategy_used": strategy_used,
                "processing_time": processing_time,
                "tier_match": tier_match,
                "answer_length": len(answer),
                "fix_status": fix_status,
                "docs_used": retrieval_metadata.get('num_documents', 0)
            }
            
            timestamp = int(time.time())
            results_file = project_root / "testing_files" / f"enhanced_rag_single_test_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            print(f"\nDetailed results saved to: {results_file.name}")
            
            return True, fix_status
            
        else:
            print("FAILED: Query processing failed")
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")
            return False, "FAILED"
            
    except Exception as e:
        print(f"TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False, "ERROR"

def main():
    print("Starting Enhanced RAG Test with Document ID Fix...")
    print()
    
    success, status = test_enhanced_rag_single()
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if success:
        if status == "WORKING":
            print("SUCCESS! Document ID fix is working excellently!")
            print("   Enhanced RAG system ready!")
        elif status == "IMPROVED":
            print("SUCCESS! Document ID fix shows clear improvement!")
            print("   Enhanced RAG system is functional!")
        else:
            print("PARTIAL SUCCESS. System working but may need optimization.")
    else:
        print("TEST FAILED. Investigation needed.")
    
    return success

if __name__ == "__main__":
    success = main()
