#!/usr/bin/env python3
"""
Example usage of the Insurance RAG AI system.

This script demonstrates how to use the RAG system both programmatically
and via the REST API for various insurance-related queries.
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.rag import generate_answer


def demo_programmatic_usage():
    """Demonstrate direct programmatic usage of the RAG system."""
    print("🔍 Insurance RAG AI - Programmatic Usage Demo\n")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Example queries covering different insurance types
    queries = [
        "What is the deductible for collision coverage in auto insurance?",
        "Does health insurance cover preventive care?", 
        "What is excluded from homeowners insurance coverage?",
        "How do I file a life insurance claim?",
        "What is covered under business liability insurance?",
        "What happens if I don't pay my insurance premium on time?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 60)
        
        try:
            # Measure response time
            start_time = time.time()
            result = generate_answer(query)
            response_time = time.time() - start_time
            
            # Display results
            print(f"Answer: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Cache Hit: {'Yes' if result['cache_hit'] else 'No'}")
            print(f"Response Time: {response_time:.2f}s")
            
        except Exception as e:
            print(f"Error: {e}")
        
        # Small delay between queries
        time.sleep(1)


def demo_api_usage():
    """Demonstrate REST API usage."""
    print("\n\nInsurance RAG AI - REST API Usage Demo\n")
    print("=" * 60)
    
    api_url = "http://localhost:8000"
    
    # Check if API is running
    try:
        health_response = requests.get(f"{api_url}/")
        if health_response.status_code != 200:
            print("API is not running. Please start with: make run")
            return
        print("API is running")
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Please start with: make run")
        return
    
    # Example API queries
    api_queries = [
        "What is covered under personal injury protection?",
        "How much does whole life insurance cost annually?",
        "What are the exclusions for business insurance?",
    ]
    
    for i, query in enumerate(api_queries, 1):
        print(f"\nAPI Query {i}: {query}")
        print("-" * 60)
        
        try:
            payload = {"question": query}
            response = requests.post(f"{api_url}/query", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Answer: {result['answer']}")
                print(f"Sources: {', '.join(result['sources'])}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Cache Hit: {'Yes' if result['cache_hit'] else 'No'}")
                print(f"Latency: {result['latency_seconds']:.3f}s")
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.5)


def demo_caching_behavior():
    """Demonstrate the caching system behavior."""
    print("\n\nCaching System Demo\n")
    print("=" * 60)
    
    test_query = "What is the maximum coverage limit for liability?"
    
    print(f"Test Query: {test_query}")
    print("\n1️⃣ First call (should miss cache):")
    
    # First call
    start_time = time.time()
    result1 = generate_answer(test_query)
    time1 = time.time() - start_time
    
    print(f"   Time: {time1:.3f}s")
    print(f"   Cache Hit: {'Yes' if result1['cache_hit'] else 'No'}")
    
    print("\n2️⃣ Second call (should hit cache):")
    
    # Second call (same query)
    start_time = time.time()
    result2 = generate_answer(test_query)
    time2 = time.time() - start_time
    
    print(f"   Time: {time2:.3f}s")
    print(f"   Cache Hit: {'Yes' if result2['cache_hit'] else 'No'}")
    print(f"   Speedup: {time1/time2:.1f}x faster")
    
    # Verify same answer
    if result1['answer'] == result2['answer']:
        print("   ✅ Cache returned identical answer")
    else:
        print("   ⚠️ Cache returned different answer")


def main():
    """Run all demos."""
    print("Insurance RAG AI - Comprehensive Demo")
    print("Testing both direct usage and caching behavior\n")
    
    # Check if vector index exists
    if not os.path.exists("vector_store/index.faiss"):
        print("Vector index not found. Please run: make setup")
        return
    
    # Programmatic demo
    demo_programmatic_usage()
    
    # Caching demo
    demo_caching_behavior()
    
    # API demo (only if user wants to test it)
    print(f"\n{'='*60}")
    print(" To test the REST API:")
    print("   1. Run: make run")
    print("   2. Run: python examples/demo.py --api-only")
    print("   3. Or visit: http://localhost:8000/docs")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--api-only":
        demo_api_usage()
    else:
        main()