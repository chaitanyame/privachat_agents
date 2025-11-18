"""Comprehensive test for both sync and async modes with result verification."""

import asyncio
import time
import httpx


async def test_sync_mode():
    """Test sync mode returns complete result."""
    print("\n" + "="*60)
    print("TEST 1: SYNC MODE (Default Behavior)")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        payload = {
            "query": "What is Python?",
            "mode": "speed",  # Faster mode for testing
            "async_mode": False,
        }
        
        print(f"\n📤 Request: {payload['query']}")
        print("   Waiting for complete result...")
        
        start = time.time()
        try:
            response = await client.post(
                "http://localhost:8001/api/v1/search",
                json=payload
            )
            elapsed = time.time() - start
            
            print(f"\n✅ Response in {elapsed:.2f}s")
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Verify sync mode response
                print(f"\n📊 SYNC MODE RESULTS:")
                print(f"   Session ID: {result.get('session_id')}")
                print(f"   Status: {result.get('status', 'completed (implicit)')}")
                print(f"   Query: {result.get('query')}")
                
                # Check answer
                answer = result.get('answer')
                if answer:
                    print(f"   ✅ Answer: {answer[:150]}...")
                    print(f"   Answer Length: {len(answer)} chars")
                else:
                    print(f"   ❌ No answer returned!")
                    return False
                
                # Check sources
                sources = result.get('sources', [])
                print(f"   ✅ Sources: {len(sources)} sources")
                if sources:
                    for i, src in enumerate(sources[:3], 1):
                        print(f"      [{i}] {src.get('title', 'N/A')[:50]}...")
                
                # Check citation mapping
                citations = result.get('citation_mapping', [])
                if citations:
                    print(f"   ✅ Citation Mapping: {len(citations)} citations")
                    for cm in citations[:3]:
                        print(f"      [{cm['citation_number']}] → {cm['domain']}")
                
                # Check metadata
                print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Model: {result.get('model_used', 'N/A')}")
                
                print("\n✅ SYNC MODE: PASSED - Complete result returned")
                return True
                
            else:
                print(f"   ❌ Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False


async def test_async_mode():
    """Test async mode with polling until completion."""
    print("\n" + "="*60)
    print("TEST 2: ASYNC MODE (Background Processing)")
    print("="*60)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {
            "query": "What is FastAPI?",
            "mode": "speed",
            "async_mode": True,
        }
        
        print(f"\n📤 Request: {payload['query']}")
        print("   Expecting immediate response...")
        
        # Step 1: Start async job
        start = time.time()
        try:
            response = await client.post(
                "http://localhost:8001/api/v1/search",
                json=payload
            )
            immediate_time = time.time() - start
            
            print(f"\n✅ Immediate response in {immediate_time:.2f}s")
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"   ❌ Error: {response.text}")
                return False
            
            result = response.json()
            session_id = result.get('session_id')
            status = result.get('status')
            
            print(f"\n📊 ASYNC MODE - INITIAL RESPONSE:")
            print(f"   Session ID: {session_id}")
            print(f"   Status: {status}")
            
            if status != "pending":
                print(f"   ⚠️  Expected 'pending', got '{status}'")
            
            if immediate_time > 2.0:
                print(f"   ⚠️  Response took {immediate_time:.2f}s (expected <2s)")
            else:
                print(f"   ✅ Fast response confirmed (<2s)")
            
            # Step 2: Poll for completion
            print(f"\n🔄 Polling for completion (max 90s)...")
            poll_count = 0
            max_polls = 45  # 90 seconds / 2 second intervals
            
            while poll_count < max_polls:
                poll_count += 1
                await asyncio.sleep(2)
                
                poll_response = await client.get(
                    f"http://localhost:8001/api/v1/search/status/{session_id}"
                )
                
                if poll_response.status_code != 200:
                    print(f"   ❌ Poll #{poll_count} failed: {poll_response.status_code}")
                    return False
                
                poll_result = poll_response.json()
                poll_status = poll_result.get('status')
                
                print(f"   Poll #{poll_count} ({poll_count*2}s): status={poll_status}")
                
                if poll_status == "completed":
                    total_time = time.time() - start
                    print(f"\n✅ Job completed in {total_time:.2f}s total")
                    
                    # Verify complete result
                    print(f"\n📊 ASYNC MODE - FINAL RESULT:")
                    
                    # Check answer
                    answer = poll_result.get('answer')
                    if answer:
                        print(f"   ✅ Answer: {answer[:150]}...")
                        print(f"   Answer Length: {len(answer)} chars")
                    else:
                        print(f"   ❌ No answer in completed result!")
                        return False
                    
                    # Check sources
                    sources = poll_result.get('sources', [])
                    print(f"   ✅ Sources: {len(sources)} sources")
                    if sources:
                        for i, src in enumerate(sources[:3], 1):
                            print(f"      [{i}] {src.get('title', 'N/A')[:50]}...")
                    
                    # Check citation mapping
                    citations = poll_result.get('citation_mapping', [])
                    if citations:
                        print(f"   ✅ Citation Mapping: {len(citations)} citations")
                        for cm in citations[:3]:
                            print(f"      [{cm['citation_number']}] → {cm['domain']}")
                    
                    # Check metadata
                    print(f"   Execution Time: {poll_result.get('execution_time', 0):.2f}s")
                    print(f"   Confidence: {poll_result.get('confidence', 0):.2f}")
                    print(f"   Model: {poll_result.get('model_used', 'N/A')}")
                    
                    print("\n✅ ASYNC MODE: PASSED - Complete result via polling")
                    return True
                
                elif poll_status == "failed":
                    error = poll_result.get('error', 'Unknown error')
                    print(f"   ❌ Job failed: {error}")
                    return False
                
                # Continue polling for pending/processing
            
            print(f"   ⏰ Timeout after {max_polls * 2}s")
            return False
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run comprehensive tests."""
    print("\n" + "🧪 COMPREHENSIVE MODE TESTING")
    print("Testing both sync and async modes with result verification\n")
    
    results = {}
    
    # Test sync mode
    results['sync'] = await test_sync_mode()
    
    # Test async mode
    results['async'] = await test_async_mode()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Sync Mode:  {'✅ PASSED' if results['sync'] else '❌ FAILED'}")
    print(f"Async Mode: {'✅ PASSED' if results['async'] else '❌ FAILED'}")
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED - Both modes working correctly!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Check output above")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
