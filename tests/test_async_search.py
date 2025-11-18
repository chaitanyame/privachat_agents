"""Test script for async search functionality with Redis job state.

Tests both sync and async modes of the /v1/search endpoint.
"""

import asyncio
import time
from typing import Any

import httpx


API_BASE_URL = "http://localhost:8001"  # API is exposed on port 8001 (see docker-compose.yml)


async def test_sync_search() -> None:
    """Test synchronous search (default behavior)."""
    print("\n" + "=" * 60)
    print("TEST 1: Synchronous Search (async_mode=false)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "query": "What is Pydantic AI?",
            "mode": "speed",
            "async_mode": False,  # Default: synchronous
        }

        print(f"\n📤 Sending sync request...")
        print(f"   Query: {payload['query']}")

        start = time.time()
        response = await client.post(f"{API_BASE_URL}/api/v1/search", json=payload)
        elapsed = time.time() - start

        print(f"\n✅ Response received in {elapsed:.2f}s")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Status: {result.get('status', 'completed')}")
            print(f"   Answer: {result.get('answer', '')[:100]}...")
            print(f"   Sources: {len(result.get('sources', []))}")
            print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")

            # Check citation mapping
            citation_mapping = result.get("citation_mapping")
            if citation_mapping:
                print(f"   Citation Mapping: {len(citation_mapping)} citations")
                for cm in citation_mapping[:3]:  # Show first 3
                    print(f"      [{cm['citation_number']}] → {cm['domain']} (mentions: {cm['mention_count']})")
        else:
            print(f"   Error: {response.text}")


async def test_async_search() -> None:
    """Test asynchronous search with background processing."""
    print("\n" + "=" * 60)
    print("TEST 2: Asynchronous Search (async_mode=true)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "query": "How do AI agents work?",
            "mode": "balanced",
            "async_mode": True,  # Enable background processing
        }

        print(f"\n📤 Sending async request...")
        print(f"   Query: {payload['query']}")

        start = time.time()
        response = await client.post(f"{API_BASE_URL}/api/v1/search", json=payload)
        immediate_response_time = time.time() - start

        print(f"\n✅ Immediate response received in {immediate_response_time:.2f}s")
        print(f"   Status: {response.status_code}")

        if response.status_code != 200:
            print(f"   Error: {response.text}")
            return

        result = response.json()
        session_id = result.get("session_id")
        print(f"   Session ID: {session_id}")
        print(f"   Status: {result.get('status')}")

        # Poll for completion
        print(f"\n🔄 Polling for completion...")
        max_polls = 60  # Poll for up to 60 seconds
        poll_interval = 2  # Poll every 2 seconds

        for poll_count in range(1, max_polls + 1):
            await asyncio.sleep(poll_interval)

            poll_response = await client.get(
                f"{API_BASE_URL}/api/v1/search/status/{session_id}"
            )

            if poll_response.status_code != 200:
                print(f"   ❌ Poll failed: {poll_response.status_code}")
                break

            poll_result = poll_response.json()
            poll_status = poll_result.get("status")

            print(f"   Poll #{poll_count}: status={poll_status}")

            if poll_status == "completed":
                total_time = time.time() - start
                print(f"\n✅ Job completed in {total_time:.2f}s")
                print(f"   Answer: {poll_result.get('answer', '')[:100]}...")
                print(f"   Sources: {len(poll_result.get('sources', []))}")
                print(f"   Execution Time: {poll_result.get('execution_time', 0):.2f}s")

                # Check citation mapping
                citation_mapping = poll_result.get("citation_mapping")
                if citation_mapping:
                    print(f"   Citation Mapping: {len(citation_mapping)} citations")
                    for cm in citation_mapping[:3]:
                        print(f"      [{cm['citation_number']}] → {cm['domain']} (mentions: {cm['mention_count']})")
                break

            elif poll_status == "failed":
                print(f"   ❌ Job failed: {poll_result.get('error')}")
                break

            elif poll_status in ["pending", "processing"]:
                continue  # Keep polling

        else:
            print(f"   ⏰ Timeout after {max_polls * poll_interval}s")


async def test_redis_ttl() -> None:
    """Test Redis TTL (24 hours)."""
    print("\n" + "=" * 60)
    print("TEST 3: Redis TTL Verification (24 hours)")
    print("=" * 60)

    # Start async job
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "query": "What is Redis?",
            "mode": "speed",
            "async_mode": True,
        }

        print(f"\n📤 Starting async job for TTL test...")
        response = await client.post(f"{API_BASE_URL}/api/v1/search", json=payload)

        if response.status_code != 200:
            print(f"   ❌ Failed to start job: {response.text}")
            return

        result = response.json()
        session_id = result.get("session_id")
        print(f"   Session ID: {session_id}")

        # Wait for completion
        await asyncio.sleep(10)

        # Check TTL via docker exec
        print(f"\n🔍 Checking Redis TTL...")
        import subprocess

        ttl_cmd = f'docker exec redis redis-cli TTL "search_job:{session_id}"'
        ttl_result = subprocess.run(
            ttl_cmd, shell=True, capture_output=True, text=True
        )

        if ttl_result.returncode == 0:
            ttl_seconds = int(ttl_result.stdout.strip())
            ttl_hours = ttl_seconds / 3600

            print(f"   TTL: {ttl_seconds}s ({ttl_hours:.2f} hours)")

            if 23 <= ttl_hours <= 24:
                print(f"   ✅ TTL correctly set to ~24 hours")
            else:
                print(f"   ⚠️  TTL unexpected: expected ~24 hours, got {ttl_hours:.2f} hours")
        else:
            print(f"   ❌ Failed to check TTL: {ttl_result.stderr}")


async def main() -> None:
    """Run all tests."""
    print("\n🧪 Testing Async Search Functionality with Redis")
    print(f"API: {API_BASE_URL}")

    try:
        # Test 1: Sync search (default behavior)
        await test_sync_search()

        # Test 2: Async search with polling
        await test_async_search()

        # Test 3: Redis TTL verification
        await test_redis_ttl()

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except Exception as e:
        import traceback

        print(f"\n❌ Test failed: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
