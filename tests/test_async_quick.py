"""Quick test for async search functionality."""

import asyncio
import time
import httpx


async def test_async_mode():
    """Test async mode returns immediate response."""
    print("Testing ASYNC MODE (should return immediately)...")

    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {
            "query": "What is Python?",
            "mode": "speed",
            "async_mode": True,  # Background processing
        }

        start = time.time()
        response = await client.post(
            "http://localhost:8001/api/v1/search",
            json=payload
        )
        elapsed = time.time() - start

        print(f"✅ Response in {elapsed:.2f}s (status: {response.status_code})")

        if response.status_code == 200:
            result = response.json()
            session_id = result.get("session_id")
            status = result.get("status")
            print(f"   Session ID: {session_id}")
            print(f"   Status: {status}")

            if status == "pending":
                print("✅ ASYNC MODE WORKING - Job queued!")

                # Poll once after 10 seconds
                print("\nWaiting 10s then polling...")
                await asyncio.sleep(10)

                poll_response = await client.get(
                    f"http://localhost:8001/api/v1/search/status/{session_id}"
                )

                if poll_response.status_code == 200:
                    poll_result = poll_response.json()
                    poll_status = poll_result.get("status")
                    print(f"   Poll Status: {poll_status}")

                    if poll_status == "completed":
                        print(f"   Answer: {poll_result.get('answer', '')[:100]}...")
                        print("✅ JOB COMPLETED!")
                    elif poll_status in ["pending", "processing"]:
                        print("   Still processing (expected for slower queries)")
                else:
                    print(f"   ❌ Poll failed: {poll_response.text}")
        else:
            print(f"   ❌ Error: {response.text}")


if __name__ == "__main__":
    asyncio.run(test_async_mode())
