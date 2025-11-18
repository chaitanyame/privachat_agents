"""Redis client for job state management with 24-hour TTL.

This module provides Redis-based storage for async search job results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from ..core.config import settings

logger = logging.getLogger(__name__)

# Job state TTL: 24 hours (86400 seconds)
JOB_TTL_SECONDS = 86400


class RedisClient:
    """Redis client for async job state management."""

    def __init__(self, redis_url: str | None = None):
        """Initialize Redis client.

        Args:
            redis_url: Redis connection URL (uses settings.REDIS_URL if not provided)
        """
        self.redis_url = redis_url or str(settings.REDIS_URL)
        self._client: aioredis.Redis | None = None
        logger.info(f"RedisClient initialized with URL: {self.redis_url}")

    async def _get_client(self) -> aioredis.Redis:
        """Get or create Redis client connection.

        Returns:
            Redis client instance

        Raises:
            RedisError: If connection fails
        """
        if self._client is None:
            try:
                self._client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._client.ping()
                logger.info("Redis connection established")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

        return self._client

    async def set_job_status(
        self,
        session_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Set job status in Redis with 24-hour TTL.

        Args:
            session_id: Unique session identifier
            status: Job status ('pending', 'processing', 'completed', 'failed')
            result: Search result data (for 'completed' status)
            error: Error message (for 'failed' status)

        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._get_client()

            job_data = {
                "status": status,
                "result": result,
                "error": error,
            }

            key = f"search_job:{session_id}"
            await client.set(
                key,
                json.dumps(job_data),
                ex=JOB_TTL_SECONDS,  # 24-hour expiration
            )

            logger.info(
                f"Job status updated: session_id={session_id}, status={status}, ttl={JOB_TTL_SECONDS}s"
            )
            return True

        except RedisError as e:
            logger.error(f"Failed to set job status in Redis: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting job status: {e}")
            return False

    async def get_job_status(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Get job status from Redis.

        Args:
            session_id: Unique session identifier

        Returns:
            Job data dict with status/result/error, or None if not found
        """
        try:
            client = await self._get_client()
            key = f"search_job:{session_id}"

            data = await client.get(key)
            if data is None:
                logger.warning(f"Job not found: session_id={session_id}")
                return None

            job_data = json.loads(data)
            logger.info(f"Job status retrieved: session_id={session_id}, status={job_data.get('status')}")
            return job_data

        except RedisError as e:
            logger.error(f"Failed to get job status from Redis: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode job data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting job status: {e}")
            return None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")
