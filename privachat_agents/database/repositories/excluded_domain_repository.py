"""Repository for managing excluded domains in web crawling.

This module provides database operations for domain exclusion patterns,
supporting exact matches, wildcard patterns, and regex patterns.
"""

import fnmatch
import re
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ExcludedDomain


class ExcludedDomainRepository:
    r"""Repository for excluded domain operations with pattern matching and caching.
    
    Supports three pattern types:
    - exact: Exact domain match (e.g., 'youtube.com')
    - wildcard: Shell-style wildcards (e.g., '*.youtube.*' matches 'www.youtube.com')
    - regex: Regular expression patterns (e.g., r'^.*\.youtube\..*$')
    
    Features:
    - In-memory caching for performance
    - Automatic cache invalidation on changes
    - Case-insensitive domain matching
    - Invalid URL handling
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize repository with database session.
        
        Args:
            db: Async SQLAlchemy database session
        """
        self.db = db
        self._cache: dict[str, list[ExcludedDomain]] | None = None

    async def _get_cached_exclusions(self) -> list[ExcludedDomain]:
        """Get cached exclusions or fetch from database.
        
        Returns:
            List of active excluded domains
        """
        if self._cache is None:
            self._cache = {}
            exclusions = await self.get_all_active()
            self._cache["exclusions"] = exclusions
            return exclusions
        return self._cache.get("exclusions", [])

    def _invalidate_cache(self) -> None:
        """Invalidate the in-memory cache."""
        self._cache = None

    async def get_all_active(self) -> list[ExcludedDomain]:
        """Retrieve all active excluded domains.
        
        Returns:
            List of active ExcludedDomain objects
            
        Example:
            >>> repo = ExcludedDomainRepository(db)
            >>> exclusions = await repo.get_all_active()
            >>> for exc in exclusions:
            ...     print(f"{exc.pattern_type}: {exc.domain_pattern}")
        """
        result = await self.db.execute(
            select(ExcludedDomain)
            .where(ExcludedDomain.is_active == True)
            .order_by(ExcludedDomain.created_at)
        )
        return list(result.scalars().all())

    async def is_domain_excluded(self, url: str) -> bool:
        """Check if a URL's domain matches any exclusion pattern.
        
        Args:
            url: Full URL to check (e.g., 'https://www.youtube.com/watch?v=123')
            
        Returns:
            True if domain is excluded, False otherwise
            
        Raises:
            ValueError: If URL is invalid or empty
            
        Example:
            >>> await repo.add_exclusion("*.youtube.*", "wildcard")
            >>> await repo.is_domain_excluded("https://www.youtube.com/watch")
            True
            >>> await repo.is_domain_excluded("https://example.com")
            False
        """
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")

        try:
            parsed = urlparse(url.strip())
            domain = parsed.netloc.lower()
            
            if not domain:
                raise ValueError(f"Invalid URL: {url}")
                
        except Exception as e:
            raise ValueError(f"Failed to parse URL '{url}': {str(e)}")

        exclusions = await self._get_cached_exclusions()

        for exclusion in exclusions:
            pattern = exclusion.domain_pattern.lower()
            pattern_type = exclusion.pattern_type

            if pattern_type == "exact":
                if domain == pattern:
                    return True

            elif pattern_type == "wildcard":
                # fnmatch for shell-style wildcards: *.youtube.* matches www.youtube.com
                if fnmatch.fnmatch(domain, pattern):
                    return True

            elif pattern_type == "regex":
                try:
                    if re.match(pattern, domain):
                        return True
                except re.error:
                    # Invalid regex pattern, skip
                    continue

        return False

    async def add_exclusion(
        self,
        domain_pattern: str,
        pattern_type: str = "exact",
        reason: str | None = None,
        created_by: str | None = None,
    ) -> ExcludedDomain:
        """Add a new domain exclusion pattern.
        
        Args:
            domain_pattern: Pattern to match (e.g., '*.youtube.*', 'youtube.com')
            pattern_type: Type of pattern - 'exact', 'wildcard', or 'regex'
            reason: Optional reason for exclusion
            created_by: Optional user who created the exclusion
            
        Returns:
            Created ExcludedDomain object
            
        Raises:
            ValueError: If pattern_type is invalid
            
        Example:
            >>> exc = await repo.add_exclusion(
            ...     "*.youtube.*",
            ...     pattern_type="wildcard",
            ...     reason="Avoid video content"
            ... )
            >>> print(exc.domain_pattern)
            *.youtube.*
        """
        if pattern_type not in ("exact", "wildcard", "regex"):
            raise ValueError(f"Invalid pattern_type: {pattern_type}")

        exclusion = ExcludedDomain(
            id=uuid.uuid4(),
            domain_pattern=domain_pattern,
            pattern_type=pattern_type,
            reason=reason,
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=created_by,
        )

        self.db.add(exclusion)
        await self.db.commit()
        await self.db.refresh(exclusion)
        
        self._invalidate_cache()
        
        return exclusion

    async def remove_exclusion(self, domain_pattern: str) -> bool:
        """Soft delete an exclusion by setting is_active to False.
        
        Args:
            domain_pattern: Exact pattern to remove
            
        Returns:
            True if removed, False if not found
            
        Example:
            >>> await repo.remove_exclusion("*.youtube.*")
            True
            >>> await repo.is_domain_excluded("https://www.youtube.com/watch")
            False
        """
        result = await self.db.execute(
            update(ExcludedDomain)
            .where(ExcludedDomain.domain_pattern == domain_pattern)
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        await self.db.commit()
        
        if result.rowcount > 0:
            self._invalidate_cache()
            return True
        
        return False

    async def get_by_pattern(self, domain_pattern: str) -> ExcludedDomain | None:
        """Get exclusion by exact domain pattern.
        
        Args:
            domain_pattern: Exact pattern to find
            
        Returns:
            ExcludedDomain object if found, None otherwise
        """
        result = await self.db.execute(
            select(ExcludedDomain).where(ExcludedDomain.domain_pattern == domain_pattern)
        )
        return result.scalar_one_or_none()
