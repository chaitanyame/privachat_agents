"""Unit tests for ExcludedDomainRepository."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from privachat_agents.database.repositories.excluded_domain_repository import (
    ExcludedDomainRepository,
)


@pytest.mark.asyncio
async def test_add_exclusion_exact(async_db_session: AsyncSession):
    """Test adding exact domain exclusion."""
    repo = ExcludedDomainRepository(async_db_session)
    
    exclusion = await repo.add_exclusion(
        domain_pattern="youtube.com",
        pattern_type="exact",
        reason="Video platform",
    )
    
    assert exclusion.domain_pattern == "youtube.com"
    assert exclusion.pattern_type == "exact"
    assert exclusion.is_active is True


@pytest.mark.asyncio
async def test_add_exclusion_wildcard(async_db_session: AsyncSession):
    """Test adding wildcard domain exclusion."""
    repo = ExcludedDomainRepository(async_db_session)
    
    exclusion = await repo.add_exclusion(
        domain_pattern="*.youtube.*",
        pattern_type="wildcard",
        reason="All YouTube domains",
    )
    
    assert exclusion.domain_pattern == "*.youtube.*"
    assert exclusion.pattern_type == "wildcard"


@pytest.mark.asyncio
async def test_is_domain_excluded_exact_match(async_db_session: AsyncSession):
    """Test exact domain matching."""
    repo = ExcludedDomainRepository(async_db_session)
    
    await repo.add_exclusion("youtube.com", "exact", "Test")
    
    is_excluded, reason = await repo.is_domain_excluded("https://youtube.com/watch?v=123")
    assert is_excluded is True
    assert reason == "Test"
    
    is_excluded, _ = await repo.is_domain_excluded("https://www.youtube.com/watch")
    assert is_excluded is False  # Exact match only
    
    is_excluded, _ = await repo.is_domain_excluded("https://google.com")
    assert is_excluded is False


@pytest.mark.asyncio
async def test_is_domain_excluded_wildcard(async_db_session: AsyncSession):
    """Test wildcard domain matching."""
    repo = ExcludedDomainRepository(async_db_session)
    
    # Wildcard pattern for all YouTube subdomains and TLDs
    await repo.add_exclusion("*.youtube.*", "wildcard", "All YouTube domains")
    
    # Should match all YouTube variants
    is_excluded, _ = await repo.is_domain_excluded("https://www.youtube.com/watch")
    assert is_excluded is True
    
    is_excluded, _ = await repo.is_domain_excluded("https://m.youtube.com/watch")
    assert is_excluded is True
    
    is_excluded, _ = await repo.is_domain_excluded("https://music.youtube.com/watch")
    assert is_excluded is True
    
    is_excluded, _ = await repo.is_domain_excluded("https://youtube.com/watch")
    assert is_excluded is False  # No subdomain, pattern requires one
    
    # Should not match non-YouTube domains
    is_excluded, _ = await repo.is_domain_excluded("https://youtu.be/abc")
    assert is_excluded is False


@pytest.mark.asyncio
async def test_is_domain_excluded_wildcard_youtu_be(async_db_session: AsyncSession):
    """Test wildcard matching for youtu.be short links."""
    repo = ExcludedDomainRepository(async_db_session)
    
    await repo.add_exclusion("youtu.be", "exact", "YouTube short links")
    
    is_excluded, _ = await repo.is_domain_excluded("https://youtu.be/abc123")
    assert is_excluded is True
    
    is_excluded, _ = await repo.is_domain_excluded("https://www.youtu.be/abc")
    assert is_excluded is False  # Has www subdomain


@pytest.mark.asyncio
async def test_is_domain_excluded_multiple_patterns(async_db_session: AsyncSession):
    """Test multiple exclusion patterns."""
    repo = ExcludedDomainRepository(async_db_session)
    
    # Add multiple exclusions
    await repo.add_exclusion("*.youtube.*", "wildcard", "YouTube with subdomains")
    await repo.add_exclusion("youtube.com", "exact", "YouTube main domain")
    await repo.add_exclusion("youtu.be", "exact", "YouTube short links")
    
    # All should be excluded
    assert (await repo.is_domain_excluded("https://www.youtube.com/watch"))[0] is True
    assert (await repo.is_domain_excluded("https://youtube.com/watch"))[0] is True
    assert (await repo.is_domain_excluded("https://youtu.be/abc"))[0] is True
    assert (await repo.is_domain_excluded("https://m.youtube.com/watch"))[0] is True


@pytest.mark.asyncio
async def test_remove_exclusion(async_db_session: AsyncSession):
    """Test removing domain exclusion."""
    repo = ExcludedDomainRepository(async_db_session)
    
    await repo.add_exclusion("test.com", "exact", "Test exclusion")
    
    # Verify it's excluded
    is_excluded, _ = await repo.is_domain_excluded("https://test.com/page")
    assert is_excluded is True
    
    # Remove it
    removed = await repo.remove_exclusion("test.com")
    assert removed is True
    
    # Should not be excluded anymore
    is_excluded, _ = await repo.is_domain_excluded("https://test.com/page")
    assert is_excluded is False
    
    # Removing non-existent returns False
    removed = await repo.remove_exclusion("nonexistent.com")
    assert removed is False


@pytest.mark.asyncio
async def test_get_all_active(async_db_session: AsyncSession):
    """Test getting all active exclusions."""
    repo = ExcludedDomainRepository(async_db_session)
    
    # Add some exclusions
    await repo.add_exclusion("domain1.com", "exact")
    await repo.add_exclusion("*.domain2.*", "wildcard")
    await repo.add_exclusion("domain3.com", "exact")
    
    # Get all active
    active = await repo.get_all_active()
    assert len(active) >= 3  # At least our 3, may have migration data
    
    patterns = [e.domain_pattern for e in active]
    assert "domain1.com" in patterns
    assert "*.domain2.*" in patterns
    assert "domain3.com" in patterns


@pytest.mark.asyncio
async def test_cache_invalidation(async_db_session: AsyncSession):
    """Test that cache is invalidated on changes."""
    repo = ExcludedDomainRepository(async_db_session)
    
    # Add exclusion
    await repo.add_exclusion("cached.com", "exact")
    
    # Get all active (populates cache)
    active1 = await repo.get_all_active()
    count1 = len(active1)
    
    # Add another exclusion
    await repo.add_exclusion("cached2.com", "exact")
    
    # Cache should be invalidated, get fresh data
    active2 = await repo.get_all_active()
    count2 = len(active2)
    
    assert count2 == count1 + 1


@pytest.mark.asyncio
async def test_invalid_url_handling(async_db_session: AsyncSession):
    """Test handling of invalid URLs."""
    repo = ExcludedDomainRepository(async_db_session)
    
    await repo.add_exclusion("test.com", "exact")
    
    # Invalid URL should return False, not crash
    is_excluded, _ = await repo.is_domain_excluded("not a url")
    assert is_excluded is False
    
    is_excluded, _ = await repo.is_domain_excluded("")
    assert is_excluded is False


@pytest.mark.asyncio
async def test_case_insensitive_matching(async_db_session: AsyncSession):
    """Test that domain matching is case-insensitive."""
    repo = ExcludedDomainRepository(async_db_session)
    
    await repo.add_exclusion("YouTube.COM", "exact")  # Mixed case
    
    # Should match regardless of case
    is_excluded, _ = await repo.is_domain_excluded("https://youtube.com/watch")
    assert is_excluded is True
    
    is_excluded, _ = await repo.is_domain_excluded("https://YOUTUBE.COM/watch")
    assert is_excluded is True
    
    is_excluded, _ = await repo.is_domain_excluded("https://YouTube.com/watch")
    assert is_excluded is True
