from unittest.mock import AsyncMock, MagicMock
import httpx
from privachat_agents.agents.search_agent import SearchAgent, SearchAgentDeps, SearchSource

mock_llm = MagicMock()
mock_llm.chat = AsyncMock()
deps = SearchAgentDeps(
    llm_client=mock_llm,
    tracer=MagicMock(),
    db=MagicMock(),
    searxng_client=AsyncMock(spec=httpx.AsyncClient),
    serperdev_api_key="test",
    max_sources=20,
    timeout=60.0
)
agent = SearchAgent(deps=deps)

print("Test 1: Domain extraction...")
domain = agent._extract_simplified_domain("https://www.wikipedia.org/article")
print(f"  Result: {domain}")
assert domain == "wikipedia", f"Expected 'wikipedia', got '{domain}'"
print("  ✅ Passed!")

print("\nTest 2: Citation mapping...")
answer = "AI agents [1] use reasoning [2]."
sources = [
    SearchSource(title="OpenAI", url="https://openai.com", snippet="", relevance=0.9, semantic_score=0.8, final_score=0.85, source_type="web"),
    SearchSource(title="arXiv", url="https://arxiv.org", snippet="", relevance=0.8, semantic_score=0.7, final_score=0.75, source_type="academic")
]
citations = agent._extract_citation_mapping(answer, sources)
print(f"  Result: {len(citations)} citations")
print(f"    [1] -> {citations[0]['domain']} (source_index={citations[0]['source_index']})")
print(f"    [2] -> {citations[1]['domain']} (source_index={citations[1]['source_index']})")
assert len(citations) == 2
assert citations[0]['domain'] == 'openai'
assert citations[1]['domain'] == 'arxiv'
print("  ✅ Passed!")

print("\nTest 3: Multiple mentions...")
answer2 = "AI [1] agents [1] work [2]. More [1]."
citations2 = agent._extract_citation_mapping(answer2, sources)
print(f"  [1] mentions: {citations2[0]['mention_count']}")
print(f"  [2] mentions: {citations2[1]['mention_count']}")
assert citations2[0]['mention_count'] == 3
assert citations2[1]['mention_count'] == 1
print("  ✅ Passed!")

print("\n🎉 All tests passed! Implementation is working correctly.")
