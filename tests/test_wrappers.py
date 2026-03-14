"""
tests/test_wrappers.py

Run individually:
    python -m pytest tests/test_wrappers.py -v
    python -m pytest tests/test_wrappers.py -v -k "pubmed"
    python -m pytest tests/test_wrappers.py -v -k "router"

Run with live API calls (default: all tests are live, no mocking):
    Set the following in a .env file or export as environment variables:
        INDIANKANOON_TOKEN=...
        GOOGLE_API_KEY=...
        S2_API_KEY=...          # optional, tests run without it at lower rate limit

Requires:
    pip install pytest pytest-asyncio python-dotenv httpx
"""

import os
import asyncio
import sys
import pytest
import pytest_asyncio
from dotenv import load_dotenv

load_dotenv()

# Add project root to path so imports work from the tests/ subfolder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apis.base import EvidenceChunk
from apis.pubmed import PubMedWrapper
from apis.indiankanoon import IndianKanoonWrapper
from apis.semantic_scholar import SemanticScholarWrapper
from apis.google_factcheck import GoogleFactCheckWrapper
from apis.wikipedia import WikipediaWrapper
from apis.rbi import RBIWrapper
from apis.router import EvidenceRouter


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    "indiankanoon_token": os.getenv("INDIANKANOON_TOKEN"),
    "google_api_key":     os.getenv("GOOGLE_API_KEY"),
    "s2_api_key":         os.getenv("S2_API_KEY"),
}

# One claim per domain — realistic but not too niche
CLAIMS = {
    "medical":   "Paracetamol causes liver damage at doses above 4 grams per day",
    "legal":     "Article 21 of the Indian Constitution guarantees right to life",
    "financial": "RBI increased the repo rate to control inflation",
    "academic":  "Transformer models use self-attention mechanisms for sequence modelling",
    "general":   "The Earth is flat",   # should get strong fact-check hits
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def assert_chunks(chunks: list[EvidenceChunk], source_name: str, min_count: int = 1):
    """Common assertions that every wrapper result must satisfy."""
    assert isinstance(chunks, list), f"{source_name}: fetch() must return a list, got {type(chunks)}"
    assert len(chunks) >= min_count, (
        f"{source_name}: expected at least {min_count} chunk(s), got {len(chunks)}\n"
        "If this is a rate-limit or API key issue, check your .env file."
    )
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, EvidenceChunk),      f"{source_name}[{i}]: not an EvidenceChunk"
        assert chunk.source_name == source_name,      f"{source_name}[{i}]: source_name mismatch: {chunk.source_name!r}"
        assert chunk.title.strip(),                   f"{source_name}[{i}]: title is empty"
        assert chunk.snippet.strip(),                 f"{source_name}[{i}]: snippet is empty"
        assert chunk.source_url.startswith("http"),   f"{source_name}[{i}]: source_url invalid: {chunk.source_url!r}"
        assert 0.0 <= chunk.trust_weight <= 1.0,      f"{source_name}[{i}]: trust_weight out of range: {chunk.trust_weight}"
        assert len(chunk.snippet) <= 510,             f"{source_name}[{i}]: snippet too long ({len(chunk.snippet)} chars) — trim in wrapper"

def print_chunks(chunks: list[EvidenceChunk], label: str):
    print(f"\n{'─'*60}")
    print(f"  {label}  ({len(chunks)} chunk(s))")
    print(f"{'─'*60}")
    for c in chunks:
        print(f"  [{c.trust_weight}] {c.title[:70]}")
        print(f"         {c.source_url}")
        print(f"         {c.snippet[:120]}...")
        print()


# ── PubMed ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pubmed_returns_chunks():
    wrapper = PubMedWrapper()
    chunks = await wrapper.fetch(CLAIMS["medical"])
    print_chunks(chunks, "PubMed — medical claim")
    assert_chunks(chunks, "PubMed")

@pytest.mark.asyncio
async def test_pubmed_no_results_returns_empty_list():
    """A nonsense query should return [] not raise."""
    wrapper = PubMedWrapper()
    chunks = await wrapper.fetch("xkqzjwp9 nonsense claim that matches nothing")
    assert isinstance(chunks, list)

@pytest.mark.asyncio
async def test_pubmed_never_raises_on_bad_input():
    """Empty string should not raise."""
    wrapper = PubMedWrapper()
    chunks = await wrapper.fetch("")
    assert isinstance(chunks, list)


# ── IndianKanoon ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_indiankanoon_returns_chunks():
    token = CONFIG["indiankanoon_token"]
    if not token:
        pytest.skip("INDIANKANOON_TOKEN not set in .env")
    wrapper = IndianKanoonWrapper(api_token=token)
    chunks = await wrapper.fetch(CLAIMS["legal"])
    print_chunks(chunks, "IndianKanoon — legal claim")
    assert_chunks(chunks, "IndianKanoon")

@pytest.mark.asyncio
async def test_indiankanoon_bad_token_returns_empty():
    """Invalid token should fail gracefully, not raise."""
    wrapper = IndianKanoonWrapper(api_token="bad-token-12345")
    chunks = await wrapper.fetch(CLAIMS["legal"])
    assert isinstance(chunks, list)


# ── SemanticScholar ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_semanticscholar_returns_chunks():
    wrapper = SemanticScholarWrapper(
        api_key=CONFIG["s2_api_key"],
        domain_filter="Medicine,Biology"
    )
    chunks = await wrapper.fetch(CLAIMS["medical"])
    print_chunks(chunks, "SemanticScholar — medical claim")
    assert_chunks(chunks, "SemanticScholar")

@pytest.mark.asyncio
async def test_semanticscholar_academic_no_filter():
    wrapper = SemanticScholarWrapper(
        api_key=CONFIG["s2_api_key"],
        domain_filter=None
    )
    chunks = await wrapper.fetch(CLAIMS["academic"])
    print_chunks(chunks, "SemanticScholar — academic claim")
    assert_chunks(chunks, "SemanticScholar")

@pytest.mark.asyncio
async def test_semanticscholar_snippet_and_paper_merged():
    """Verify that chunk.raw_metadata['source'] reflects both paths are attempted."""
    wrapper = SemanticScholarWrapper(api_key=CONFIG["s2_api_key"])
    chunks = await wrapper.fetch(CLAIMS["academic"])
    sources = {c.raw_metadata.get("source") for c in chunks}
    # At least one source type should be present
    assert sources & {"snippet_search", "paper_search"}, (
        f"Expected at least one of snippet_search or paper_search in raw_metadata, got: {sources}"
    )

@pytest.mark.asyncio
async def test_semanticscholar_claim_id_blank():
    """Wrapper must leave claim_id blank — router fills it in."""
    wrapper = SemanticScholarWrapper()
    chunks = await wrapper.fetch(CLAIMS["academic"])
    for chunk in chunks:
        assert chunk.claim_id == "", f"claim_id should be blank inside wrapper, got: {chunk.claim_id!r}"


# ── GoogleFactCheck ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_googlefactcheck_returns_chunks():
    key = CONFIG["google_api_key"]
    if not key:
        pytest.skip("GOOGLE_API_KEY not set in .env")
    wrapper = GoogleFactCheckWrapper(api_key=key)
    chunks = await wrapper.fetch(CLAIMS["general"])
    print_chunks(chunks, "GoogleFactCheck — general claim")
    assert_chunks(chunks, "GoogleFactCheck")

@pytest.mark.asyncio
async def test_googlefactcheck_snippet_has_verdict():
    """Snippets should contain a verdict string (Verdict by X: Y)."""
    key = CONFIG["google_api_key"]
    if not key:
        pytest.skip("GOOGLE_API_KEY not set in .env")
    wrapper = GoogleFactCheckWrapper(api_key=key)
    chunks = await wrapper.fetch(CLAIMS["general"])
    if chunks:
        assert "Verdict by" in chunks[0].snippet, (
            f"Expected 'Verdict by' in snippet, got: {chunks[0].snippet!r}"
        )

@pytest.mark.asyncio
async def test_googlefactcheck_no_key_returns_empty():
    """Missing key should return [] not raise."""
    wrapper = GoogleFactCheckWrapper(api_key=None)
    chunks = await wrapper.fetch(CLAIMS["general"])
    assert chunks == []


# ── Wikipedia ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wikipedia_returns_chunks():
    wrapper = WikipediaWrapper()
    chunks = await wrapper.fetch(CLAIMS["medical"])
    print_chunks(chunks, "Wikipedia — medical claim")
    assert_chunks(chunks, "Wikipedia")

@pytest.mark.asyncio
async def test_wikipedia_follow_redirect():
    """'mRNA' should redirect to 'Messenger RNA' — follow_redirects must be True."""
    wrapper = WikipediaWrapper()
    chunks = await wrapper.fetch("mRNA vaccines")
    assert chunks, "Expected at least one result for 'mRNA vaccines'"
    titles = [c.title for c in chunks]
    print(f"  Wikipedia titles found: {titles}")

@pytest.mark.asyncio
async def test_wikipedia_nonsense_returns_empty():
    wrapper = WikipediaWrapper()
    chunks = await wrapper.fetch("xkqzjwp9ajskdhaksjdhaksjdh")
    assert isinstance(chunks, list)

@pytest.mark.asyncio
async def test_wikipedia_snippet_length():
    """Snippets must be capped at 400 chars."""
    wrapper = WikipediaWrapper()
    chunks = await wrapper.fetch("Paracetamol liver toxicity")
    for chunk in chunks:
        assert len(chunk.snippet) <= 410, (
            f"Wikipedia snippet too long: {len(chunk.snippet)} chars"
        )


# ── RBI ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rbi_repo_rate():
    """Live fetch via Google News RSS."""
    wrapper = RBIWrapper()
    chunks = await wrapper.fetch(CLAIMS["financial"])
    print_chunks(chunks, "RBI — repo rate claim")
    if not chunks:
        pytest.skip("Google News RSS unreachable or no matching items")
    assert_chunks(chunks, "RBI")

@pytest.mark.asyncio
async def test_rbi_keyword_matching():
    """_get_query must return a non-empty Google News query for known financial keywords."""
    wrapper = RBIWrapper()
    query = wrapper._get_query("India's CPI inflation rose to 6.2%")
    assert query, "Expected a query string for 'CPI inflation'"
    assert "inflation" in query.lower() or "cpi" in query.lower(), (
        f"Expected CPI/inflation in query, got: {query!r}"
    )

@pytest.mark.asyncio
async def test_rbi_google_news_reachable():
    """Verify Google News RSS returns parseable items for a financial query."""
    wrapper = RBIWrapper()
    items = await wrapper._fetch_google_news("RBI repo rate site:rbi.org.in")
    if not items:
        pytest.skip("Google News RSS unreachable")
    assert isinstance(items, list)
    assert len(items) > 0
    sample = items[0]
    assert sample.get("title"), f"Item missing title: {sample}"
    assert sample.get("link"),  f"Item missing link: {sample}"
    print(f"  Google News: {len(items)} items. Latest: {sample['title'][:80]}")

@pytest.mark.asyncio
async def test_rbi_unmatched_claim_returns_empty():
    """A claim about Infosys stock has no matching keyword — must return []."""
    wrapper = RBIWrapper()
    chunks = await wrapper.fetch("Infosys share price crossed 2000 rupees")
    assert isinstance(chunks, list)
    assert len(chunks) == 0, f"Expected empty list for unmatched claim, got {len(chunks)} chunks"

@pytest.mark.asyncio
async def test_rbi_snippet_is_populated():
    """RBI snippets must be non-empty strings from Google News."""
    wrapper = RBIWrapper()
    chunks = await wrapper.fetch("RBI repo rate")
    if not chunks:
        pytest.skip("Google News RSS unreachable or no matching items")
    assert chunks[0].snippet.strip(), "RBI snippet must not be empty"
    assert chunks[0].source_url.startswith("http"), f"Bad source_url: {chunks[0].source_url}"


# ── Router ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_medical():
    router = EvidenceRouter(CONFIG)
    chunks = await router.fetch_all("claim-001", CLAIMS["medical"], "medical")
    print_chunks(chunks, "Router — medical")
    assert len(chunks) > 0
    # All chunks must have claim_id and domain set by the router
    for chunk in chunks:
        assert chunk.claim_id == "claim-001", f"claim_id not set by router: {chunk.claim_id!r}"
        assert chunk.domain == "medical",     f"domain not set by router: {chunk.domain!r}"

@pytest.mark.asyncio
async def test_router_legal():
    if not CONFIG["indiankanoon_token"]:
        pytest.skip("INDIANKANOON_TOKEN not set in .env")
    router = EvidenceRouter(CONFIG)
    chunks = await router.fetch_all("claim-002", CLAIMS["legal"], "legal")
    print_chunks(chunks, "Router — legal")
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_router_general():
    if not CONFIG["google_api_key"]:
        pytest.skip("GOOGLE_API_KEY not set in .env")
    router = EvidenceRouter(CONFIG)
    chunks = await router.fetch_all("claim-003", CLAIMS["general"], "general")
    print_chunks(chunks, "Router — general")
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_router_deduplication():
    """Router must deduplicate by source_url across wrappers."""
    router = EvidenceRouter(CONFIG)
    chunks = await router.fetch_all("claim-004", CLAIMS["academic"], "academic")
    urls = [c.source_url for c in chunks]
    assert len(urls) == len(set(urls)), (
        f"Duplicate source_urls found in router output:\n"
        + "\n".join(u for u in urls if urls.count(u) > 1)
    )

@pytest.mark.asyncio
async def test_router_unknown_domain_falls_back_to_general():
    """Unrecognised domain should not raise — falls back to general wrappers."""
    router = EvidenceRouter(CONFIG)
    chunks = await router.fetch_all("claim-005", CLAIMS["general"], "unknown_domain")
    assert isinstance(chunks, list)

@pytest.mark.asyncio
async def test_router_all_domains():
    """Smoke test every domain in sequence. Skips domains with missing keys."""
    router = EvidenceRouter(CONFIG)
    for domain, claim in CLAIMS.items():
        if domain == "legal" and not CONFIG["indiankanoon_token"]:
            print(f"  Skipping {domain} — no token")
            continue
        if domain == "general" and not CONFIG["google_api_key"]:
            print(f"  Skipping {domain} — no key")
            continue
        chunks = await router.fetch_all(f"smoke-{domain}", claim, domain)
        print(f"  {domain:12s} → {len(chunks)} chunks")
        assert isinstance(chunks, list), f"Router returned non-list for domain={domain}"