import asyncio
import logging
import httpx
from apis.base import BaseAPIWrapper, EvidenceChunk

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,abstract,year,citationCount,publicationTypes,externalIds,openAccessPdf"
SNIPPET_FIELDS = "snippet.text,snippet.snippetKind,snippet.section"
STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","do","does",
    "will","would","could","should","may","might","to","of","in","on","at","by",
    "for","with","and","or","but","not","it","this","that","known","said","found",
    "shown","reported","suggested","claimed","according","above","below","causes",
    "cause","per","doses","dose","grams","gram","day","days",
}

def _clean_query(claim: str) -> str:
    """Strip stopwords, replace hyphens (S2 returns nothing for hyphenated terms), cap at 8 keywords."""
    words = claim.replace("-", " ").split()
    keywords = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    return " ".join(keywords[:8])

def _paper_url(paper: dict) -> str:
    ext = paper.get("externalIds") or {}
    if doi := ext.get("DOI"):
        return f"https://doi.org/{doi}"
    if pmid := ext.get("PMID"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    pid = paper.get("paperId", "")
    return f"https://www.semanticscholar.org/paper/{pid}"


class SemanticScholarWrapper(BaseAPIWrapper):

    def __init__(self, api_key: str | None = None, domain_filter: str | None = None):
        """
        api_key:       Optional S2 API key. Without it: 1 req/sec, 5000 req/day.
                       Get a free key at semanticscholar.org/product/api for 10 req/sec.
        domain_filter: Optional S2 fieldsOfStudy string, e.g. "Medicine,Biology".
                       Pass from the router based on domain -- not this wrapper's concern.
        """
        self._headers = {"x-api-key": api_key} if api_key else {}
        self._domain_filter = domain_filter
        self._has_api_key = api_key is not None

    async def fetch(self, claim: str) -> list[EvidenceChunk]:
        try:
            query = _clean_query(claim)
            if not query:
                return []

            if self._has_api_key:
                # With a key: 10 req/sec -- safe to run concurrently
                snippets, papers = await asyncio.gather(
                    self._fetch_snippets(query),
                    self._fetch_papers(query),
                )
            else:
                # Without a key: 1 req/sec public limit -- run sequentially with a delay
                # to avoid the 429 that fires when both calls land in the same second
                snippets = await self._fetch_snippets(query)
                await asyncio.sleep(1.1)
                papers = await self._fetch_papers(query)

            return self._merge(snippets, papers)
        except Exception as e:
            logger.error("[SemanticScholar] fetch failed: %s", e)
            return []

    async def _fetch_snippets(self, query: str) -> list[dict]:
        params = {
            "query": query,
            "fields": SNIPPET_FIELDS,
            "limit": self.MAX_RESULTS,
            "minCitationCount": 5,
        }
        if self._domain_filter:
            params["fieldsOfStudy"] = self._domain_filter
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(f"{BASE_URL}/snippet/search", params=params, headers=self._headers)
                resp.raise_for_status()
                return resp.json().get("data", [])
        except httpx.HTTPStatusError as e:
            logger.warning("[SemanticScholar] snippet/search HTTP %s", e.response.status_code)
            return []
        except Exception as e:
            logger.warning("[SemanticScholar] snippet/search failed: %s", e)
            return []

    async def _fetch_papers(self, query: str) -> list[dict]:
        params = {
            "query": query,
            "fields": PAPER_FIELDS,
            "limit": self.MAX_RESULTS,
            "minCitationCount": 5,
        }
        if self._domain_filter:
            params["fieldsOfStudy"] = self._domain_filter
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(f"{BASE_URL}/paper/search", params=params, headers=self._headers)
                resp.raise_for_status()
                return resp.json().get("data", [])
        except httpx.HTTPStatusError as e:
            logger.warning("[SemanticScholar] paper/search HTTP %s", e.response.status_code)
            return []
        except Exception as e:
            logger.warning("[SemanticScholar] paper/search failed: %s", e)
            return []

    def _merge(self, snippets: list[dict], papers: list[dict]) -> list[EvidenceChunk]:
        seen: dict[str, EvidenceChunk] = {}

        for result in snippets:
            paper = result.get("paper") or {}
            pid = paper.get("paperId", "")
            if not pid:
                continue
            s = result.get("snippet") or {}
            kind = s.get("snippetKind", "")
            section = (s.get("section") or {}).get("title", "")
            prefix = "[Abstract] " if kind == "abstract" else (f"[{section}] " if section else "")
            seen[pid] = self._make_chunk(
                claim_id="",
                source_name="SemanticScholar",
                source_url=_paper_url(paper),
                title=paper.get("title", ""),
                snippet=(prefix + s.get("text", ""))[:500],
                published_date=str(paper.get("year", "")),
                domain="",
                raw_metadata={"paperId": pid, "citationCount": paper.get("citationCount"), "source": "snippet_search"},
            )

        for paper in papers:
            pid = paper.get("paperId", "")
            if not pid or pid in seen:
                continue
            abstract = paper.get("abstract") or ""
            seen[pid] = self._make_chunk(
                claim_id="",
                source_name="SemanticScholar",
                source_url=_paper_url(paper),
                title=paper.get("title", ""),
                snippet=("[Abstract] " + abstract)[:500] if abstract else paper.get("title", ""),
                published_date=str(paper.get("year", "")),
                domain="",
                raw_metadata={"paperId": pid, "citationCount": paper.get("citationCount"), "source": "paper_search"},
            )

        return list(seen.values())