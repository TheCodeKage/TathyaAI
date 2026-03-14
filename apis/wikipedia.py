import asyncio
import logging
import httpx
from apis.base import BaseAPIWrapper, EvidenceChunk

logger = logging.getLogger(__name__)

SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Wikipedia blocks requests without a descriptive User-Agent (HTTP 403).
# Policy: https://w.wiki/4wJS
USER_AGENT = "TathyaAI/1.0 (fact-checking research project; https://github.com/your-repo)"

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","do","does",
    "will","would","could","should","may","might","to","of","in","on","at","by",
    "for","with","and","or","but","not","it","this","that","above","below",
    "causes","cause","per","doses","dose","grams","gram","day","days","known",
    "said","found","shown","reported","suggested","claimed","according",
    "damage","damages","increased","increase","decrease","rate","rates",
    "high","higher","low","lower","risk","risks","level","levels",
}

def _extract_subjects(claim: str) -> list[str]:
    """
    Extract 1-2 subject nouns to use as separate Wikipedia search queries.
    Wikipedia opensearch matches article titles — multi-word keyword phrases
    rarely match. Searching for each subject individually works much better.

    Strategy: take the first two non-stopword tokens with len > 3.
    These are almost always the main subjects of the claim.

    Examples:
        "Paracetamol causes liver damage at doses above 4 grams per day"
        → ["Paracetamol", "liver"]          searches: "Paracetamol", "liver"

        "Article 21 of the Indian Constitution guarantees right to life"
        → ["Article", "Indian"]             searches: "Article 21", "Indian Constitution"

        "RBI increased the repo rate to control inflation"
        → ["RBI", "repo"]                   searches: "RBI", "repo rate"

        "Transformer models use self-attention mechanisms"
        → ["Transformer", "self-attention"] searches: "Transformer", "Attention mechanism"
    """
    words = claim.replace("-", " ").split()
    subjects = [w for w in words if w.lower() not in STOPWORDS and len(w) >= 3]
    # Return top 2 as individual search terms — not joined
    return subjects[:2]


class WikipediaWrapper(BaseAPIWrapper):

    async def fetch(self, claim: str) -> list[EvidenceChunk]:
        try:
            subjects = _extract_subjects(claim)
            if not subjects:
                return []

            # Search for each subject separately, run concurrently
            search_tasks = [self._search_titles(s) for s in subjects]
            title_batches = await asyncio.gather(*search_tasks)

            # Flatten, deduplicate titles, keep order
            seen_titles: set[str] = set()
            titles: list[str] = []
            for batch in title_batches:
                for t in batch:
                    if t not in seen_titles:
                        seen_titles.add(t)
                        titles.append(t)

            if not titles:
                logger.info("[Wikipedia] no titles found for subjects: %r", subjects)
                return []

            # Cap total summary fetches to MAX_RESULTS
            summaries = await asyncio.gather(*[self._fetch_summary(t) for t in titles[:self.MAX_RESULTS]])
            return [c for c in summaries if c is not None]

        except Exception as e:
            logger.error("[Wikipedia] fetch failed: %s", e)
            return []

    async def _search_titles(self, query: str) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(SEARCH_URL, params={
                    "action":    "opensearch",
                    "search":    query,
                    "limit":     3,              # top 3 per subject; deduped upstream
                    "namespace": 0,
                    "format":    "json",
                }, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                data = resp.json()
                return data[1] if len(data) > 1 else []
        except Exception as e:
            logger.warning("[Wikipedia] title search failed for %r: %s", query, e)
            return []

    async def _fetch_summary(self, title: str) -> EvidenceChunk | None:
        try:
            encoded = title.replace(" ", "_")
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(
                    f"{SUMMARY_URL}/{encoded}",
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()

            extract = data.get("extract", "")
            if not extract:
                return None

            return self._make_chunk(
                claim_id="",
                source_name="Wikipedia",
                source_url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                title=data.get("title", title),
                snippet=extract[:400],
                published_date=data.get("timestamp", "")[:10],
                domain="",
                raw_metadata={
                    "pageid":      data.get("pageid"),
                    "description": data.get("description", ""),
                },
            )

        except httpx.HTTPStatusError as e:
            logger.warning("[Wikipedia] summary HTTP %s for title=%r", e.response.status_code, title)
            return None
        except Exception as e:
            logger.warning("[Wikipedia] summary failed for title=%r: %s", title, e)
            return None