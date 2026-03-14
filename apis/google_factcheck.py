"""
Google Fact Check Tools API wrapper.
Docs: https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search

No billing required — the Fact Check Tools API is free.
Get a key at: console.cloud.google.com → enable "Fact Check Tools API" → create credentials.

What it returns: verdicts from professional fact-checkers (Snopes, AFP, FactCheck.org,
Alt News, Boom Live, etc.) who have already reviewed a claim. This is the highest-signal
source for "general" domain claims because you're getting a human expert's verdict, not
just related literature.

Limitation: coverage is sparse. Many claims won't have a match. Always pair with
WikipediaWrapper as fallback in the router.
"""

import logging
import httpx
from apis.base import BaseAPIWrapper, EvidenceChunk

logger = logging.getLogger(__name__)

BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class GoogleFactCheckWrapper(BaseAPIWrapper):

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    async def fetch(self, claim: str) -> list[EvidenceChunk]:
        if not self.api_key:
            logger.warning("[GoogleFactCheck] no API key provided, skipping")
            return []
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(BASE_URL, params={
                    "query":        claim,
                    "key":          self.api_key,
                    "pageSize":     self.MAX_RESULTS,
                    "languageCode": "en",           # also catches "hi" results in practice
                })
                resp.raise_for_status()
                claims = resp.json().get("claims", [])

            chunks = []
            for item in claims:
                # Each claim can have multiple reviews from different fact-checkers
                for review in item.get("claimReview", []):
                    url = review.get("url", "")
                    if not url:
                        continue

                    publisher = review.get("publisher", {}).get("name", "Unknown publisher")
                    rating = review.get("textualRating", "")      # e.g. "False", "Misleading"
                    claim_text = item.get("text", "")
                    claimant = item.get("claimant", "")

                    # Build a descriptive snippet so the LLM has full context
                    snippet_parts = []
                    if claimant:
                        snippet_parts.append(f"Claimed by: {claimant}.")
                    if claim_text:
                        snippet_parts.append(f"Claim: {claim_text}.")
                    if rating:
                        snippet_parts.append(f"Verdict by {publisher}: {rating}.")
                    snippet = " ".join(snippet_parts)[:400]

                    chunks.append(self._make_chunk(
                        claim_id="",
                        source_name="GoogleFactCheck",
                        source_url=url,
                        title=review.get("title", claim_text)[:120],
                        snippet=snippet,
                        published_date=review.get("reviewDate", "")[:10],  # trim to YYYY-MM-DD
                        domain="",
                        raw_metadata={
                            "publisher":     publisher,
                            "textualRating": rating,
                            "claimant":      claimant,
                            "claimDate":     item.get("claimDate", ""),
                        },
                    ))

            return chunks[:self.MAX_RESULTS]   # one claim can have many reviews; cap total

        except httpx.HTTPStatusError as e:
            logger.warning("[GoogleFactCheck] HTTP %s", e.response.status_code)
            return []
        except Exception as e:
            logger.error("[GoogleFactCheck] fetch failed: %s", e)
            return []