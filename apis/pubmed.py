import logging
import httpx
from apis.base import BaseAPIWrapper, EvidenceChunk

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","do","does",
    "will","would","could","should","may","might","to","of","in","on","at","by",
    "for","with","and","or","but","not","it","this","that","known","said","found",
    "shown","reported","suggested","claimed","according","above","below","causes",
    "cause","per","doses","dose","grams","gram","day","days",
}

def _clean_query(claim: str) -> str:
    """
    PubMed term search does not handle natural language sentences.
    Strip stopwords and cap at 6 keywords.
    Also replaces hyphens — PubMed handles them, but spaces are safer for multi-word terms.

    Example:
        "Paracetamol causes liver damage at doses above 4 grams per day"
        → "Paracetamol liver damage 4"
    """
    words = claim.replace("-", " ").split()
    keywords = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    return " ".join(keywords[:6])


class PubMedWrapper(BaseAPIWrapper):

    async def fetch(self, claim: str) -> list[EvidenceChunk]:
        try:
            query = _clean_query(claim)
            if not query:
                return []

            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                # Step 1: get IDs
                search = await client.get(f"{BASE_URL}/esearch.fcgi", params={
                    "db": "pubmed", "term": query,
                    "retmax": self.MAX_RESULTS, "retmode": "json",
                })
                search.raise_for_status()
                ids = search.json()["esearchresult"]["idlist"]
                if not ids:
                    logger.info("[PubMed] no IDs returned for query: %r", query)
                    return []

                # Step 2: fetch summaries
                summary = await client.get(f"{BASE_URL}/esummary.fcgi", params={
                    "db": "pubmed", "id": ",".join(ids), "retmode": "json",
                })
                summary.raise_for_status()
                result = summary.json()["result"]

            chunks = []
            for uid in ids:
                art = result.get(uid, {})
                title = art.get("title", "")
                if not title:
                    continue
                chunks.append(self._make_chunk(
                    claim_id="",
                    source_name="PubMed",
                    source_url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    title=title,
                    snippet=(title + ". " + art.get("source", ""))[:400],
                    published_date=art.get("pubdate", ""),
                    domain="",
                    raw_metadata={
                        "uid":     uid,
                        "authors": [a.get("name") for a in art.get("authors", [])],
                        "source":  art.get("source", ""),
                    },
                ))
            return chunks

        except httpx.HTTPStatusError as e:
            logger.warning("[PubMed] HTTP %s", e.response.status_code)
            return []
        except Exception as e:
            logger.error("[PubMed] fetch failed: %s", e)
            return []