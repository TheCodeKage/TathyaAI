import asyncio
import logging
from apis.pubmed import PubMedWrapper
from apis.indiankanoon import IndianKanoonWrapper
from apis.semantic_scholar import SemanticScholarWrapper
from apis.google_factcheck import GoogleFactCheckWrapper
from apis.wikipedia import WikipediaWrapper
from apis.rbi import RBIWrapper
from apis.base import EvidenceChunk

logger = logging.getLogger(__name__)

# S2 fieldsOfStudy filter per domain — router's concern, not the wrapper's
_S2_FIELD_FILTER = {
    "medical":   "Medicine,Biology",
    "financial": "Economics",
    "academic":  None,
    "legal":     None,
    "general":   None,
}


class EvidenceRouter:
    def __init__(self, config: dict):
        s2_key = config.get("s2_api_key")
        ik_token = config.get("indiankanoon_token")
        gfc_key = config.get("google_api_key")

        self.wrappers = {
            "medical":   [PubMedWrapper(),
                          SemanticScholarWrapper(api_key=s2_key, domain_filter=_S2_FIELD_FILTER["medical"]),
                          WikipediaWrapper()],
            "legal":     [IndianKanoonWrapper(api_token=ik_token),
                          WikipediaWrapper()],
            "financial": [RBIWrapper(),
                          WikipediaWrapper()],
            "academic":  [SemanticScholarWrapper(api_key=s2_key, domain_filter=None),
                          WikipediaWrapper()],
            "general":   [GoogleFactCheckWrapper(api_key=gfc_key),
                          WikipediaWrapper()],
        }

    async def fetch_all(self, claim_id: str, claim_text: str, domain: str) -> list[EvidenceChunk]:
        wrappers = self.wrappers.get(domain, self.wrappers["general"])

        results = await asyncio.gather(*[w.fetch(claim_text) for w in wrappers])

        seen_urls: set[str] = set()
        chunks: list[EvidenceChunk] = []
        for batch in results:
            for chunk in batch:
                if chunk.source_url not in seen_urls:
                    chunk.claim_id = claim_id   # tag here, not inside the wrapper
                    chunk.domain = domain        # same — wrapper leaves this blank
                    seen_urls.add(chunk.source_url)
                    chunks.append(chunk)

        logger.info("[Router] claim=%r domain=%s → %d chunks total", claim_text[:60], domain, len(chunks))
        return chunks