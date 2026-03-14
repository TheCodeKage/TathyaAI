from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import httpx

TRUST_WEIGHTS = {
    "PubMed":           0.95,
    "WHO":              0.95,
    "IndianKanoon":     0.90,
    "SemanticScholar":  0.85,
    "OpenAlex":         0.85,
    "RBI":              0.90,
    "GoogleFactCheck":  0.80,
    "Wikipedia":        0.60,
    "SerpAPI":          0.40,
}


@dataclass
class EvidenceChunk:
    claim_id: str          # which claim this evidence belongs to
    source_name: str       # e.g. "PubMed", "IndianKanoon"
    source_url: str        # direct link to the original
    title: str             # article/case/document title
    snippet: str           # the relevant excerpt (200-400 chars max)
    published_date: str    # ISO format, or "" if unavailable
    trust_weight: float    # 0.0 – 1.0, assigned by source
    domain: str            # "medical" | "legal" | "financial" | etc.
    raw_metadata: dict     # anything extra, for debugging only


class BaseAPIWrapper(ABC):
    TIMEOUT = 10  # seconds
    MAX_RESULTS = 5

    @abstractmethod
    async def fetch(self, claim: str) -> list[EvidenceChunk]:
        """Fetch evidence for a claim. Always returns a list, never raises."""
        ...

    def _make_chunk(self, **kwargs) -> EvidenceChunk:
        kwargs.setdefault("trust_weight", TRUST_WEIGHTS.get(kwargs.get("source_name", ""), 0.5))
        return EvidenceChunk(**kwargs)
