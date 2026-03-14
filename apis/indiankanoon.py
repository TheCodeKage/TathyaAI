import httpx
from apis.base import BaseAPIWrapper, EvidenceChunk

class IndianKanoonWrapper(BaseAPIWrapper):
    BASE = "https://api.indiankanoon.org"

    def __init__(self, api_token: str):
        self.headers = {"Authorization": f"Token {api_token}"}

    async def fetch(self, claim: str) -> list[EvidenceChunk]:
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.post(
                    f"{self.BASE}/search/",
                    headers=self.headers,
                    data={"formInput": claim, "pagenum": 0}
                )
                docs = resp.json().get("docs", [])[:self.MAX_RESULTS]

                return [self._make_chunk(
                    claim_id="",
                    source_name="IndianKanoon",
                    source_url=f"https://indiankanoon.org/doc/{d['tid']}/",
                    title=d.get("title", ""),
                    snippet=d.get("headline", "")[:400],
                    published_date=d.get("publishdate", ""),
                    domain="legal",
                    raw_metadata=d
                ) for d in docs]

        except Exception as e:
            print(f"[IndianKanoon] fetch failed: {e}")
            return []