"""
RBI financial data wrapper.

Data source: Google News RSS feed, filtered to rbi.org.in and business.india.com sources.

Why not DBIE or RBI RSS directly:
  - DBIE requires browser session cookies — programmatic access silently fails.
  - rbi.org.in RSS feeds (pressreleases_rss.xml) block non-browser User-Agents
    and frequently return connection errors from scripts.

Why Google News RSS:
  - Free, no API key, no auth, no User-Agent enforcement.
  - URL: https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en
  - Indexes RBI press releases within minutes of publication.
  - Returns structured RSS with title, link, pubDate, source name.
  - The `when:7d` operator restricts to the last 7 days for recency.
    Remove it for broader historical coverage.
"""

import logging
import re
import xml.etree.ElementTree as ET
import httpx
from apis.base import BaseAPIWrapper, EvidenceChunk

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

# Claim keyword → Google News search query
# Queries are scoped to authoritative Indian financial sources.
# `site:rbi.org.in` alone is too restrictive — RBI doesn't always rank first.
# Adding `OR site:pib.gov.in` catches PIB press releases of MPC decisions.
KEYWORD_QUERY_MAP = {
    "repo rate":        "RBI repo rate site:rbi.org.in OR site:pib.gov.in",
    "reverse repo":     "RBI reverse repo rate site:rbi.org.in OR site:pib.gov.in",
    "inflation":        "India CPI inflation RBI site:rbi.org.in OR site:mospi.gov.in",
    "cpi":              "India CPI inflation RBI site:rbi.org.in OR site:mospi.gov.in",
    "wpi":              "India WPI wholesale price index site:mospi.gov.in",
    "gdp":              "India GDP growth rate site:mospi.gov.in OR site:pib.gov.in",
    "forex":            "India foreign exchange reserves RBI site:rbi.org.in",
    "foreign exchange": "India foreign exchange reserves RBI site:rbi.org.in",
    "money supply":     "India money supply M3 RBI site:rbi.org.in",
    "m3":               "India money supply M3 RBI site:rbi.org.in",
    "bank credit":      "India bank credit growth RBI site:rbi.org.in",
    "credit":           "India bank credit growth RBI site:rbi.org.in",
    "deposit":          "India aggregate deposits banks RBI site:rbi.org.in",
    "interest rate":    "RBI interest rate monetary policy site:rbi.org.in OR site:pib.gov.in",
    "rbi":              "RBI monetary policy site:rbi.org.in OR site:pib.gov.in",
    "mpc":              "RBI MPC monetary policy committee site:rbi.org.in OR site:pib.gov.in",
    "sebi":             "SEBI regulation India site:sebi.gov.in",
    "nse":              "NSE BSE stock market India",
    "sensex":           "BSE Sensex India stock market",
    "nifty":            "NSE Nifty India stock market",
}


class RBIWrapper(BaseAPIWrapper):

    async def fetch(self, claim: str) -> list[EvidenceChunk]:
        query = self._get_query(claim)
        if not query:
            logger.info("[RBI] no keyword match for claim: %r", claim[:60])
            return []
        try:
            items = await self._fetch_google_news(query)
            if not items:
                return []
            return [self._build_chunk(item) for item in items[:self.MAX_RESULTS]]
        except Exception as e:
            logger.error("[RBI] fetch failed: %s", e)
            return []

    # ── Keyword matching ──────────────────────────────────────────────────────

    def _get_query(self, claim: str) -> str:
        lower = claim.lower()
        for keyword, query in KEYWORD_QUERY_MAP.items():
            if keyword in lower:
                return query
        return ""

    # ── Google News RSS fetch ─────────────────────────────────────────────────

    async def _fetch_google_news(self, query: str) -> list[dict]:
        """
        Fetch Google News RSS for a query.
        Returns a list of dicts with keys: title, link, source, pubDate.

        Google News RSS item structure:
          <item>
            <title>RBI keeps repo rate unchanged at 6.5%</title>
            <link>https://news.google.com/rss/articles/...</link>
            <pubDate>Fri, 05 Apr 2024 10:30:00 GMT</pubDate>
            <source url="https://rbi.org.in">Reserve Bank of India</source>
            <description>...</description>
          </item>
        """
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.get(GOOGLE_NEWS_RSS, params={
                    "q":    query,
                    "hl":   "en-IN",
                    "gl":   "IN",
                    "ceid": "IN:en",
                })
                resp.raise_for_status()

            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                return []

            items = []
            for item in channel.findall("item"):
                source_el = item.find("source")
                items.append({
                    "title":   (item.findtext("title") or "").strip(),
                    "link":    (item.findtext("link") or "").strip(),
                    "pubDate": (item.findtext("pubDate") or "")[:22],
                    "source":  source_el.text.strip() if source_el is not None else "",
                    "description": _strip_html(item.findtext("description") or ""),
                })
            return items

        except httpx.HTTPStatusError as e:
            logger.warning("[RBI] Google News HTTP %s", e.response.status_code)
            return []
        except ET.ParseError as e:
            logger.warning("[RBI] RSS parse error: %s", e)
            return []
        except Exception as e:
            logger.warning("[RBI] Google News fetch failed: %s", e)
            return []

    # ── Chunk builder ─────────────────────────────────────────────────────────

    def _build_chunk(self, item: dict) -> EvidenceChunk:
        title = item.get("title", "")
        description = item.get("description", "")
        source = item.get("source", "RBI")
        snippet = (title + ". " + description).strip()[:400] if description else title[:400]

        return self._make_chunk(
            claim_id="",
            source_name="RBI",
            source_url=item.get("link", ""),
            title=title,
            snippet=snippet,
            published_date=item.get("pubDate", ""),
            domain="",
            raw_metadata={"news_source": source},
        )


def _strip_html(text: str) -> str:
    """Remove HTML tags from Google News description snippets."""
    return re.sub(r"<[^>]+>", "", text).strip()