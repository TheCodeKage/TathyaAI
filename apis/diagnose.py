"""
diagnose.py — run this first before pytest to identify the root cause.

    python diagnose.py

Checks:
  1. Raw network connectivity to each API host
  2. Whether each wrapper's fetch() logs any warnings/errors
  3. Whether the issue is DNS, SSL, timeout, or HTTP error
"""

import asyncio
import logging
import sys
import os

# Show ALL log output so wrapper warnings are visible
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from dotenv import load_dotenv
load_dotenv()

HOSTS = [
    ("PubMed",          "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=paracetamol&retmax=1&retmode=json"),
    ("SemanticScholar", "https://api.semanticscholar.org/graph/v1/paper/search?query=paracetamol&fields=title&limit=1"),
    ("Wikipedia",       "https://en.wikipedia.org/api/rest_v1/page/summary/Paracetamol"),
    ("S2 Snippet",      "https://api.semanticscholar.org/graph/v1/snippet/search?query=paracetamol&limit=1"),
]

async def check_connectivity():
    print("\n" + "="*60)
    print("STEP 1 — Raw HTTP connectivity")
    print("="*60)
    async with httpx.AsyncClient(timeout=10) as client:
        for name, url in HOSTS:
            try:
                resp = await client.get(url)
                print(f"  {'OK' if resp.status_code < 400 else 'FAIL':4s}  {name:20s}  HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    print(f"        Response: {resp.text[:200]}")
            except httpx.ConnectError as e:
                print(f"  CONN  {name:20s}  ConnectError — DNS failure or no internet: {e}")
            except httpx.TimeoutException as e:
                print(f"  TIME  {name:20s}  Timeout — host unreachable or firewall: {e}")
            except Exception as e:
                print(f"  ERR   {name:20s}  {type(e).__name__}: {e}")

async def check_wrappers():
    print("\n" + "="*60)
    print("STEP 2 — Wrapper fetch() with debug logging")
    print("="*60)

    from apis.pubmed import PubMedWrapper
    from apis.semantic_scholar import SemanticScholarWrapper
    from apis.wikipedia import WikipediaWrapper

    claim = "Paracetamol causes liver damage at doses above 4 grams per day"

    print(f"\n--- PubMed ---")
    chunks = await PubMedWrapper().fetch(claim)
    print(f"Result: {len(chunks)} chunks")

    print(f"\n--- SemanticScholar ---")
    chunks = await SemanticScholarWrapper(api_key=os.getenv("S2_API_KEY")).fetch(claim)
    print(f"Result: {len(chunks)} chunks")

    print(f"\n--- Wikipedia ---")
    chunks = await WikipediaWrapper().fetch(claim)
    print(f"Result: {len(chunks)} chunks")

async def check_pubmed_raw():
    print("\n" + "="*60)
    print("STEP 3 — PubMed raw response inspection")
    print("="*60)
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": "paracetamol liver", "retmax": 3, "retmode": "json"}
            )
            print(f"  Status: {resp.status_code}")
            print(f"  Body:   {resp.text[:500]}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print()
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": "33798653", "retmode": "json"}
            )
            print(f"  esummary status: {resp.status_code}")
            import json
            data = resp.json()
            result = data.get("result", {})
            uid = list(result.keys() - {"uids"})[0] if result else None
            if uid:
                art = result[uid]
                print(f"  Sample title: {art.get('title','')[:80]}")
                print(f"  Sample keys:  {list(art.keys())}")
            else:
                print(f"  Body: {resp.text[:300]}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

async def main():
    await check_connectivity()
    await check_pubmed_raw()
    await check_wrappers()
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("""
What to look for:
  ConnectError  → No internet, or firewall blocking outbound HTTPS.
                  Check your network / VPN / proxy settings.
  Timeout       → Host is reachable but slow. Try increasing TIMEOUT in base.py.
  HTTP 403      → API key missing or invalid.
  HTTP 429      → Rate limited. Add delays or get an API key.
  HTTP 4xx/5xx  → API-side issue. Check the raw body printed above.
  0 chunks but no error → A parsing bug. Check STEP 3 raw response shape
                          against what the wrapper expects.
""")

asyncio.run(main())