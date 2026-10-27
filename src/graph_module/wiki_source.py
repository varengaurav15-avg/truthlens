"""
Live evidence retrieval from Wikipedia.

Replaces the 36-fact seed file as the primary evidence source. Instead of
shipping facts, we fetch them at query time from ~7M English articles that
thousands of editors keep current. Nothing to maintain, nothing to re-seed.

Pipeline
--------
    claim -> NER entities -> Wikipedia search (typo tolerant)
          -> page extract -> sentence split
          -> keep sentences that overlap the claim
          -> NLI verifier decides SUPPORTED / REFUTED / NOT_ENOUGH_INFO

Typo tolerance is free: Wikipedia's search endpoint runs its own fuzzy match,
so "Barrack Obamma" resolves to the "Barack Obama" page.

No API key, no signup. Wikipedia asks only for a descriptive User-Agent.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

API = "https://en.wikipedia.org/w/api.php"
# Purpose-built lead-summary endpoint. Returns ~2KB of pre-rendered intro text
# regardless of article size, so it succeeds on pages where the generic
# action=query extract times out (Amitabh Bachchan, Taylor Swift, Barack Obama).
REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
UA = "TruthLens/2.0 (educational misinformation detector; contact: student project)"

_page_cache: Dict[str, List[str]] = {}
_search_cache: Dict[str, List[str]] = {}

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on",
    "at", "to", "for", "and", "or", "but", "not", "never", "no", "by", "with",
    "from", "as", "that", "this", "it", "its", "has", "have", "had", "he", "she",
    "they", "his", "her", "their", "who", "which", "also", "been", "when",
}


LAST_ERROR: str = ""   # inspectable by diagnostics; see test_wiki.py

# Wikipedia's guidance for unauthenticated clients is roughly one request per
# second. 0.12s was far too aggressive over a 23-claim run and produced random
# throttling, which made evaluation results flap between runs.
_MIN_INTERVAL = 0.35
_last_call = 0.0


def _throttle():
    """
    Space out requests.

    Wikipedia rate-limits unauthenticated clients. Firing a dozen calls in a
    couple of seconds gets you throttled, and because failures used to be
    cached as empty results, the process never recovered — the first claim
    would work and every later one returned no evidence.
    """
    global _last_call
    now = time.monotonic()
    wait = _MIN_INTERVAL - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _get_json(params: dict, timeout: float = 6.0, attempts: int = 3):
    """
    GET the Wikipedia API, with throttling and backoff.

    Raises on final failure so callers can distinguish "request failed" from
    "no results" — the two must never be conflated, or a throttled request
    looks exactly like a claim with no evidence.
    """
    global LAST_ERROR
    params = {**params, "format": "json", "formatversion": "2"}

    try:
        import requests
    except ImportError:
        requests = None

    last_exc = None
    for attempt in range(attempts):
        try:
            _throttle()
            if requests is not None:
                r = requests.get(
                    API, params=params, headers={"User-Agent": UA}, timeout=timeout
                )
                if r.status_code in (429, 503):
                    raise RuntimeError(f"rate limited (HTTP {r.status_code})")
                r.raise_for_status()
                LAST_ERROR = ""
                return r.json()

            import json
            import urllib.parse
            import urllib.request

            url = f"{API}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                LAST_ERROR = ""
                return json.loads(resp.read().decode("utf-8"))

        except Exception as exc:
            last_exc = exc
            LAST_ERROR = f"{type(exc).__name__}: {exc}"
            if attempt < attempts - 1:
                time.sleep(0.4 * (2 ** attempt))   # 0.4s, 0.8s

    raise last_exc if last_exc else RuntimeError("wikipedia request failed")


def _fetch_summary(title: str, timeout: float = 10.0) -> str:
    """
    Lead paragraph via the REST summary endpoint.

    Small, fast and size-independent — the reliable path for large articles.
    Raises on failure so the caller can fall back to action=query.
    """
    global LAST_ERROR
    import urllib.parse

    url = REST_SUMMARY + urllib.parse.quote(title.replace(" ", "_"), safe="")

    last_exc = None
    for attempt in range(3):
        try:
            _throttle()
            try:
                import requests
            except ImportError:
                requests = None

            if requests is not None:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
                if r.status_code in (429, 503):
                    raise RuntimeError(f"rate limited (HTTP {r.status_code})")
                r.raise_for_status()
                data = r.json()
            else:
                import json
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

            LAST_ERROR = ""
            return str(data.get("extract") or "")

        except Exception as exc:
            last_exc = exc
            LAST_ERROR = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(0.4 * (2 ** attempt))

    raise last_exc if last_exc else RuntimeError("summary fetch failed")


def _tokens(text: str) -> set:
    from src.text_module.textmatch import tokens as _t

    return _t(text)


def search_titles(query: str, limit: int = 2, timeout: float = 6.0) -> List[str]:
    """
    Resolve a possibly-misspelled entity name to Wikipedia page titles.

    A FAILED request is never cached. Caching failures is what made throttling
    permanent: one 429 poisoned that query for the life of the process.
    """
    key = f"{query}|{limit}"
    if key in _search_cache:
        return _search_cache[key]

    from . import wiki_cache

    disk = wiki_cache.get(f"search:{key}")
    if disk is not None:
        _search_cache[key] = disk
        return disk

    try:
        data = _get_json(
            {"action": "query", "list": "search", "srsearch": query, "srlimit": limit},
            timeout,
        )
        titles = [h["title"] for h in data.get("query", {}).get("search", [])]

        # Wikipedia's own spelling suggestion, when the search found little
        if not titles:
            sugg = data.get("query", {}).get("searchinfo", {}).get("suggestion")
            if sugg:
                data = _get_json(
                    {"action": "query", "list": "search", "srsearch": sugg, "srlimit": limit},
                    timeout,
                )
                titles = [h["title"] for h in data.get("query", {}).get("search", [])]

        _search_cache[key] = titles      # only cache a real answer
        wiki_cache.put(f"search:{key}", titles)
        return titles

    except Exception:
        return []                         # transient — let the next call retry


# Sections that describe FICTION, not the subject. Pulling from these is how
# "Amitabh Bachchan played an aged defence lawyer" became "Amitabh Bachchan is
# a retired lawyer" — a film review read as biography.
_NOISE_SECTIONS = (
    "plot", "synopsis", "story", "cast", "characters", "reception",
    "critical", "review", "soundtrack", "track listing", "music",
    "box office", "release", "marketing", "production", "development",
    "filmography", "discography", "bibliography", "references",
    "external links", "see also", "further reading", "notes",
    "in popular culture", "legacy in fiction", "portrayals",
)

# Sections that DO describe the subject factually
_BIO_SECTIONS = (
    "early life", "personal life", "career", "biography", "death", "family",
    "education", "politics", "political", "history", "background", "honours",
    "honors", "awards", "achievements", "health", "illness",
)


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "")
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if 25 <= len(p.strip()) <= 400]


def _sections(extract: str):
    """Yield (section_name, body). The lead section is named ''."""
    chunks = re.split(r"\n=+ *([^=\n]+?) *=+\n", "\n" + (extract or ""))
    yield "", chunks[0]
    for i in range(1, len(chunks) - 1, 2):
        yield chunks[i].strip().lower(), chunks[i + 1]


def page_sentences(title: str, timeout: float = 6.0) -> List[Dict[str, Any]]:
    """
    Sentences from a Wikipedia article, tagged with provenance.

    Each item is {text, section, is_lead, is_bio}. The lead section carries the
    defining facts ("X is an Indian actor...") and is weighted heavily; plot
    and review sections are dropped entirely.
    """
    if title in _page_cache:
        return _page_cache[title]

    from . import wiki_cache

    disk = wiki_cache.get(f"page:{title}")
    if disk:
        _page_cache[title] = disk
        return disk

    def _fetch(mode: str, tmo: float):
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "exlimit": "1",
            "titles": title,
            "redirects": "1",
        }
        if mode == "intro":
            params["exintro"] = "1"        # lead section only — tiny payload
        elif mode == "capped":
            params["exchars"] = "16000"
        data = _get_json(params, tmo)
        pages = data.get("query", {}).get("pages", []) or []
        return "".join(p.get("extract") or "" for p in pages)

    # REST summary first. The action=query extract endpoint has to render the
    # whole article before truncating, so on very large pages it simply never
    # returned in time — "Amitabh Bachchan" failed on every single run while
    # its search result cached fine. The REST endpoint serves a pre-rendered
    # lead of ~2KB and is effectively size-independent.
    lead = ""
    try:
        lead = _fetch_summary(title, min(timeout, 10.0))
    except Exception:
        lead = ""

    # ALWAYS try for the full body too, not just when the lead is thin.
    # Lead-only retrieval cannot answer specific claims — a boxer's birth date
    # or an album's track position live in the body, so those claims returned
    # "no evidence" and the system abstained. The disk cache makes this a
    # one-time cost per article.
    extract = lead
    for mode in ("capped", "full"):
        try:
            body = _fetch(mode, max(timeout, 20.0))
            if len(body.strip()) > len(extract.strip()):
                extract = body
            if extract.strip():
                break
        except Exception:
            continue

    if not extract.strip():
        return []          # transient or genuinely empty — do NOT cache

    out: List[Dict[str, Any]] = []
    for name, body in _sections(extract):
        if name and any(n in name for n in _NOISE_SECTIONS):
            continue
        is_lead = name == ""
        is_bio = is_lead or any(b in name for b in _BIO_SECTIONS)
        for s in _split_sentences(body):
            out.append(
                {"text": s, "section": name or "lead", "is_lead": is_lead, "is_bio": is_bio}
            )

    out = out[:400]
    if out:                # only cache a real, non-empty result
        _page_cache[title] = out
        wiki_cache.put(f"page:{title}", out)
    return out


def gather_evidence(
    claim: str,
    entity_names: List[str],
    max_pages: int = 4,
    max_sentences: int = 12,
    timeout: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Return the Wikipedia sentences most relevant to `claim`.

    Each item is {text, source, url, relevance} and is fed to the NLI verifier
    as a premise. Returns [] when nothing relevant is found — the caller then
    reports NOT_ENOUGH_INFO rather than guessing.
    """
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return []

    # Resolve entities to pages. Fall back to searching the claim itself when
    # NER found nothing usable.
    # Only the two most specific entities get a lookup. Every extra entity is
    # another search plus another page fetch; at 3+ per claim the run was
    # making ~140 requests and getting throttled at random.
    # Rank by specificity: multi-word names identify a subject far better than
    # a single common noun. Sorting by raw length alone picked "President"
    # over "Osama".
    ranked = sorted(
        [n for n in entity_names if len(str(n).strip()) >= 3],
        key=lambda n: (-len(str(n).split()), -len(str(n))),
    )[:2]

    titles: List[str] = []
    for name in ranked:
        for t in search_titles(name, limit=1, timeout=timeout):
            if t not in titles:
                titles.append(t)
        if len(titles) >= max_pages:
            break

    # ALWAYS search the claim itself as well. Entity lookup alone resolves the
    # wrong article often enough to matter — searching "Kerplunk" returns the
    # board game, while searching the whole claim returns the Green Day album.
    # Making this conditional on thin entity results was a false economy; the
    # disk cache absorbs the extra request.
    for t in search_titles(claim, limit=2, timeout=timeout):
        if t not in titles:
            titles.append(t)

    # Fuzzy overlap, not exact. A misspelled claim ("Barrack Obamma") must
    # still match correctly-spelled article text, or typo tolerance dies here
    # even though the search step resolved the right page.
    from src.text_module.textmatch import fuzzy_overlap

    # Short claims ("osama the president of usa" has only 3 content words)
    # can rarely hit two overlapping tokens, so scale the bar to claim length.
    min_overlap = 1 if len(claim_tokens) <= 4 else 2

    scored: List[Dict[str, Any]] = []
    for title in titles[:max_pages]:
        for item in page_sentences(title, timeout=timeout):
            sent = item["text"]
            matched = fuzzy_overlap(claim_tokens, _tokens(sent))
            if matched < min_overlap:
                continue

            score = matched / max(1, len(claim_tokens))

            # The lead paragraph holds the defining facts about a subject.
            # Weight it far above body text so biography beats trivia.
            if item.get("is_lead"):
                score *= 2.2
            elif item.get("is_bio"):
                score *= 1.3

            # Quoted material is usually somebody's opinion or a character's
            # line, not a statement of fact about the subject.
            if '"' in sent or "“" in sent:
                score *= 0.5

            scored.append(
                {
                    "text": sent,
                    "source": title,
                    "section": item.get("section", "lead"),
                    "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                    "relevance": round(score, 4),
                }
            )

    scored.sort(key=lambda s: s["relevance"], reverse=True)

    seen, unique = set(), []
    for s in scored:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
        if len(unique) >= max_sentences:
            break
    return unique


def healthcheck(timeout: float = 5.0) -> Dict[str, Any]:
    try:
        titles = search_titles("Barack Obama", limit=1, timeout=timeout)
        return {"reachable": bool(titles), "sample": titles[:1]}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}
