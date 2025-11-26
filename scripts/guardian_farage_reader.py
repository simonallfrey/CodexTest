#!/usr/bin/env python3
"""
Guardian reader that rewrites article summaries in the voice of a far-right British
politician using the local `tgpt` client. Designed as clean, readable reference Python.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import unescape
import hashlib
import json
import os
import re
import secrets
import string
import subprocess
import sys
from pathlib import Path
from textwrap import wrap
from typing import Dict, Iterable, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

TGPT_BIN = "/usr/local/bin/tgpt"
GUARDIAN_URL = "https://www.theguardian.com/international/rss"
DEFAULT_LIMIT = 10
CACHE_PATH = Path(".guardian_farage_cache.json")
PROMPT_INSTRUCTION = (
    "summarise the following in the character of a far right british politician such as Nigel Farrage; "
    "serious and direct; no jokes or memes; no meta talk; no quotation marks; keep it concise (3-5 sentences); "
    "no preamble—start directly with the summary"
)


@dataclass(frozen=True)
class Article:
    title: str
    summary: str


def fetch_guardian(url: str = GUARDIAN_URL, timeout: int = 10) -> str:
    """Download the Guardian RSS feed."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GuardianHeadlineFetcher/1.0)",
        "Accept": "application/rss+xml, text/xml;q=0.9, */*;q=0.8",
    }
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def clean_text(html_text: str) -> str:
    """Strip tags and decode entities for a plain-text summary."""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_rss(xml_text: str) -> List[Article]:
    """Parse RSS XML into Article objects."""
    root = ET.fromstring(xml_text)
    articles: List[Article] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        description = item.findtext("description") or ""
        if not title:
            continue
        articles.append(Article(title=clean_text(title), summary=clean_text(description)))
    return articles


def to_three_lines(text: str, width: int = 80) -> List[str]:
    """Wrap text to at most three lines for brevity."""
    if not text:
        return ["(no summary available)"]
    wrapped = wrap(text, width=width)
    return wrapped[:3] or ["(no summary available)"]


def random_noise(length: int = 16) -> str:
    """Return a line of random characters to precede the instruction."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_prompt(summary: str) -> str:
    """Construct the prompt passed to tgpt."""
    parts = [
        "ignore the next line",
        random_noise(),
        PROMPT_INSTRUCTION,
        summary,
    ]
    return "\n".join(parts)


def strip_meta_lines(lines: Iterable[str]) -> List[str]:
    """Remove spinner/meta lines and leading boilerplate."""
    filtered = [ln for ln in lines if "loading" not in ln.lower()]
    while filtered and not filtered[0].strip():
        filtered.pop(0)
    while filtered and re.match(r"(?i)here['’]s (a )?summary", filtered[0].strip()):
        filtered.pop(0)
    return filtered


def call_tgpt(prompt: str, bin_path: str = TGPT_BIN, timeout: int = 45) -> str:
    """Call tgpt with the given prompt and return cleaned text."""
    if not os.path.exists(bin_path):
        return "(tgpt binary not found at /usr/local/bin/tgpt)"

    try:
        result = subprocess.run(
            [bin_path, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return f"(tgpt failed: {exc})"

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        return f"(tgpt returned non-zero exit code {result.returncode}: {err})"

    lines = strip_meta_lines(result.stdout.splitlines())
    response = "\n".join(lines).strip()
    return response or "(tgpt returned empty response)"


def color(text: str, code: str, enabled: bool) -> str:
    """Optional ANSI coloring when stdout is a TTY."""
    return f"\033[{code}m{text}\033[0m" if enabled else text


def article_key(article: Article) -> str:
    """Stable key for caching per-article responses."""
    digest = hashlib.sha256(f"{article.title}|{article.summary}".encode("utf-8")).hexdigest()
    return digest


def load_cache(path: Path) -> Dict[str, str]:
    """Load cached responses from disk."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def save_cache(path: Path, cache: Dict[str, str]) -> None:
    """Persist cache to disk."""
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except Exception:
        # Fail silently; caching is optional
        pass


def render(articles: Sequence[Article], limit: int, use_color: bool, cache_path: Path, refresh: bool) -> None:
    """Print headlines and tgpt-rendered summaries."""
    capped = articles[:limit] if limit > 0 else articles
    cache = load_cache(cache_path)

    print(color("# Guardian Headlines (Nigel-styled summaries via tgpt)", "96;1", use_color))
    print(f"Source: {GUARDIAN_URL}")
    print(f"Limit: {len(capped)} articles\n")

    for idx, article in enumerate(capped, start=1):
        prompt = build_prompt(" ".join(to_three_lines(article.summary, width=120)))
        key = article_key(article)
        if not refresh and key in cache:
            response = cache[key]
        else:
            response = call_tgpt(prompt)
            cache[key] = response
        print(color(f"### {idx}. {article.title}", "92;1", use_color))
        print(response)
        print("\n---\n")

    save_cache(cache_path, cache)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Guardian reader with Nigel-styled summaries via tgpt (local client)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of articles to process (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=CACHE_PATH,
        help=f"Path to cache file (default: {CACHE_PATH})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing cache and recompute summaries.",
    )
    args = parser.parse_args(argv)

    try:
        feed = fetch_guardian()
        articles = parse_rss(feed)
    except (HTTPError, URLError, TimeoutError) as exc:
        sys.stderr.write(f"Network error: {exc}\n")
        return 1
    except ET.ParseError as exc:
        sys.stderr.write(f"Failed to parse RSS feed: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1

    if not articles:
        sys.stderr.write("No headlines found in the Guardian feed.\n")
        return 1

    use_color = sys.stdout.isatty()
    render(articles, limit=args.limit, use_color=use_color, cache_path=args.cache_path, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
