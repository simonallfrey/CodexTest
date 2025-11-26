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
import shutil
import string
import subprocess
import sys
import textwrap
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
PROMPT_INSTRUCTION_STARMER = (
    "summarise the following in the character of a centre-left British politician such as Keir Starmer; "
    "measured, pragmatic tone; no jokes or memes; no meta talk; no labels or prefaces; "
    "no quotation marks; keep it concise (3-5 sentences); start directly with the summary"
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


def build_dual_prompt(summary: str) -> str:
    """Construct a single prompt requesting both personas, delimited for parsing."""
    parts = [
        "ignore the next line",
        random_noise(80),
        (
            "Provide two concise summaries (3-5 sentences each) of the following text. "
            "First write in the voice of a far-right British politician (e.g., Nigel Farrage) under the marker '===FARAGE==='. "
            "Then write in the voice of a centre-left British politician (e.g., Keir Starmer) under the marker '===STARMER==='. "
            "Do not include any other text, introductions, or labels. Format exactly:\n"
            "===FARAGE===\n<farage summary>\n===STARMER===\n<starmer summary>"
        ),
        summary,
    ]
    return "\n".join(parts)


def strip_meta_lines(lines: Iterable[str]) -> List[str]:
    """Remove spinner/meta lines and leading boilerplate."""
    filtered = [ln for ln in lines if "loading" not in ln.lower()]
    while filtered and not filtered[0].strip():
        filtered.pop(0)
    while filtered and re.match(r"(?i)(here['’]s (a )?summary|here['’]s (a )?view)", filtered[0].strip()):
        filtered.pop(0)
    return filtered


def parse_dual_response(text: str) -> tuple[str, str]:
    """Extract Farage and Starmer sections from a delimited response."""
    farage, starmer = "", ""
    lower = text.lower()
    if "===farage===" in lower and "===starmer===" in lower:
        # split on markers case-insensitively
        parts = re.split(r"===farage===|\s*===starmer===\s*", text, flags=re.IGNORECASE)
        # re.split yields ['', before, between, after] depending on text; pick non-empty
        non_empty = [p.strip() for p in parts if p.strip()]
        if len(non_empty) >= 2:
            farage, starmer = non_empty[0], non_empty[1]
    if not farage:
        farage = text.strip()
    if not starmer:
        starmer = text.strip()
    return farage, starmer


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


def read_key() -> str:
    """Read a single keypress, falling back to newline on non-tty."""
    if not sys.stdin.isatty():
        return "\n"
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getch()
        try:
            return ch.decode()
        except Exception:
            return ""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def wrap_paragraphs(text: str, width: int) -> List[str]:
    """Wrap multi-paragraph text to a target width."""
    lines: List[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width) or [""])
    return lines or [""]


def build_page_text(
    idx: int,
    total: int,
    article: Article,
    response: str,
    term_cols: int,
    term_lines: int,
    use_color: bool,
) -> str:
    """Create a vertically centered page for one article."""
    width = max(40, term_cols - 4)
    content: List[str] = []
    content.append(color(f"### {idx}/{total} {article.title}", "92;1", use_color))
    content.append("")
    content.extend(wrap_paragraphs(response, width=width))

    # Center vertically
    available_lines = max(10, term_lines - 2)
    pad_top = max(0, (available_lines - len(content)) // 2)
    padded = [""] * pad_top + content
    return "\n".join(padded)


def render(
    articles: Sequence[Article],
    limit: int,
    use_color: bool,
    cache_path: Path,
    refresh: bool,
) -> None:
    """Render headlines with two persona summaries and interactive paging."""
    capped = articles[:limit] if limit > 0 else articles
    cache = load_cache(cache_path)

    # Prepare responses (respect cache) with a single tgpt call per article
    responses_farage: List[str] = []
    responses_starmer: List[str] = []
    for article in capped:
        key = article_key(article)
        farage_key = f"{key}:farage"
        starmer_key = f"{key}:starmer"

        if not refresh and farage_key in cache and starmer_key in cache:
            resp_farage = cache[farage_key]
            resp_starmer = cache[starmer_key]
        else:
            prompt_dual = build_dual_prompt(" ".join(to_three_lines(article.summary, width=120)))
            sys.stderr.write(f"Querying combined personas for: {article.title}\n")
            sys.stderr.write(f"Prompt:\n{prompt_dual}\n\n")
            sys.stderr.flush()
            combined = call_tgpt(prompt_dual)
            resp_farage, resp_starmer = parse_dual_response(combined)
            cache[farage_key] = resp_farage
            cache[starmer_key] = resp_starmer

        responses_farage.append(resp_farage)
        responses_starmer.append(resp_starmer)

    # Persist cache regardless of rendering mode
    save_cache(cache_path, cache)

    # Non-interactive: dump all pages sequentially
    if not sys.stdout.isatty():
        print(color("# Guardian Headlines (Nigel-styled summaries via tgpt)", "96;1", use_color))
        print(f"Source: {GUARDIAN_URL}")
        print(f"Limit: {len(capped)} articles\n")
        for idx, (article, resp_f, resp_s) in enumerate(
            zip(capped, responses_farage, responses_starmer), start=1
        ):
            print(color(f"### {idx}. {article.title}", "92;1", use_color))
            print(color("Farage:", "91;1", use_color))
            print(resp_f)
            print("")
            print(color("Starmer:", "94;1", use_color))
            print(resp_s)
            print("\n---\n")
        return

    # Interactive paging: one summary per full-screen page, vertically centered
    term = shutil.get_terminal_size(fallback=(80, 24))
    total_articles = len(capped)
    idx = 0

    # Prepare a front page listing all titles
    title_lines = [color("# Guardian Headlines (Nigel-styled summaries via tgpt)", "96;1", use_color)]
    title_lines.append(f"Source: {GUARDIAN_URL}")
    title_lines.append(f"Limit: {len(capped)} articles\n")
    for i, article in enumerate(capped, start=1):
        title_lines.append(f"{i}. {article.title}")
    titles_page = "\n".join(title_lines)
    show_titles = True

    while True:
        if show_titles:
            body = titles_page
            footer = f"-- Titles page (1 of {total_articles + 1}) -- [Enter/space/n/j: next, q: quit] "
        else:
            combined_response = (
                color("Farage:", "91;1", use_color)
                + "\n"
                + responses_farage[idx]
                + "\n\n"
                + color("Starmer:", "94;1", use_color)
                + "\n"
                + responses_starmer[idx]
            )
            page_text = build_page_text(
                idx=idx + 1,
                total=total_articles,
                article=capped[idx],
                response=combined_response,
                term_cols=term.columns,
                term_lines=term.lines,
                use_color=use_color,
            )
            body = (
                color("# Guardian Headlines (Nigel-styled summaries via tgpt)", "96;1", use_color)
                + f"\nSource: {GUARDIAN_URL}\n\n"
                + page_text
            )
            footer = (
                f"-- Page {idx + 2}/{total_articles + 1} -- [Enter/space/n/j: next, p/k: prev, q: quit] "
            )

        sys.stdout.write("\033[2J\033[H")  # clear screen
        sys.stdout.write(body)

        # Move cursor to bottom line for pager hint
        sys.stdout.write(f"\033[{term.lines};1H")
        sys.stdout.write(footer)
        sys.stdout.flush()

        key = read_key()
        if not key:
            continue
        if key in {"q", "Q"}:
            break
        if show_titles:
            if key in {"p", "k", "P", "K"}:
                # stay on titles
                continue
            # advance to first article
            show_titles = False
            idx = 0
            continue
        if key in {"p", "k", "P", "K"}:
            if idx == 0:
                show_titles = True
            else:
                idx -= 1
            continue
        # default advance
        if idx + 1 >= total_articles:
            # wrap to titles
            show_titles = True
            idx = 0
        else:
            idx += 1


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
