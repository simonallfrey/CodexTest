#!/usr/bin/env python3
"""
Guardian headlines reader rendered with Textual.

Behavior
- Fetches Guardian RSS headlines and summaries.
- Builds a Farage-style summary via a single tgpt prompt per article (cached).
- Presents a split-view TUI: headlines list (left) and summary pane (right).
- Searching: press "/" to enter a query; "n"/"N" to jump next/previous match.
- Navigation: arrow keys or j/k to move; Enter/space to open selection.

Requirements
- Python 3.10+
- textual >= 0.40 (install via `pip install textual`)
- Network access for RSS and tgpt.
- `/usr/local/bin/tgpt` available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Dict, Iterable, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from html import unescape
import xml.etree.ElementTree as ET

from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, ListItem, ListView, Static

import subprocess

# Configuration
TGPT_BIN = "/usr/local/bin/tgpt"
GUARDIAN_URL = "https://www.theguardian.com/international/rss"
CACHE_PATH = Path(".guardian_farage_cache.json")
DEFAULT_LIMIT = 30
PROMPT_INSTRUCTION = (
    "summarise the following in the character of a far right british politician such as Nigel Farrage; "
    "serious and direct; no jokes or memes; no meta talk; no quotation marks; keep it concise (3-5 sentences); "
    "no preamble—start directly with the summary"
)


# Data models
@dataclass(frozen=True)
class Article:
    title: str
    summary: str


# Data helpers
def fetch_guardian(url: str = GUARDIAN_URL, timeout: int = 10) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GuardianHeadlineFetcher/1.0)",
        "Accept": "application/rss+xml, text/xml;q=0.9, */*;q=0.8",
    }
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def clean_text(html_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_rss(xml_text: str) -> List[Article]:
    root = ET.fromstring(xml_text)
    articles: List[Article] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        description = item.findtext("description") or ""
        if not title:
            continue
        articles.append(Article(title=clean_text(title), summary=clean_text(description)))
    return articles


def to_three_lines(text: str, width: int = 80) -> str:
    if not text:
        return "(no summary available)"
    wrapped = wrap(text, width=width)
    return " ".join(wrapped[:3])


def random_noise(length: int = 80) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_prompt(summary: str) -> str:
    return "\n".join(
        [
            "ignore the next line",
            random_noise(80),
            PROMPT_INSTRUCTION,
            summary,
        ]
    )


def strip_meta_lines(lines: Iterable[str]) -> List[str]:
    filtered = [ln for ln in lines if "loading" not in ln.lower()]
    while filtered and not filtered[0].strip():
        filtered.pop(0)
    while filtered and re.match(r"(?i)(here['’]s (a )?summary|here['’]s (a )?view)", filtered[0].strip()):
        filtered.pop(0)
    return filtered


def call_tgpt(prompt: str, bin_path: str = TGPT_BIN, timeout: int = 45) -> str:
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


def article_key(article: Article) -> str:
    return hashlib.sha256(f"{article.title}|{article.summary}".encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def save_cache(path: Path, cache: Dict[str, str]) -> None:
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def prepare_responses(
    articles: Sequence[Article],
    cache: Dict[str, str],
    refresh: bool,
) -> List[str]:
    """Return Farage-styled responses for each article, populating cache."""
    results: List[str] = []
    for article in articles:
        key = f"{article_key(article)}:farage"
        if not refresh and key in cache:
            results.append(cache[key])
            continue

        prompt = build_prompt(to_three_lines(article.summary, width=120))
        sys.stderr.write(f"Querying Farage persona for: {article.title}\n")
        sys.stderr.write(f"Prompt:\n{prompt}\n\n")
        sys.stderr.flush()
        response = call_tgpt(prompt)
        cache[key] = response
        results.append(response)
    return results


def render_noninteractive(
    articles: Sequence[Article],
    responses: Sequence[str],
    use_color: bool,
) -> None:
    """Print all summaries sequentially (no paging)."""
    print(color("# Guardian Headlines (Nigel-styled summaries via tgpt)", "96;1", use_color))
    print(f"Source: {GUARDIAN_URL}")
    print(f"Limit: {len(articles)} articles\n")
    for idx, (article, response) in enumerate(zip(articles, responses), start=1):
        print(color(f"### {idx}. {article.title}", "92;1", use_color))
        print(response)
        print("\n---\n")


def find_match(
    query: str,
    titles: Sequence[str],
    responses: Sequence[str],
    start: int,
    direction: int = 1,
) -> int | None:
    """Find the next index containing the query (case-insensitive), wrapping once."""
    total = len(titles)
    if total == 0:
        return None
    q = query.lower()
    idx = start
    for _ in range(total):
        idx = (idx + direction) % total
        if q in titles[idx].lower() or q in responses[idx].lower():
            return idx
    return None


# Textual UI
class GuardianApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    Input {
        dock: bottom;
        display: none;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("/", "search", "Search"),
        ("n", "search_next", "Next match"),
        ("N", "search_prev", "Prev match"),
        ("G", "last_article", "Last article"),
    ]

    def __init__(
        self,
        limit: int,
        refresh: bool,
        cache_path: Path,
        articles: list[Article] | None = None,
        summaries: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.limit = limit
        self.refresh = refresh
        self.cache_path = cache_path
        self.cache = load_cache(cache_path)
        self.input_active = False
        self.initial_articles = articles or []
        self.initial_summaries = summaries or []
        self.articles: list[Article] = []
        self.summaries: list[str] = []
        self.matches_query: str | None = None
        self.current_index: int = 0
        self.showing_overview: bool = True
        self._pending_g: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ListView(id="list")
        yield Input(placeholder="Search...", id="search_input")
        yield Footer()

    async def on_mount(self) -> None:
        # Data is pre-fetched before launching the TUI.
        self.articles = self.initial_articles
        self.summaries = self.initial_summaries
        self.render_overview()

    def render_overview(self) -> None:
        """Show overview page with all titles."""
        self.showing_overview = True
        list_view = self.query_one(ListView)
        list_view.clear()
        titles = "\n".join(f"{idx+1}. {article.title}" for idx, article in enumerate(self.articles))
        list_view.append(ListItem(Static(f"[b]Guardian Headlines[/b]\n{titles}")))
        list_view.index = 0

    def show_article(self, index: int) -> None:
        if not self.articles:
            return
        index = max(0, min(index, len(self.articles) - 1))
        self.current_index = index
        self.showing_overview = False
        body = f"[b]{self.articles[index].title}[/b]\n\n{self.summaries[index]}"
        list_view = self.query_one(ListView)
        list_view.clear()
        list_view.append(ListItem(Static(body)))
        list_view.index = 0

    def action_quit(self) -> None:
        self.exit()

    def action_search(self) -> None:
        search_input = self.query_one("#search_input", Input)
        search_input.display = True
        search_input.value = ""
        search_input.focus()
        self.input_active = True
        self._pending_g = False

    def action_search_next(self) -> None:
        self._jump_match(direction=1)

    def action_search_prev(self) -> None:
        self._jump_match(direction=-1)

    def _jump_match(self, direction: int) -> None:
        if not self.matches_query:
            return
        found = find_match(
            self.matches_query,
            [a.title for a in self.articles],
            self.summaries,
            self.current_index,
            direction=direction,
        )
        if found is not None:
            self.show_article(found)

    def action_last_article(self) -> None:
        if self.articles:
            self.show_article(len(self.articles) - 1)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # If selecting the first overview item, ignore; otherwise show selected article.
        if event.index == 0:
            return
        self.show_article(event.index - 1)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        event.input.display = False
        self.input_active = False
        self._pending_g = False
        if not query:
            return
        self.matches_query = query
        found = find_match(
            query,
            [a.title for a in self.articles],
            self.summaries,
            self.current_index,
            direction=1,
        )
        if found is not None:
            self.show_article(found)

    def on_key(self, event: events.Key) -> None:
        # Preserve j/k navigation even when not bound explicitly
        if self.input_active:
            return
        if event.key == "g":
            if self._pending_g:
                self.render_overview()
                self._pending_g = False
            else:
                self._pending_g = True
            return
        self._pending_g = False

        if event.key in {"j", "down"}:
            if self.showing_overview:
                self.show_article(0)
            else:
                self.show_article(self.current_index + 1)
        elif event.key in {"k", "up"}:
            if self.showing_overview:
                return
            self.show_article(self.current_index - 1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Textual Guardian reader with Farage-styled summaries via tgpt.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max articles to load (default: 30)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and recompute summaries.")
    args = parser.parse_args(argv)

    try:
        feed = fetch_guardian()
        all_articles = parse_rss(feed)
        articles = all_articles[: args.limit] if args.limit > 0 else all_articles
    except (HTTPError, URLError, TimeoutError) as exc:
        sys.stderr.write(f"Network error: {exc}\n")
        return 1
    except ET.ParseError as exc:
        sys.stderr.write(f"Failed to parse RSS feed: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1

    cache = load_cache(CACHE_PATH)
    summaries = prepare_responses(articles, cache=cache, refresh=args.refresh)
    save_cache(CACHE_PATH, cache)

    app = GuardianApp(
        limit=args.limit,
        refresh=args.refresh,
        cache_path=CACHE_PATH,
        articles=articles,
        summaries=summaries,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
