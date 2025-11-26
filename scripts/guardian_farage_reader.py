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

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, ListItem, ListView, Static

import subprocess
import xml.etree.ElementTree as ET

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


# Textual UI
class SummaryView(Static):
    """Displays a single article summary."""

    def update_content(self, title: str, body: str) -> None:
        self.update(f"[b]{title}[/b]\n\n{body}")


class GuardianApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    Horizontal {
        height: 1fr;
    }
    ListView {
        width: 40%;
        border: tall $primary;
    }
    SummaryView {
        border: tall $primary;
        padding: 1 2;
    }
    Input {
        dock: bottom;
        display: none;
    }
    """

    class Loaded(Message):
        def __init__(self, articles: list[Article], summaries: list[str]) -> None:
            self.articles = articles
            self.summaries = summaries
            super().__init__()

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("/", "search", "Search"),
        ("n", "search_next", "Next match"),
        ("N", "search_prev", "Prev match"),
    ]

    articles: reactive[list[Article]] = reactive([])
    summaries: reactive[list[str]] = reactive([])
    matches_query: reactive[str | None] = reactive(None)
    current_index: reactive[int] = reactive(0)

    def __init__(self, limit: int, refresh: bool, cache_path: Path) -> None:
        super().__init__()
        self.limit = limit
        self.refresh = refresh
        self.cache_path = cache_path
        self.cache = load_cache(cache_path)
        self.input_active = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield ListView(id="list")
            yield SummaryView(id="summary")
        yield Input(placeholder="Search...", id="search_input")
        yield Footer()

    async def on_mount(self) -> None:
        await self.run_worker(self.load_data(), exclusive=True)

    async def load_data(self):
        try:
            feed = fetch_guardian()
            articles = parse_rss(feed)[: self.limit] if self.limit > 0 else parse_rss(feed)
        except Exception as exc:  # network or parsing errors
            self.exit(f"Failed to load feed: {exc}")
            return

        summaries: list[str] = []
        for article in articles:
            key = f"{article_key(article)}:farage"
            if not self.refresh and key in self.cache:
                summaries.append(self.cache[key])
                continue
            prompt = build_prompt(to_three_lines(article.summary, width=120))
            self.console.print(f"[cyan]Querying tgpt for:[/cyan] {article.title}")
            self.console.print(f"[dim]{prompt}[/dim]\n")
            response = call_tgpt(prompt)
            self.cache[key] = response
            summaries.append(response)
        save_cache(self.cache_path, self.cache)
        await self.post_message(self.Loaded(articles, summaries))

    def on_loaded(self, message: Loaded) -> None:
        self.articles = message.articles
        self.summaries = message.summaries
        list_view = self.query_one(ListView)
        list_view.clear()
        for idx, article in enumerate(self.articles):
            list_view.append(ListItem(Static(f"{idx+1}. {article.title}")))
        if self.articles:
            self.show_article(0)

    def show_article(self, index: int) -> None:
        if not self.articles:
            return
        index = max(0, min(index, len(self.articles) - 1))
        self.current_index = index
        self.query_one(ListView).index = index
        self.query_one(SummaryView).update_content(
            self.articles[index].title, self.summaries[index]
        )

    def action_quit(self) -> None:
        self.exit()

    def action_search(self) -> None:
        search_input = self.query_one("#search_input", Input)
        search_input.display = True
        search_input.value = ""
        search_input.focus()
        self.input_active = True

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

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.show_article(event.index)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        event.input.display = False
        self.input_active = False
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
        if event.key in {"j", "down"}:
            self.show_article(self.current_index + 1)
        elif event.key in {"k", "up"}:
            self.show_article(self.current_index - 1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Textual Guardian reader with Farage-styled summaries via tgpt.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max articles to load (default: 30)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and recompute summaries.")
    args = parser.parse_args(argv)

    app = GuardianApp(limit=args.limit, refresh=args.refresh, cache_path=CACHE_PATH)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
