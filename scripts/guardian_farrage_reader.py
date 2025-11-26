#!/usr/bin/env python3
"""
Guardian headlines reader rendered with Textual.

Behavior
- Fetch Guardian RSS headlines and summaries.
- Build a Farage-style summary via a single tgpt prompt per article (cached).
- TUI flow: overview page with all titles; selecting shows a single-article view.
- Navigation: arrows or j/k move; Enter/space opens; gg returns to overview; G jumps to last article.
- Search: "/" to enter a query; "n"/"N" to jump next/previous match.

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
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from textwrap import wrap
from typing import Dict, Iterable, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, ListItem, ListView, LoadingIndicator, Static

# Configuration defaults (overridable by env/CLI)
ENV_TGPT_BIN = os.getenv("TGPT_BIN", "/usr/local/bin/tgpt")
ENV_FEED_URL = os.getenv("GUARDIAN_FEED_URL", "https://www.theguardian.com/international/rss")
ENV_CACHE = Path(os.getenv("GUARDIAN_CACHE_PATH", ".guardian_farage_cache.json"))
DEFAULT_LIMIT = 30
DEFAULT_NOISE = 80
DEFAULT_WRAP = 120
PROMPT_INSTRUCTION = (
    "summarise the following in the character of a far right british politician such as Nigel Farrage; "
    "serious and direct; no jokes or memes; no meta talk; no quotation marks; keep it concise (3-5 sentences); "
    "no preamble—start directly with the summary"
)


# HTML stripping helper
class PlaintextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.chunks.append(data)

    def get_text(self) -> str:
        text = " ".join(self.chunks)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def strip_html(html: str) -> str:
    parser = PlaintextParser()
    parser.feed(html)
    return parser.get_text()


@dataclass(frozen=True)
class Article:
    title: str
    summary: str


# Data helpers
def fetch_guardian(url: str, timeout: int = 10) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GuardianHeadlineFetcher/1.0)",
        "Accept": "application/rss+xml, text/xml;q=0.9, */*;q=0.8",
    }
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def parse_rss(xml_text: str) -> List[Article]:
    root = ET.fromstring(xml_text)
    articles: List[Article] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        description = item.findtext("description") or ""
        if not title:
            continue
        articles.append(Article(title=strip_html(title), summary=strip_html(description)))
    return articles


def to_three_lines(text: str, width: int = 80) -> str:
    if not text:
        return "(no summary available)"
    wrapped = wrap(text, width=width)
    return " ".join(wrapped[:3])


def random_noise(length: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_prompt(summary: str, noise_len: int) -> str:
    return "\n".join(
        [
            "ignore the next line",
            random_noise(noise_len),
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


def call_tgpt(prompt: str, bin_path: str, timeout: int = 45) -> str:
    if not os.path.exists(bin_path):
        return "(tgpt binary not found at provided path)"
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


# Messages for background updates
class ArticlesLoaded(Message):
    def __init__(self, articles: list[Article]) -> None:
        self.articles = articles
        super().__init__()


class SummaryReady(Message):
    def __init__(self, index: int, summary: str) -> None:
        self.index = index
        self.summary = summary
        super().__init__()


# Screens
class HeadlinesScreen(Screen):
    BINDINGS = [
        ("q", "app.quit", "Quit"),
        ("/", "search", "Search"),
        ("n", "search_next", "Next"),
        ("N", "search_prev", "Prev"),
        ("G", "last_article", "Last"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.list_view = ListView(id="headlines")
        yield self.list_view
        yield Input(placeholder="Search...", id="search_input", classes="hidden")
        yield Footer()

    def on_mount(self) -> None:
        self.matches_query: str | None = None
        self.pending_g = False

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.index
        # index corresponds to article index
        self.app.push_screen(ArticleScreen(index))

    def action_last_article(self) -> None:
        if self.app.articles:
            self.app.push_screen(ArticleScreen(len(self.app.articles) - 1))

    def action_search(self) -> None:
        search_input = self.query_one("#search_input", Input)
        search_input.remove_class("hidden")
        search_input.value = ""
        search_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        event.input.add_class("hidden")
        if not query:
            return
        self.matches_query = query
        self._jump_match(direction=1)

    def action_search_next(self) -> None:
        self._jump_match(direction=1)

    def action_search_prev(self) -> None:
        self._jump_match(direction=-1)

    def _jump_match(self, direction: int) -> None:
        if not self.matches_query or not self.app.articles:
            return
        found = find_match(
            self.matches_query,
            [a.title for a in self.app.articles],
            [self.app.summaries.get(i, "") for i in range(len(self.app.articles))],
            self.app.current_index,
            direction=direction,
        )
        if found is not None:
            self.app.current_index = found
            self.app.push_screen(ArticleScreen(found))

    def on_key(self, event: events.Key) -> None:
        if event.key == "g":
            if self.pending_g:
                # gg -> go to first article
                if self.app.articles:
                    self.app.push_screen(ArticleScreen(0))
                self.pending_g = False
            else:
                self.pending_g = True
            return
        self.pending_g = False

        if event.key in {"j", "down"}:
            if self.app.articles:
                next_idx = min(len(self.app.articles) - 1, self.app.current_index + 1)
                self.app.current_index = next_idx
                self.app.push_screen(ArticleScreen(next_idx))
        elif event.key in {"k", "up"}:
            if self.app.articles:
                prev_idx = max(0, self.app.current_index - 1)
                self.app.current_index = prev_idx
                self.app.push_screen(ArticleScreen(prev_idx))


class ArticleScreen(Screen):
    BINDINGS = [
        ("q", "pop_screen", "Back"),
        ("esc", "pop_screen", "Back"),
        ("j", "next_article", "Next"),
        ("k", "prev_article", "Prev"),
        ("G", "last_article", "Last"),
        ("g", "maybe_top", "Top"),
        ("/", "search", "Search"),
        ("n", "search_next", "Next"),
        ("N", "search_prev", "Prev"),
    ]

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.pending_g = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self.body = Static()
        yield self.body
        yield Footer()

    def on_show(self) -> None:
        self.pending_g = False
        self.render_article()

    def render_article(self) -> None:
        articles = self.app.articles
        if not articles:
            self.body.update("No articles loaded.")
            return
        idx = max(0, min(self.index, len(articles) - 1))
        self.app.current_index = idx
        title = articles[idx].title
        summary = self.app.summaries.get(idx, "(summary loading...)")
        self.body.update(f"[b]{title}[/b]\n\n{summary}")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_next_article(self) -> None:
        if self.app.articles and self.index + 1 < len(self.app.articles):
            self.index += 1
            self.render_article()

    def action_prev_article(self) -> None:
        if self.app.articles and self.index > 0:
            self.index -= 1
            self.render_article()
        elif self.index == 0:
            self.app.pop_screen()

    def action_last_article(self) -> None:
        if self.app.articles:
            self.index = len(self.app.articles) - 1
            self.render_article()

    def action_maybe_top(self) -> None:
        if self.pending_g:
            if self.app.articles:
                self.index = 0
                self.render_article()
            self.pending_g = False
        else:
            self.pending_g = True

    def action_search(self) -> None:
        # delegate to headlines search (push screen to handle search input)
        self.app.pop_screen()
        self.app.push_screen("headlines")
        headlines = self.app.get_screen("headlines")
        if isinstance(headlines, HeadlinesScreen):
            headlines.action_search()

    def action_search_next(self) -> None:
        self._jump(direction=1)

    def action_search_prev(self) -> None:
        self._jump(direction=-1)

    def _jump(self, direction: int) -> None:
        query = self.app.matches_query
        if not query or not self.app.articles:
            return
        found = find_match(
            query,
            [a.title for a in self.app.articles],
            [self.app.summaries.get(i, "") for i in range(len(self.app.articles))],
            self.app.current_index,
            direction=direction,
        )
        if found is not None:
            self.index = found
            self.render_article()

    def on_key(self, event: events.Key) -> None:
        if event.key == "g":
            self.action_maybe_top()
        else:
            self.pending_g = False


# Application
class GuardianApp(App[None]):
    CSS_PATH = None  # inline CSS used
    SCREENS = {"headlines": HeadlinesScreen}

    articles: reactive[list[Article]] = reactive([])
    summaries: reactive[Dict[int, str]] = reactive({})
    matches_query: reactive[str | None] = reactive(None)
    current_index: reactive[int] = reactive(0)

    def __init__(
        self,
        feed_url: str,
        tgpt_bin: str,
        cache_path: Path,
        noise_len: int,
        wrap_width: int,
        refresh: bool,
        quiet: bool,
        cache_only: bool,
    ) -> None:
        super().__init__()
        self.feed_url = feed_url
        self.tgpt_bin = tgpt_bin
        self.cache_path = cache_path
        self.noise_len = noise_len
        self.wrap_width = wrap_width
        self.refresh = refresh
        self.quiet = quiet
        self.cache_only = cache_only
        self.cache: Dict[str, str] = load_cache(cache_path)

    def compose(self) -> ComposeResult:
        yield HeadlinesScreen()

    def on_mount(self) -> None:
        # start background fetch immediately
        self.call_later(self._show_loading)
        self.run_worker(self._load_feed(), thread=True, exclusive=True)

    def _show_loading(self) -> None:
        screen = self.get_screen("headlines")
        if isinstance(screen, HeadlinesScreen):
            screen.list_view.clear()
            screen.list_view.append(ListItem(LoadingIndicator()))

    def on_articles_loaded(self, message: ArticlesLoaded) -> None:
        self.articles = message.articles
        self.summaries = {idx: self.cache.get(f"{article_key(a)}:farage", "") for idx, a in enumerate(self.articles)}
        screen = self.get_screen("headlines")
        if isinstance(screen, HeadlinesScreen):
            screen.list_view.clear()
            for idx, article in enumerate(self.articles):
                screen.list_view.append(ListItem(Static(f"{idx+1}. {article.title}")))
        # start summaries in background
        self.run_worker(self._generate_summaries(), thread=True, exclusive=True)

    def on_summary_ready(self, message: SummaryReady) -> None:
        self.summaries = {**self.summaries, message.index: message.summary}
        cache_key = f"{article_key(self.articles[message.index])}:farage"
        self.cache[cache_key] = message.summary
        save_cache(self.cache_path, self.cache)
        # if article screen open and same index, refresh
        current = self.screen
        if isinstance(current, ArticleScreen) and current.index == message.index:
            current.render_article()

    async def _load_feed(self):
        try:
            feed = fetch_guardian(url=self.feed_url)
            all_articles = parse_rss(feed)
        except Exception as exc:
            self.exit(f"Failed to load feed: {exc}")
            return
        articles = all_articles
        self.post_message(ArticlesLoaded(articles))

    async def _generate_summaries(self):
        if not self.articles:
            return
        for idx, article in enumerate(self.articles):
            key = f"{article_key(article)}:farage"
            if not self.refresh and key in self.cache:
                summary = self.cache[key]
            elif self.cache_only:
                summary = "(cache-only mode: no summary available)"
            else:
                prompt = build_prompt(to_three_lines(article.summary, width=self.wrap_width), noise_len=self.noise_len)
                if not self.quiet:
                    self.console.print(f"[cyan]Querying tgpt for:[/cyan] {article.title}")
                    self.console.print(f"[dim]{prompt}[/dim]\n")
                summary = call_tgpt(prompt, bin_path=self.tgpt_bin)
            self.post_message(SummaryReady(idx, summary))

    def action_quit(self) -> None:
        self.exit()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Textual Guardian reader with Farage-styled summaries via tgpt.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max articles to load (default: 30)")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and recompute summaries.")
    parser.add_argument("--tgpt-bin", default=ENV_TGPT_BIN, help="Path to tgpt binary (env: TGPT_BIN).")
    parser.add_argument("--feed-url", default=ENV_FEED_URL, help="Guardian RSS feed URL (env: GUARDIAN_FEED_URL).")
    parser.add_argument("--cache-path", type=Path, default=ENV_CACHE, help="Cache file path (env: GUARDIAN_CACHE_PATH).")
    parser.add_argument("--noise-len", type=int, default=DEFAULT_NOISE, help="Noise line length (default: 80).")
    parser.add_argument("--wrap", type=int, default=DEFAULT_WRAP, help="Wrap width for tgpt prompt summaries (default: 120).")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose prompt logging to stderr.")
    parser.add_argument("--cache-only", action="store_true", help="Do not call tgpt; use cache or placeholder text.")
    args = parser.parse_args(argv)

    app = GuardianApp(
        feed_url=args.feed_url,
        tgpt_bin=args.tgpt_bin,
        cache_path=args.cache_path,
        noise_len=args.noise_len,
        wrap_width=args.wrap,
        refresh=args.refresh,
        quiet=args.quiet,
        cache_only=args.cache_only,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
