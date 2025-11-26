#!/usr/bin/env python3
"""
Fetch BBC News headlines and render them as a short Markdown list.
If `bat` is available, the output is piped to it for nicer paging.
"""

from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import shutil
import subprocess
import sys
from typing import List, Tuple

BBC_NEWS_URLS = [
    "https://news.bbc.co.uk/",
    "https://www.bbc.co.uk/news",
]


class HeadlineParser(HTMLParser):
    """Collect text inside headline tags."""

    def __init__(self) -> None:
        super().__init__()
        self._active_tag: str | None = None
        self._current: list[str] = []
        self.headlines: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # attrs kept for interface
        if tag in {"h1", "h2", "h3"}:
            self._active_tag = tag
            self._current = []

    def handle_data(self, data: str) -> None:
        if self._active_tag:
            self._current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._active_tag == tag:
            text = "".join(self._current).strip()
            if text and text not in self.headlines:
                self.headlines.append(text)
            self._active_tag = None
            self._current = []


def fetch_first_available() -> Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BBCHeadlineFetcher/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }
    last_err: Exception | None = None
    for url in BBC_NEWS_URLS:
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=10) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                html = resp.read().decode(charset, errors="replace")
                return html, url
        except (HTTPError, URLError, TimeoutError) as exc:
            last_err = exc
    raise RuntimeError(f"Failed to fetch BBC News: {last_err}") from last_err


def render_markdown(headlines: List[str], source: str) -> str:
    lines: list[str] = ["# BBC News Headlines", ""]
    for idx, text in enumerate(headlines[:20], start=1):
        lines.append(f"- {idx}. {text}")
    lines.append("")
    lines.append(f"Source: {source}")
    return "\n".join(lines)


def show_with_bat(markdown: str) -> bool:
    bat = shutil.which("bat")
    if not bat:
        return False
    result = subprocess.run([bat, "-l", "md"], input=markdown.encode(), check=False)
    return result.returncode == 0


def main() -> int:
    try:
        html, url = fetch_first_available()
    except Exception as exc:  # network or decode errors
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    parser = HeadlineParser()
    parser.feed(html)
    headlines = parser.headlines
    if not headlines:
        sys.stderr.write("Error: No headlines found in the page.\n")
        return 1

    output = render_markdown(headlines, url)
    if not show_with_bat(output):
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
