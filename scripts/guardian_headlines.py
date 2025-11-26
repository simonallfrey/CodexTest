#!/usr/bin/env python3
"""
Fetch Guardian headlines via RSS and show them with short, direct summaries.
Interactive paging is handled in Python (no external pager needed).
"""

from html import unescape
from urllib.error import HTTPError, URLError
import os
import re
import shutil
import sys
from urllib.request import Request, urlopen
from textwrap import wrap
from typing import List, Tuple
import xml.etree.ElementTree as ET

GUARDIAN_URL = "https://www.theguardian.com/international/rss"


def fetch_guardian() -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GuardianHeadlineFetcher/1.0)",
        "Accept": "application/rss+xml, text/xml;q=0.9, */*;q=0.8",
    }
    request = Request(GUARDIAN_URL, headers=headers)
    with urlopen(request, timeout=10) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def clean_text(html_text: str) -> str:
    # Strip tags and decode entities for a plain summary
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_rss(xml_text: str) -> Tuple[List[str], List[str]]:
    root = ET.fromstring(xml_text)
    headlines: list[str] = []
    summaries: list[str] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        description = item.findtext("description") or ""
        if title:
            headlines.append(clean_text(title))
            summaries.append(clean_text(description))
    return headlines, summaries


def to_three_lines(text: str, width: int = 80) -> List[str]:
    if not text:
        return ["(no summary available)"]
    wrapped = wrap(text, width=width)
    return wrapped[:3] or ["(no summary available)"]


def color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def render_pages(headlines: List[str], summaries: List[str]) -> list[str]:
    term = shutil.get_terminal_size(fallback=(80, 24))
    # Leave room for prompts at the bottom
    page_lines = max(10, term.lines - 2)
    quick_screen_lines = max(5, page_lines - 6)
    quick_width = max(40, term.columns - 4)
    detail_width = max(40, term.columns - 6)
    use_color = sys.stdout.isatty()

    pages: list[list[str]] = []

    # Quick Scan page
    qs: list[str] = []
    qs.append(color("# Guardian Headlines", "96;1", use_color))
    qs.append("")
    qs.append(color("## Quick Scan (one screen)", "93;1", use_color))
    for idx, title in enumerate(headlines[:quick_screen_lines], start=1):
        truncated = title if len(title) <= quick_width else title[: quick_width - 1] + "…"
        qs.append(f"- {idx}. {truncated}")
    qs.append("")
    qs.append(f"Source: {GUARDIAN_URL}")
    if len(qs) < page_lines:
        qs.extend([""] * (page_lines - len(qs)))
    pages.append(qs)

    # Detail pages: one per article
    for idx, title in enumerate(headlines, start=1):
        summary = summaries[idx - 1] if idx - 1 < len(summaries) else ""
        tri = to_three_lines(summary, width=detail_width)
        block: list[str] = []
        block.append(color(f"### {idx}. {title}", "92;1", use_color))
        block.append("")
        block.extend([color("> " + line, "90", use_color) for line in tri])
        block.append("")
        block.append("---")
        if len(block) < page_lines:
            block.extend([""] * (page_lines - len(block)))
        pages.append(block)

    return ["\n".join(p) for p in pages]


def main() -> int:
    try:
        feed = fetch_guardian()
    except (HTTPError, URLError, TimeoutError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Error: unexpected failure: {exc}\n")
        return 1

    try:
        headlines, summaries = parse_rss(feed)
    except Exception as exc:
        sys.stderr.write(f"Error: failed to parse RSS feed: {exc}\n")
        return 1

    if not headlines:
        sys.stderr.write("Error: No headlines found in the Guardian feed.\n")
        return 1

    pages = render_pages(headlines, summaries)

    # If stdout is not a TTY, dump everything at once
    if not sys.stdout.isatty():
        sys.stdout.write("\n\f\n".join(pages))
        return 0

    def read_key() -> str:
        # Single-key reader; falls back to newline if stdin is not a TTY
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

    total = len(pages)
    idx = 0
    while True:
        sys.stdout.write("\033[2J\033[H")  # clear screen
        sys.stdout.write(pages[idx])
        sys.stdout.write(
            f"\n\n-- Page {idx + 1}/{total} -- [Enter/space/n/j: next, p/k: prev, q: quit] "
        )
        sys.stdout.flush()

        try:
            cmd = read_key()
        except EOFError:
            break

        if not cmd:
            continue
        if cmd in {"q", "Q"}:
            break
        if cmd in {"p", "k", "P", "K"}:
            idx = max(0, idx - 1)
            continue
        # default: next
        if idx + 1 >= total:
            break
        idx += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
