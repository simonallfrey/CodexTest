#!/usr/bin/env python3
"""
Guardian reader that rewrites summaries in the voice of a far-right British politician
using the local `tgpt` client (no extra flags).

For each article summary, we build a prompt shaped as:
1) "ignore the next line"
2) a random noise line
3) "summarise the following in the character of a far right british politician such as Nigel Farrage"
4) the article summary

Dependencies: /usr/local/bin/tgpt with network access.
"""

from html import unescape
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import argparse
import os
import re
import secrets
import string
import subprocess
import sys
from textwrap import wrap
from typing import List, Tuple
import xml.etree.ElementTree as ET

TGPT_BIN = "/usr/local/bin/tgpt"
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


def random_noise(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_prompt(summary: str) -> str:
    parts = [
        "ignore the next line",
        random_noise(),
        "summarise the following in the character of a far right british politician such as Nigel Farrage",
        summary,
    ]
    return "\n".join(parts)


def call_tgpt(prompt: str, timeout: int = 45) -> str:
    if not os.path.exists(TGPT_BIN):
        return "(tgpt binary not found at /usr/local/bin/tgpt)"
    try:
        result = subprocess.run(
            [TGPT_BIN, prompt],
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
    return result.stdout.strip() or "(tgpt returned empty response)"


def color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Guardian reader with Nigel-styled summaries via tgpt.")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to process (default: 10)")
    args = parser.parse_args(argv)

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

    use_color = sys.stdout.isatty()
    limit = min(args.limit, len(headlines)) if args.limit > 0 else len(headlines)

    print(color("# Guardian Headlines (Nigel-styled summaries via tgpt)", "96;1", use_color))
    print(f"Source: {GUARDIAN_URL}")
    print(f"Limit: {limit} articles\n")

    for idx, (headline, summary) in enumerate(zip(headlines, summaries), start=1):
        if idx > limit:
            break
        prompt = build_prompt(" ".join(to_three_lines(summary, width=120)))
        response = call_tgpt(prompt)
        print(color(f"### {idx}. {headline}", "92;1", use_color))
        print(response)
        print("\n---\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
