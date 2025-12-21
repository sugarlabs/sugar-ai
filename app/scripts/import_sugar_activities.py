"""Fetch and sanitize the FLOSS manual "Make your own Sugar activities" and save
it as a plain Markdown file under `docs/` for RAG ingestion.

Run as: python -m app.scripts.import_sugar_activities
"""
from __future__ import annotations
import re
import os
import sys
from typing import Optional

import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://archive.flossmanuals.net/make-your-own-sugar-activities/index.html"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "docs", "Sugar_Activities.md")


def clean_text(text: str) -> str:
    # Normalize whitespace and remove repeated blank lines
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Try common content containers used by the archive site
    candidates = [
        soup.find(id="content"),
        soup.find("main"),
        soup.find("article"),
        soup.find("div", class_="book"),
    ]

    content = None
    for c in candidates:
        if c:
            content = c
            break

    if content is None:
        content = soup.body

    parts = []
    for el in content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(separator=" ", strip=True)
        if text:
            if el.name.startswith("h"):
                parts.append(f"## {text}")
            else:
                parts.append(text)

    md = "\n\n".join(parts)
    return clean_text(md)


def fetch_and_save(url: str = SOURCE_URL, out_path: Optional[str] = None) -> str:
    if out_path is None:
        out_path = os.path.abspath(OUT_PATH)

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    md = extract_content(resp.text)

    header = f"# Sugar Activities Guide (sourced from FLOSS Manu      al)\n\nSource: {url}\n\n"
    content = header + md
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    return out_path


def main():
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        docs_dir = os.path.join(project_root, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        out_file = os.path.join(docs_dir, "Sugar_Activities.md")
        path = fetch_and_save(url=SOURCE_URL, out_path=out_file)
        print(f"Saved Sugar activities guide to: {path}")
    except Exception as e:
        print(f"Error fetching or saving guide: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
