"""
Fetch and extract the practical instructional sections for creating Sugar activities
from FLOSS Manuals, and write to docs/Sugar_Activities.md for RAG ingestion.

Usage: python scripts/import_sugar_activities.py
"""

import os
import sys
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://archive.flossmanuals.net/make-your-own-sugar-activities/"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "Sugar_Activities.md")

LICENSE_HEADER = (
    "# Sugar Activities: Complete Guide\n\n"
    "**Source:** FLOSS Manuals – “Make Your Own Sugar Activities”  \n"
    "**Authors:** James D. Simmons and contributors  \n"
    "**License:** GNU General Public License v2 or later (GPL-2.0-or-later)  \n"
    f"**Original source URL:** {BASE_URL}\n\n"
)

CHAPTERS = [
    ("INTRODUCTION", "index"),
    ("WHAT IS SUGAR?", "what-is-sugar"),
    ("WHAT IS A SUGAR ACTIVITY?", "what-is-a-sugar-activity"),
    ("WHAT DO I NEED TO KNOW TO WRITE A SUGAR ACTIVITY?", "what-do-i-need-to-know-to-write-a-sugar-activity"),
    ("PROGRAMMING", None),  # Section header only
    ("SETTING UP A DEVELOPMENT ENVIRONMENT", "setting-up-a-development-environment"),
    ("CREATING YOUR FIRST ACTIVITY", "creating-your-first-activity"),
    ("A STANDALONE PYTHON PROGRAM FOR READING ETEXTS", "a-standalone-python-program-for-reading-etexts"),
    ("INHERIT FROM SUGAR3.ACTIVITY.ACTIVITY", "inherit-from-sugar3-activity-activity"),
    ("PACKAGE THE ACTIVITY", "package-the-activity"),
    ("ADD REFINEMENTS", "add-refinements"),
    ("ADD YOUR ACTIVITY CODE TO VERSION CONTROL", "add-your-activity-code-to-version-control"),
    ("GOING INTERNATIONAL WITH POOTLE", "going-international-with-pootle"),
    ("DISTRIBUTE YOUR ACTIVITY", "distribute-your-activity"),
    ("DEBUGGING SUGAR ACTIVITIES", "debugging-sugar-activities"),
]

def html_to_markdown(soup):
    lines = []
    for el in soup.descendants:
        if el.name in ["h1", "h2", "h3", "h4"]:
            level = int(el.name[1])
            text = el.get_text(strip=True)
            if text:
                lines.append(f"{'#' * level} {text}\n")
        elif el.name == "p":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(text + "\n")
        elif el.name == "li":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(f"- {text}")
    md = "\n".join(lines)
    md = '\n'.join([line.rstrip() for line in md.splitlines()])
    while '\n\n\n' in md:
        md = md.replace('\n\n\n', '\n\n')
    return md.strip()

def fetch_chapter(title, slug):
    if slug == "index":
        url = f"{BASE_URL}index.html"
    else:
        url = f"{BASE_URL}{slug}.html"
    print(f"Fetching: {title}")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            print(f"WARNING: Skipping missing chapter: {title} ({url})")
            return None
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="book")
            or soup.body
        )
        if not content:
            print(f"WARNING: No main content found for chapter: {title} ({url})")
            return None
        md = html_to_markdown(content)
        return f"## {title}\n\n{md}\n"
    except Exception as e:
        print(f"WARNING: Error fetching chapter {title} ({url}): {e}")
        return None

def main():
    all_md = [LICENSE_HEADER]
    for title, slug in CHAPTERS:
        if slug is None:
            print(f"Writing section header: {title}")
            all_md.append(f"\n## {title}\n")
            continue
        chapter_md = fetch_chapter(title, slug)
        if chapter_md:
            all_md.append(chapter_md)
        else:
            print(f"Skipped: {title}")
    # Print total chapters collected after all_md is populated
    print("Total chapters collected:", len(all_md))
    out_path = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(all_md).strip() + "\n")
    print(f"\nSaved complete Sugar Activities documentation to: {out_path}")

if __name__ == "__main__":
    main()
