"""
Dynamic document fetching system for Sugar-AI.
Fetches Sugar documentation from GitHub and converts to plain text for RAG.
"""

import os
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Documentation sources to fetch
DOCS_TO_FETCH = [
    {
        "url": "https://raw.githubusercontent.com/sugarlabs/sugar-docs/master/src/desktop-activity.md",
        "filename": "sugar-desktop-activity.txt"
    },
    {
        "url": "https://raw.githubusercontent.com/sugarlabs/sugar-docs/master/src/web-activity.md",
        "filename": "sugar-web-activity.txt"
    },
    {
        "url": "https://raw.githubusercontent.com/sugarlabs/sugar-docs/master/src/contributing.md",
        "filename": "sugar-contributing.txt"
    }
]


def strip_markdown_headers(text: str) -> str:
    """
    Convert markdown headers to plain text by removing # symbols.
    Converts "# Header" to "Header", "## Subheader" to "Subheader", etc.
    """
    # Replace markdown headers (# symbols) with plain text
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text


def markdown_to_plaintext(markdown_content: str) -> str:
    """
    Convert markdown content to clean plain text.
    Strips markdown headers and removes HTML tags.
    """
    # First strip markdown headers
    text = strip_markdown_headers(markdown_content)
    
    # Remove HTML tags
    text = remove_html_tags(text)
    
    # Clean up excessive whitespace
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def fetch_document(url: str, github_token: str = None) -> Dict[str, any]:
    """
    Fetch a document from a GitHub URL.
    
    Args:
        url: GitHub raw content URL
        github_token: Optional GitHub API token for authentication
    
    Returns:
        Dict with 'success', 'content', and 'error' keys
    """
    try:
        headers = {}
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        
        logger.info(f"Fetching document from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Successfully fetched: {url}")
            return {
                "success": True,
                "content": response.text,
                "error": None
            }
        elif response.status_code == 404:
            error_msg = f"Document not found (404): {url}"
            logger.error(error_msg)
            return {
                "success": False,
                "content": None,
                "error": error_msg
            }
        else:
            error_msg = f"Failed to fetch {url}: HTTP {response.status_code}"
            logger.error(error_msg)
            return {
                "success": False,
                "content": None,
                "error": error_msg
            }
    
    except requests.ConnectionError as e:
        error_msg = f"Network error fetching {url}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "content": None,
            "error": error_msg
        }
    except requests.Timeout:
        error_msg = f"Request timeout fetching {url}"
        logger.error(error_msg)
        return {
            "success": False,
            "content": None,
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error fetching {url}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "content": None,
            "error": error_msg
        }


def save_document(content: str, filename: str, docs_dir: str = "docs") -> Dict[str, any]:
    """
    Save a document to the docs directory.
    
    Args:
        content: Plain text content to save
        filename: Target filename
        docs_dir: Target directory (default: 'docs')
    
    Returns:
        Dict with 'success' and 'filepath' keys
    """
    try:
        # Ensure docs directory exists
        Path(docs_dir).mkdir(parents=True, exist_ok=True)
        
        filepath = os.path.join(docs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Saved document: {filepath}")
        return {
            "success": True,
            "filepath": filepath
        }
    
    except Exception as e:
        error_msg = f"Error saving {filename}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "filepath": None,
            "error": error_msg
        }


def fetch_all_docs(github_token: str = None, docs_to_fetch: List[Dict] = None) -> Dict[str, any]:
    """
    Fetch all Sugar documentation from GitHub and save as plain text files.
    
    Args:
        github_token: Optional GitHub API token for authentication
        docs_to_fetch: List of docs to fetch (default: DOCS_TO_FETCH)
    
    Returns:
        Dict containing fetch results and summary
    """
    if docs_to_fetch is None:
        docs_to_fetch = DOCS_TO_FETCH
    
    timestamp = datetime.now().isoformat()
    results = {
        "success": True,
        "timestamp": timestamp,
        "fetched_docs": [],
        "failed_docs": [],
        "errors": []
    }
    
    logger.info(f"Starting document fetch at {timestamp}")
    logger.info(f"GitHub token present: {bool(github_token)}")
    
    for doc in docs_to_fetch:
        url = doc["url"]
        filename = doc["filename"]
        
        # Fetch document
        fetch_result = fetch_document(url, github_token)
        
        if not fetch_result["success"]:
            results["failed_docs"].append(filename)
            results["errors"].append(fetch_result["error"])
            continue
        
        # Convert markdown to plain text
        plain_text = markdown_to_plaintext(fetch_result["content"])
        
        # Add header with source and timestamp
        header = f"# Fetched from {url} on {timestamp}\n\n"
        final_content = header + plain_text
        
        # Save document
        save_result = save_document(final_content, filename)
        
        if save_result["success"]:
            results["fetched_docs"].append(filename)
            logger.info(f"Successfully processed: {filename}")
        else:
            results["failed_docs"].append(filename)
            results["errors"].append(save_result.get("error", "Unknown error"))
    
    # Determine overall success
    if results["failed_docs"]:
        results["success"] = False
    
    return results


def main():
    """
    Main entry point for standalone script execution.
    Can be called as: python scripts/fetch_sugar_docs.py
    """
    # Get GitHub token from environment (optional)
    github_token = os.getenv("GITHUB_TOKEN", None)
    
    if github_token:
        logger.info("Using GitHub token from environment")
    else:
        logger.info("No GitHub token provided. Using unauthenticated requests (subject to rate limits)")
    
    # Fetch all documents
    results = fetch_all_docs(github_token=github_token)
    
    # Print summary
    total_docs = len(results["fetched_docs"]) + len(results["failed_docs"])
    
    print("\n" + "="*60)
    print("SUGAR-AI DOCUMENT FETCH SUMMARY")
    print("="*60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total documents attempted: {total_docs}")
    print(f"Successfully fetched: {len(results['fetched_docs'])}")
    print(f"Failed: {len(results['failed_docs'])}")
    
    if results["fetched_docs"]:
        print("\nFetched documents:")
        for doc in results["fetched_docs"]:
            print(f"  ✓ {doc}")
    
    if results["failed_docs"]:
        print("\nFailed documents:")
        for i, doc in enumerate(results["failed_docs"]):
            print(f"  ✗ {doc}")
            if i < len(results["errors"]):
                print(f"    Error: {results['errors'][i]}")
    
    print("\n" + "="*60)
    
    # Return summary message
    if results["success"]:
        print(f"Fetched {len(results['fetched_docs'])} docs successfully")
        return 0
    else:
        print(f"Fetch completed with {len(results['failed_docs'])} errors")
        return 1


if __name__ == "__main__":
    exit(main())
