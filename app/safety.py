# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Child-Safety Guardrails and Content Filtering for Sugar-AI."""

import re
from typing import Tuple, List, Dict, Any

SAFE_CHILD_RESPONSE = "I'm here to help you learn coding safely! Let's get back to writing fun code."

# Common PII Regex Patterns
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE)
PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b|\b\d{10,12}\b')
SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CREDIT_CARD_PATTERN = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')

# Blocked keyword patterns organized by category
BLOCKED_PATTERNS = {
    "profanity_and_abuse": [
        r'\b(?:fuck|fucking|fucked|shit|bitch|bastard|asshole|cunt|dick|pussy|fag|nigger|nigga|whore|slut)\b',
    ],
    "adult_and_nsfw": [
        r'\b(?:porn|pornography|xxx|erotic|hentai|sex|nude|nudity|penis|vagina|dildo|orgasm|intercourse)\b',
    ],
    "violence_and_harm": [
        r'\b(?:kill\s+(?:yourself|someone|people)|commit\s+suicide|suicide|self-harm|cut\s+myself|murder|how\s+to\s+(?:make|build)\s+a\s+bomb|(?:make|build)\s+a\s+bomb|terrorist|assassinate|shoot\s+someone|gun\s+violence)\b',
    ],
    "malicious_cyber": [
        r'\b(?:(?:create|make|write|build|develop)\s+(?:malware|ransomware|keylogger|virus|trojan|rootkit|spyware|botnet|exploit))\b',
        r'\b(?:malware|ransomware|keylogger|trojan)\b',
        r'\b(?:hack\s+(?:into|bank|website|account|server)|ddos\s+attack|steal\s+passwords|sql\s+injection\s+attack)\b',
    ],
}

COMPILED_BLOCKED_PATTERNS = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in BLOCKED_PATTERNS.items()
}


def redact_pii(text: str) -> str:
    """Redact Personally Identifiable Information (PII) like emails, phones, SSNs, credit cards."""
    if not text:
        return text

    redacted = text
    redacted = EMAIL_PATTERN.sub("[REDACTED EMAIL]", redacted)
    redacted = SSN_PATTERN.sub("[REDACTED SSN]", redacted)
    redacted = CREDIT_CARD_PATTERN.sub("[REDACTED CARD]", redacted)
    redacted = PHONE_PATTERN.sub("[REDACTED PHONE]", redacted)
    return redacted


def has_pii(text: str) -> bool:
    """Check if text contains PII."""
    if not text:
        return False
    return bool(
        EMAIL_PATTERN.search(text)
        or PHONE_PATTERN.search(text)
        or SSN_PATTERN.search(text)
        or CREDIT_CARD_PATTERN.search(text)
    )


def is_content_safe(text: str) -> Tuple[bool, str]:
    """Check whether content is safe for children.

    Returns:
        Tuple[bool, str]: (is_safe, reason)
    """
    if not text or not isinstance(text, str):
        return True, ""

    # Check for PII
    if has_pii(text):
        return False, "PII detected: Personal information is not allowed."

    # Check blocked categories
    for category, patterns in COMPILED_BLOCKED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                category_name = category.replace("_", " ").title()
                return False, f"Inappropriate content detected ({category_name})."

    return True, ""


def check_messages_safe(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Check if any message in a list of chat messages is unsafe."""
    if not messages:
        return True, ""

    for idx, msg in enumerate(messages):
        content = msg.get("content", "")
        is_safe, reason = is_content_safe(content)
        if not is_safe:
            return False, f"Message {idx+1}: {reason}"

    return True, ""


def get_safe_canned_response() -> str:
    """Return child-friendly canned safe response."""
    return SAFE_CHILD_RESPONSE
