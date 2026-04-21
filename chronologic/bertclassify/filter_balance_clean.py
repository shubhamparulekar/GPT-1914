#!/usr/bin/env python3
"""filter_balance_clean.py

Provides normalize_text() used by run_deberta.py to pre-process text before
DeBERTa inference, matching the cleaning applied during dataset preparation.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize whitespace and unicode in *text*.

    - Apply NFC unicode normalization
    - Collapse runs of whitespace (including newlines) to a single space
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
