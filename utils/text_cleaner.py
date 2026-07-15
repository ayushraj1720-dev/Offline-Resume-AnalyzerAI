"""
utils/text_cleaner.py
Basic text cleaning utilities for ResumeSense AI.
Runs completely offline.
"""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove extra blank lines."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_sentences(text: str) -> list[str]:
    """
    Split text into simple sentences.
    Falls back to line-based splitting for resumes.
    """
    text = clean_text(text)

    sentences = re.split(r"[.!?]\s+|\n+", text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 2
    ]


def get_meaningful_tokens(text: str) -> list[str]:
    """
    Return lowercase meaningful words.
    """

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}", text.lower())

    stopwords = {
        "the","and","for","with","from","that","this",
        "have","has","had","are","was","were","will",
        "your","you","their","our","its","into","using",
        "use","used","also","than","then","very","more",
        "less","such","each","other","over","under"
    }

    return [
        t
        for t in tokens
        if t not in stopwords
    ]
