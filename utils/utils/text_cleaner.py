"""
utils/text_cleaner.py
=====================
Text preprocessing and normalization for ResumeSense AI.

WHY THIS MODULE EXISTS:
    Raw PDF text is messy. Before any NLP module (skill extractor, ATS scorer,
    JD matcher) touches the text, it needs to be normalized — lowercased,
    tokenized, stripped of stopwords, and cleaned of PDF artifacts like
    page numbers, URLs, and special characters.

    Centralizing all cleaning logic here means:
    - Every downstream module gets identically preprocessed text.
    - If we tweak cleaning logic, it improves the whole pipeline at once.
    - Unit testing is easy (pure functions, no side effects).

DESIGN PHILOSOPHY:
    We offer MULTIPLE levels of cleaning so each module can pick what it needs:
    - `clean_for_display()`    → light clean, keeps structure, for UI display
    - `clean_for_matching()`   → aggressive clean, for TF-IDF / skill matching
    - `tokenize()`             → word list, for frequency analysis
    - `remove_stopwords()`     → filters noise words from a token list
    - `extract_sentences()`    → sentence list, for readability scoring

All functions are pure (input → output, no state), offline, and fast.
NLTK data (stopwords, punkt tokenizer) is downloaded once and cached locally.
"""

import re
import string
from typing import Union

import nltk

# ---------------------------------------------------------------------------
# NLTK one-time data download
# ---------------------------------------------------------------------------
# These small data files (~few hundred KB each) are downloaded once to
# ~/nltk_data/ and reused on every subsequent run — fully offline after that.
# We wrap in try/except so the app doesn't crash if there's no internet
# on the judge's machine (falls back to a simple regex tokenizer).

def _ensure_nltk_data():
    """Download required NLTK datasets if not already cached."""
    packages = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }
    for path, package in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass  # No internet — we'll use fallback tokenizer below

_ensure_nltk_data()

# ---------------------------------------------------------------------------
# Stopwords setup
# ---------------------------------------------------------------------------
# We combine NLTK's English stopwords with a custom list of resume-specific
# noise words (e.g. "responsible", "duties") that add no signal for matching.

_RESUME_NOISE_WORDS = {
    "responsible", "responsibilities", "duties", "including", "worked",
    "work", "working", "job", "role", "position", "company", "team",
    "also", "well", "using", "used", "use", "various", "etc", "per",
    "within", "across", "along", "via", "able", "ability", "good",
    "strong", "excellent", "great", "highly", "seeking", "looking",
    "motivated", "passionate", "dynamic", "results", "driven",
}

try:
    from nltk.corpus import stopwords as nltk_stopwords
    _STOPWORDS = set(nltk_stopwords.words("english")) | _RESUME_NOISE_WORDS
except Exception:
    # Fallback: a minimal hardcoded stopword list
    _STOPWORDS = {
        "i", "me", "my", "we", "our", "you", "he", "she", "it", "they",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "a", "an", "the", "and", "but", "or", "for",
        "in", "on", "at", "to", "of", "with", "by", "from", "up", "as",
    } | _RESUME_NOISE_WORDS


# ---------------------------------------------------------------------------
# Public API — Level 1: Light cleaning (for display)
# ---------------------------------------------------------------------------

def clean_for_display(text: str) -> str:
    """
    Light cleaning that preserves the resume's readable structure.

    Use this when showing extracted text in the Streamlit UI so the user
    can verify their resume was read correctly.

    Removes:
        - Non-printable / control characters
        - Excessive blank lines (3+ → 1)
        - Leading/trailing whitespace per line

    Preserves:
        - Capitalization, punctuation, line breaks, paragraph structure
    """
    if not text:
        return ""

    # Remove non-printable characters (keep newline \n and tab \t)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00C0-\u024F]", " ", text)

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]

    # Collapse 3+ consecutive blank lines into 1
    result, blanks = [], 0
    for line in lines:
        if line == "":
            blanks += 1
            if blanks <= 1:
                result.append("")
        else:
            blanks = 0
            result.append(line)

    return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# Public API — Level 2: Aggressive cleaning (for NLP matching)
# ---------------------------------------------------------------------------

def clean_for_matching(text: str) -> str:
    """
    Aggressive cleaning that produces a normalized string for NLP tasks
    like TF-IDF vectorization, skill matching, and keyword density scoring.

    Steps:
        1. Lowercase everything
        2. Remove URLs (http://..., www....)
        3. Remove email addresses
        4. Remove phone numbers
        5. Remove page numbers (standalone digits)
        6. Remove punctuation (keeps hyphens inside words: "full-stack")
        7. Collapse multiple spaces/newlines into single space
        8. Strip

    Returns a single flat string (no line breaks).
    """
    if not text:
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove phone numbers (various formats)
    text = re.sub(r"[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]", " ", text)

    # Remove standalone numbers (page numbers, years when isolated)
    text = re.sub(r"\b\d{1,4}\b", " ", text)

    # Remove punctuation except hyphens between word chars (e.g. full-stack)
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)  # lone hyphens → space
    text = text.translate(
        str.maketrans(string.punctuation.replace("-", ""), " " * (len(string.punctuation) - 1))
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Public API — Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str, clean: bool = True) -> list[str]:
    """
    Split text into a list of word tokens.

    Parameters
    ----------
    text  : str — input text (raw or pre-cleaned)
    clean : bool — if True, runs clean_for_matching() first

    Returns
    -------
    list[str] — lowercase word tokens (no punctuation)

    Uses NLTK's punkt tokenizer if available, otherwise splits on whitespace.
    """
    if not text:
        return []

    if clean:
        text = clean_for_matching(text)

    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        # Fallback: simple whitespace split
        tokens = text.split()

    # Keep only alphabetic tokens and hyphenated terms (e.g. "full-stack")
    tokens = [t for t in tokens if re.match(r"^[a-z][a-z\-]*[a-z]$", t) or
              (len(t) > 1 and t.isalpha())]

    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Filter stopwords from a token list.

    Parameters
    ----------
    tokens : list[str] — from tokenize()

    Returns
    -------
    list[str] — tokens with stopwords removed
    """
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def get_meaningful_tokens(text: str) -> list[str]:
    """
    Convenience function: tokenize + remove stopwords in one call.

    This is what most modules (skill_extractor, ats_scorer) will use.
    """
    return remove_stopwords(tokenize(text, clean=True))


# ---------------------------------------------------------------------------
# Public API — Sentence extraction
# ---------------------------------------------------------------------------

def extract_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Used by:
        - ats_scorer.py  → average sentence length (readability)
        - strength_weakness.py → action verb detection per sentence

    Returns
    -------
    list[str] — non-empty sentences, stripped
    """
    if not text:
        return []

    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on period/exclamation/question mark
        sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Public API — Utility helpers
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def extract_emails(text: str) -> list[str]:
    """Extract all email addresses from text (for contact info detection)."""
    return re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)


def extract_phone_numbers(text: str) -> list[str]:
    """Extract phone-number-like patterns from text (for contact info detection)."""
    return re.findall(r"[\+\(]?[0-9][0-9\s\-\(\)\.]{7,}[0-9]", text)


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text (LinkedIn, GitHub, portfolio links)."""
    return re.findall(r"https?://\S+|www\.\S+", text)


def word_count(text: str) -> int:
    """Return approximate word count of raw text."""
    return len(text.split()) if text else 0


def char_count(text: str) -> int:
    """Return character count of raw text."""
    return len(text) if text else 0
