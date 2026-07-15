"""
utils/section_detector.py
=========================
Detect and extract sections from a resume (Education, Experience, Skills, etc.)
using local regex pattern matching.

WHY THIS MODULE EXISTS:
    Resumes have a standard structure with predictable section headers.
    Detecting these sections lets us:
    - Score "section completeness" (has Education? has Experience? etc.)
    - Extract skills from the Skills section specifically (higher precision)
    - Build a structured JSON/PDF report with proper grouping
    - Highlight which resume sections match a job description
    - Return formatted text for display in the UI

DETECTION STRATEGY:
    Use the SECTION_ALIASES from config.py (a dict of section names → regex patterns).
    Scan the resume text line-by-line, looking for header lines that match.
    Once a header is found, collect all text until the next header.
    This is simple, fast, transparent, and works 100% offline with no models.

EDGE CASES:
    - Resume with no sections? → return what was found, score it as incomplete
    - Two "Skills" headers? → merge them (we take first occurrence, but detect both)
    - Section with no content? → return empty dict for that section
    - All-caps vs Title Case vs mixed? → regex is case-insensitive, handles both
"""

import re
from typing import Optional

from config import SECTION_ALIASES
from utils.text_cleaner import normalize_whitespace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_resume_sections(resume_text: str) -> dict:
    """
    Detect and extract all known resume sections from raw text.

    Parameters
    ----------
    resume_text : str
        The full resume text as returned by pdf_extractor.py

    Returns
    -------
    dict with keys matching SECTION_ALIASES (e.g., "education", "experience", "skills").
    Each key maps to either:
        - str : the extracted text for that section
        - "" : section not found

    Example
    -------
        sections = detect_resume_sections(resume_text)
        education = sections["education"]
        skills = sections["skills"]
        if not sections["experience"]:
            print("Warning: no experience section detected")
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return {name: "" for name in SECTION_ALIASES.keys()}

    # Normalize: remove excess newlines but keep paragraphs
    text = resume_text.strip()

    # Split into lines for header detection
    lines = text.split("\n")

    # Find all section headers and their line numbers
    section_starts = _find_section_headers(lines)

    if not section_starts:
        # No sections detected — return empty result
        return {name: "" for name in SECTION_ALIASES.keys()}

    # Extract text between headers
    sections = {}
    for section_name in SECTION_ALIASES.keys():
        sections[section_name] = ""

    # For each detected header, extract text until next header
    for i, (line_num, detected_section_name) in enumerate(section_starts):
        start = line_num + 1  # text starts on next line after header
        end = len(lines)

        # If there's a next section, stop before it
        if i + 1 < len(section_starts):
            end = section_starts[i + 1][0]

        section_text = "\n".join(lines[start:end]).strip()
        if section_text:
            sections[detected_section_name] = section_text

    return sections


def get_detected_sections_summary(sections: dict) -> dict:
    """
    Return a summary of which sections were found and their lengths.

    Useful for the ATS scorer (section_completeness) and UI display.

    Parameters
    ----------
    sections : dict
        From detect_resume_sections()

    Returns
    -------
    dict with keys: section_name → {"found": bool, "word_count": int, "line_count": int}
    """
    summary = {}
    for section_name, text in sections.items():
        summary[section_name] = {
            "found": bool(text.strip()),
            "word_count": len(text.split()) if text else 0,
            "line_count": len(text.split("\n")) if text else 0,
        }
    return summary


def extract_contact_info(resume_text: str) -> dict:
    """
    Extract contact information (email, phone, LinkedIn, GitHub).

    Usually found in the Contact/Personal section, but we search the whole text
    since some resumes put links inline or at the top.

    Parameters
    ----------
    resume_text : str

    Returns
    -------
    dict with keys: email, phone, linkedin, github, website
    Each maps to a string (first match) or empty string if not found.
    """
    from utils.text_cleaner import extract_emails, extract_phone_numbers, extract_urls

    emails = extract_emails(resume_text)
    phones = extract_phone_numbers(resume_text)
    urls = extract_urls(resume_text)

    # Separate URLs by type
    linkedin = ""
    github = ""
    website = ""
    for url in urls:
        if "linkedin" in url.lower():
            linkedin = url
        elif "github" in url.lower():
            github = url
        else:
            website = url  # first non-LinkedIn, non-GitHub URL

    return {
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
        "linkedin": linkedin,
        "github": github,
        "website": website,
    }


def format_sections_for_display(sections: dict) -> str:
    """
    Format detected sections as readable text for Streamlit display.

    Parameters
    ----------
    sections : dict
        From detect_resume_sections()

    Returns
    -------
    str — formatted text with section headers and content
    """
    output = []
    for section_name, text in sections.items():
        if text.strip():
            output.append(f"\n{'─' * 60}")
            output.append(f"📌 {section_name.upper()}")
            output.append(f"{'─' * 60}")
            output.append(text)

    return "\n".join(output) if output else "[No sections detected]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_section_headers(lines: list[str]) -> list[tuple[int, str]]:
    """
    Scan lines and find resume section headers.

    Returns a list of tuples: (line_number, section_name)
    where line_number is the 0-indexed line with the header.

    Lines are matched against SECTION_ALIASES patterns (case-insensitive).
    """
    found_headers = []

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Skip if line is too short or is just numbers
        if len(line_lower) < 3 or line_lower.isdigit():
            continue

        # Try to match against each section's aliases
        for section_name, aliases in SECTION_ALIASES.items():
            for alias in aliases:
                # Check if this line is (almost) exactly an alias
                # Allow for trailing colons, bullets, etc.
                pattern = re.escape(alias) + r"[\s:•\-]*$"
                if re.match(pattern, line_lower):
                    found_headers.append((i, section_name))
                    break  # matched this line, move to next line

    return found_headers


def estimate_section_importance(resume_text: str, section_name: str, text: str) -> float:
    """
    Rough heuristic: how prominent is this section in the resume?

    Returns a 0-1 score based on:
        - Position in the resume (earlier is more important)
        - Length relative to total resume

    Used by the report to decide formatting/prominence.
    """
    if not text or not resume_text:
        return 0.0

    # Position score: earlier sections get higher score
    pos_in_resume = resume_text.find(text) / len(resume_text) if text in resume_text else 0.5
    position_score = 1.0 - pos_in_resume  # invert: earlier → higher

    # Length score: how much of the resume is this section?
    length_score = len(text) / len(resume_text) if resume_text else 0

    # Combine: weight position more heavily (it's a strong signal of importance)
    importance = 0.6 * position_score + 0.4 * length_score
    return min(importance, 1.0)  # cap at 1.0


def merge_similar_sections(sections: dict, threshold: float = 0.8) -> dict:
    """
    If two sections seem to contain nearly identical text (>threshold similarity),
    merge them into one.

    Used as a cleanup step if section detection found duplicates.

    Parameters
    ----------
    sections : dict
    threshold : float — cosine similarity threshold (0-1)

    Returns
    -------
    dict — sections with duplicates merged
    """
    # Simple string-based similarity: Jaccard similarity of word sets
    def jaccard_sim(text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    merged = dict(sections)

    section_names = list(sections.keys())
    for i in range(len(section_names)):
        for j in range(i + 1, len(section_names)):
            name1, name2 = section_names[i], section_names[j]
            text1, text2 = merged[name1], merged[name2]

            if text1 and text2:
                sim = jaccard_sim(text1, text2)
                if sim >= threshold:
                    # Merge: keep longer text, clear the shorter one
                    if len(text1) >= len(text2):
                        merged[name2] = ""
                    else:
                        merged[name1] = ""
                        merged[name2] = text2

    return merged
