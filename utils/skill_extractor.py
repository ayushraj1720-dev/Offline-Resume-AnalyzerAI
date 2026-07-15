"""
utils/skill_extractor.py
========================
Extract skills from resume text using local AI and keyword matching.

WHY THIS MODULE EXISTS:
    Identifying which skills a candidate has is core to ATS matching,
    resume scoring, and the "missing skills" feature. We extract skills by:
    1. Matching resume text against a local skill taxonomy (JSON file).
    2. Using semantic similarity (sentence-transformers) for fuzzy matches.
    3. Falling back to TF-IDF if semantic matching is unavailable.
    4. Returning ranked skill matches with confidence scores.

SKILL TAXONOMY:
    We maintain a JSON file (models/skill_taxonomy.json) with known skills
    grouped by category (Programming Languages, Frameworks, Cloud, etc.).
    This is 100% local, version-controlled, and offline.

DETECTION METHODS (tried in order):
    1. Exact keyword match (case-insensitive) — highest confidence
    2. Semantic similarity via sentence-transformers (if available) — medium-high confidence
    3. Fuzzy token overlap — fallback, lower confidence

SCORING:
    Each detected skill gets a confidence score (0-1) based on:
    - How the skill was matched (exact > semantic > fuzzy)
    - Frequency in the resume (more mentions = higher confidence)
    - Position in resume (Skills section > Experience > other)
    - Context (appears near action verbs = higher confidence)
"""

import json
import re
from typing import Optional, Union

from config import SKILL_TAXONOMY_PATH, USE_SEMANTIC_MATCHING, EMBEDDING_MODEL_CACHE_DIR
from utils.text_cleaner import (
    clean_for_matching,
    tokenize,
    remove_stopwords,
    extract_sentences,
)


# ---------------------------------------------------------------------------
# Global state: cached skill taxonomy and embedding model
# ---------------------------------------------------------------------------

_SKILL_TAXONOMY = None
_EMBEDDING_MODEL = None


def _load_skill_taxonomy() -> dict:
    """
    Load skill taxonomy from JSON file.

    Returns dict with structure:
        {
            "programming": ["python", "java", ...],
            "frameworks": ["django", "react", ...],
            ...
        }

    If file doesn't exist, returns a minimal fallback taxonomy.
    """
    global _SKILL_TAXONOMY

    if _SKILL_TAXONOMY is not None:
        return _SKILL_TAXONOMY

    try:
        with open(SKILL_TAXONOMY_PATH, "r") as f:
            _SKILL_TAXONOMY = json.load(f)
    except FileNotFoundError:
        # Fallback: minimal taxonomy for when the JSON hasn't been created yet
        _SKILL_TAXONOMY = _get_fallback_taxonomy()

    return _SKILL_TAXONOMY


def _get_fallback_taxonomy() -> dict:
    """
    Fallback skill taxonomy used if models/skill_taxonomy.json is missing.

    This ensures the app doesn't crash even if the taxonomy file is lost.
    It's a minimal set covering the most common skills.
    """
    return {
        "programming_languages": [
            "python", "java", "javascript", "typescript", "c++", "c#", "go",
            "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
        ],
        "web_frameworks": [
            "django", "flask", "fastapi", "react", "vue", "angular",
            "node.js", "express", "spring", "asp.net", "laravel",
        ],
        "databases": [
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            "dynamodb", "cassandra", "oracle", "firebase",
        ],
        "cloud_platforms": [
            "aws", "azure", "gcp", "heroku", "digitalocean", "linode",
        ],
        "devops_tools": [
            "docker", "kubernetes", "jenkins", "gitlab ci", "github actions",
            "terraform", "ansible", "prometheus", "grafana",
        ],
        "data_tools": [
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
            "spark", "hadoop", "tableau", "powerbi", "sql",
        ],
        "soft_skills": [
            "leadership", "communication", "problem-solving", "teamwork",
            "project management", "agile", "scrum", "critical thinking",
        ],
    }


def _load_embedding_model():
    """
    Load sentence-transformers embedding model for semantic skill matching.

    Downloaded once from HuggingFace, cached locally in EMBEDDING_MODEL_CACHE_DIR.
    On subsequent runs (or if no internet), the cached model is used.
    Returns None if model can't be loaded (fallback to TF-IDF).
    """
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    if not USE_SEMANTIC_MATCHING:
        return None

    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL_NAME

        _EMBEDDING_MODEL = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=EMBEDDING_MODEL_CACHE_DIR,
        )
        return _EMBEDDING_MODEL
    except Exception as e:
        # No internet, model not cached, or import failed — fallback to TF-IDF
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_skills_from_resume(resume_text: str, sections: Optional[dict] = None) -> dict:
    """
    Extract all skills from resume text.

    Parameters
    ----------
    resume_text : str
        Full resume text from pdf_extractor.py
    sections : dict, optional
        Sections dict from section_detector.py. If provided, we weight skills
        found in the "Skills" section higher than elsewhere.

    Returns
    -------
    dict with keys:
        "skills"           : list[dict] — ranked skills with metadata
        "total_unique"     : int — unique skill count
        "total_mentions"   : int — sum of all skill mention counts
        "skill_categories" : dict — skills grouped by category

    Each skill dict has:
        "name"       : str   — the skill
        "category"   : str   — from taxonomy (e.g., "programming_languages")
        "confidence" : float — 0-1 confidence score
        "count"      : int   — how many times mentioned
        "method"     : str   — "exact", "semantic", or "fuzzy"
        "source"     : str   — which resume section (e.g., "experience")

    Example
    -------
        result = extract_skills_from_resume(resume_text, sections)
        for skill in result["skills"]:
            print(f"{skill['name']} ({skill['confidence']:.2f})")
    """
    taxonomy = _load_skill_taxonomy()

    # Build a flat list of all known skills for quick lookup
    all_skills = {}
    for category, skills_list in taxonomy.items():
        for skill in skills_list:
            all_skills[skill.lower()] = category

    # Extract candidate skills from resume
    detected = _detect_skill_candidates(resume_text, all_skills)

    # Rank and filter
    ranked_skills = _rank_skills(detected, resume_text, sections)

    # Group by category
    by_category = {}
    for skill in ranked_skills:
        cat = skill["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(skill)

    return {
        "skills": ranked_skills,
        "total_unique": len(ranked_skills),
        "total_mentions": sum(s["count"] for s in ranked_skills),
        "skill_categories": by_category,
    }


def match_skills_to_jd(resume_skills: list[dict], jd_text: str) -> dict:
    """
    Compare resume skills against Job Description text.

    Useful for "missing skills" detection and resume vs JD matching score.

    Parameters
    ----------
    resume_skills : list[dict]
        From extract_skills_from_resume()["skills"]
    jd_text : str
        Job description text

    Returns
    -------
    dict with keys:
        "matched_skills"    : list[dict] — skills from resume that appear in JD
        "missing_skills"    : list[dict] — skills from JD not in resume
        "match_score"       : float — 0-1, % of JD skills the resume has
        "matched_count"     : int
        "jd_total"          : int
    """
    jd_skills = extract_skills_from_jd(jd_text)
    jd_skill_names = {s["name"].lower() for s in jd_skills}
    resume_skill_names = {s["name"].lower() for s in resume_skills}

    matched = [s for s in resume_skills if s["name"].lower() in jd_skill_names]
    missing = [s for s in jd_skills if s["name"].lower() not in resume_skill_names]

    match_score = len(matched) / len(jd_skill_names) if jd_skill_names else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_score": match_score,
        "matched_count": len(matched),
        "jd_total": len(jd_skill_names),
    }


def extract_skills_from_jd(jd_text: str) -> list[dict]:
    """
    Extract skills mentioned in a Job Description.

    Uses the same taxonomy and detection methods as resume skill extraction.
    Used by match_skills_to_jd() and the "resume vs JD" comparison feature.

    Parameters
    ----------
    jd_text : str
        Job description text

    Returns
    -------
    list[dict] — skills with same structure as resume skill dicts
    """
    taxonomy = _load_skill_taxonomy()

    all_skills = {}
    for category, skills_list in taxonomy.items():
        for skill in skills_list:
            all_skills[skill.lower()] = category

    detected = _detect_skill_candidates(jd_text, all_skills)
    ranked = _rank_skills(detected, jd_text)

    return ranked


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_skill_candidates(text: str, all_skills: dict) -> dict:
    """
    Find potential skill matches in text using multiple methods.

    Returns dict: {skill_name → {count, mentions_positions, methods_used}}
    """
    text_clean = clean_for_matching(text)
    tokens = tokenize(text, clean=False)  # tokens already lowercased by clean_for_matching

    detected = {}

    # -----------------------------------------------------------------------
    # Method 1: Exact keyword match
    # -----------------------------------------------------------------------
    for skill_name in all_skills.keys():
        # Match whole words only (surrounded by word boundaries)
        pattern = r"\b" + re.escape(skill_name) + r"\b"
        matches = re.finditer(pattern, text_clean)
        count = len(list(matches))

        if count > 0:
            detected.setdefault(skill_name, {
                "count": 0,
                "methods": set(),
                "positions": [],
            })
            detected[skill_name]["count"] += count
            detected[skill_name]["methods"].add("exact")

    # -----------------------------------------------------------------------
    # Method 2: Semantic similarity (if model available)
    # -----------------------------------------------------------------------
    model = _load_embedding_model()
    if model:
        try:
            _detect_via_semantic_matching(text, all_skills, detected, model)
        except Exception:
            pass  # fall through to fuzzy matching

    # -----------------------------------------------------------------------
    # Method 3: Fuzzy token overlap (fallback)
    # -----------------------------------------------------------------------
    _detect_via_fuzzy_overlap(tokens, all_skills, detected)

    return detected


def _detect_via_semantic_matching(text: str, all_skills: dict, detected: dict, model) -> None:
    """
    Use sentence-transformers to find semantically similar skills.

    A skill like "full-stack development" might match "full stack" or "fullstack".
    Semantic embeddings catch these without explicit aliases.
    """
    try:
        # Encode skill names and sentences from text
        skill_embeddings = model.encode(list(all_skills.keys()))
        sentences = extract_sentences(text)

        for sent in sentences:
            sent_embedding = model.encode(sent)

            # Check similarity to each skill
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([sent_embedding], skill_embeddings)[0]

            for i, sim in enumerate(similarities):
                if sim > 0.6:  # threshold for "semantically similar"
                    skill_name = list(all_skills.keys())[i]
                    # Only increment if not already detected via exact match
                    if skill_name not in detected or "semantic" not in detected[skill_name]["methods"]:
                        detected.setdefault(skill_name, {
                            "count": 0,
                            "methods": set(),
                            "positions": [],
                        })
                        detected[skill_name]["count"] += 1
                        detected[skill_name]["methods"].add("semantic")
    except Exception:
        pass  # silently fall back to fuzzy matching


def _detect_via_fuzzy_overlap(tokens: list[str], all_skills: dict, detected: dict) -> None:
    """
    Fuzzy matching: if 2+ tokens from a multi-word skill appear in resume,
    consider it a match (lower confidence).

    E.g., "machine learning" matches if tokens include "machine" and "learning".
    """
    for skill_name in all_skills.keys():
        skill_tokens = skill_name.split()
        if len(skill_tokens) < 2:
            continue  # single-word skills handled by exact match

        # Count how many tokens of this skill appear in resume
        overlap = sum(1 for t in skill_tokens if t in tokens)
        if overlap >= len(skill_tokens) - 1:  # at least n-1 tokens match
            detected.setdefault(skill_name, {
                "count": 0,
                "methods": set(),
                "positions": [],
            })
            detected[skill_name]["count"] += 1
            detected[skill_name]["methods"].add("fuzzy")


def _rank_skills(detected: dict, resume_text: str, sections: Optional[dict] = None) -> list[dict]:
    """
    Convert raw detections into ranked, scored skill dicts.

    Scoring factors:
    - Detection method (exact > semantic > fuzzy)
    - Frequency (more mentions = higher score)
    - Section (Skills section > Experience > other)
    """
    taxonomy = _load_skill_taxonomy()

    # Invert taxonomy for fast lookup: skill_name → category
    skill_to_category = {}
    for category, skills_list in taxonomy.items():
        for skill in skills_list:
            skill_to_category[skill.lower()] = category

    ranked = []

    for skill_name, info in detected.items():
        # Base confidence by method
        methods = info["methods"]
        if "exact" in methods:
            confidence = 0.95
        elif "semantic" in methods:
            confidence = 0.75
        else:
            confidence = 0.5

        # Boost by frequency
        frequency_boost = min(info["count"] * 0.05, 0.2)
        confidence = min(confidence + frequency_boost, 1.0)

        # Determine source section
        source_section = _determine_skill_source_section(skill_name, resume_text, sections)

        ranked.append({
            "name": skill_name,
            "category": skill_to_category.get(skill_name, "other"),
            "confidence": confidence,
            "count": info["count"],
            "method": list(methods)[0],  # primary detection method
            "source": source_section,
        })

    # Sort by confidence descending, then by count
    ranked.sort(key=lambda s: (-s["confidence"], -s["count"]))
    return ranked


def _determine_skill_source_section(skill_name: str, resume_text: str, sections: Optional[dict] = None) -> str:
    """
    Determine which resume section this skill appears in.

    If sections dict is provided, search in order: Skills, Experience, Education, other.
    If not provided, return "unknown".
    """
    if not sections:
        return "unknown"

    skill_pattern = r"\b" + re.escape(skill_name) + r"\b"

    # Priority order: Skills section is most relevant
    for section_name in ["skills", "experience", "education", "projects"]:
        section_text = sections.get(section_name, "")
        if section_text and re.search(skill_pattern, section_text, re.IGNORECASE):
            return section_name

    return "other"
