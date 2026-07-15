"""
utils/jd_matcher.py
===================
Match resume against job description (JD) to measure compatibility.

WHY THIS MODULE EXISTS:
    Job descriptions vary wildly — same role at different companies looks
    different on paper. This module answers:
    - "How well does my resume match this specific job?"
    - "What skills does the job want that I'm missing?"
    - "Which parts of my resume are most relevant to this job?"

MATCHING STRATEGY:
    We use 3 complementary methods, all 100% offline:
    1. TF-IDF vectorization (keyword overlap) — fast, transparent
    2. Semantic similarity via sentence-transformers — catches paraphrases
    3. Skill-level matching — job wants Python + ML, resume has both?

RESULT:
    - Overall "JD match score" (0-100)
    - Per-skill matching (matched, missing, extra)
    - Per-section relevance (which sections of resume are most relevant)
    - Tailoring recommendations
"""

import re
from typing import Optional

from config import USE_SEMANTIC_MATCHING, EMBEDDING_MODEL_CACHE_DIR, EMBEDDING_MODEL_NAME
from utils.text_cleaner import (
    clean_for_matching,
    extract_sentences,
    get_meaningful_tokens,
)
from utils.skill_extractor import (
    extract_skills_from_resume,
    extract_skills_from_jd,
    match_skills_to_jd,
)


# ---------------------------------------------------------------------------
# Global state: cached embedding model
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = None


def _load_embedding_model():
    """Load sentence-transformers model if available (cached locally)."""
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    if not USE_SEMANTIC_MATCHING:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=EMBEDDING_MODEL_CACHE_DIR,
        )
        return _EMBEDDING_MODEL
    except Exception:
        # No internet, model not cached, or import failed
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_resume_to_jd(resume_text: str, jd_text: str, sections: Optional[dict] = None) -> dict:
    """
    Compare resume against a job description.

    Parameters
    ----------
    resume_text : str
        Full resume text from pdf_extractor.py
    jd_text : str
        Job description text
    sections : dict, optional
        Resume sections from section_detector.py

    Returns
    -------
    dict with keys:
        "overall_match_score"  : float (0-100) — how well resume matches JD
        "keyword_match_score"  : float (0-1)   — TF-IDF keyword overlap
        "semantic_match_score" : float (0-1)   — semantic similarity score
        "skill_match_result"   : dict          — from skill_extractor.match_skills_to_jd()
        "section_relevance"    : dict          — which resume sections match JD best
        "jd_keywords"          : list[str]     — top keywords extracted from JD
        "resume_keywords"      : list[str]     — top keywords from resume
        "recommendations"      : list[str]     — tailoring suggestions
        "match_grade"          : str           — letter grade: A, B, C, D, F

    Example
    -------
        result = match_resume_to_jd(resume_text, jd_text, sections)
        print(f"Match Score: {result['overall_match_score']}/100 ({result['match_grade']})")
        for rec in result["recommendations"]:
            print(f"  → {rec}")
    """
    # 1. Extract skills from both
    resume_skills_result = extract_skills_from_resume(resume_text, sections)
    resume_skills = resume_skills_result["skills"]

    jd_skills = extract_skills_from_jd(jd_text)

    # 2. Skill-level matching
    skill_match = match_skills_to_jd(resume_skills, jd_text)

    # 3. Keyword-level matching
    keyword_score = _compute_keyword_match_score(resume_text, jd_text)

    # 4. Semantic-level matching (if model available)
    semantic_score = _compute_semantic_match_score(resume_text, jd_text)

    # 5. Section-level relevance (which resume sections best match JD)
    section_relevance = _compute_section_relevance(resume_text, jd_text, sections)

    # 6. Extract top keywords
    jd_keywords = _extract_top_keywords(jd_text, n=10)
    resume_keywords = _extract_top_keywords(resume_text, n=10)

    # 7. Compute overall match score
    # Weight: skills (40%), keywords (30%), semantic (20%), sections (10%)
    skill_weight = min(skill_match["match_score"] * 1.0, 1.0)  # 0-1 scale
    overall_match = (
        skill_weight * 0.40 +
        keyword_score * 0.30 +
        semantic_score * 0.20 +
        (sum(section_relevance.values()) / max(len(section_relevance), 1)) * 0.10
    )
    overall_match_score = round(overall_match * 100, 1)

    # 8. Generate recommendations
    recommendations = _generate_jd_recommendations(
        skill_match, overall_match_score, resume_text, jd_text
    )

    # 9. Assign letter grade
    match_grade = _score_to_grade(overall_match_score)

    return {
        "overall_match_score": overall_match_score,
        "keyword_match_score": keyword_score,
        "semantic_match_score": semantic_score,
        "skill_match_result": skill_match,
        "section_relevance": section_relevance,
        "jd_keywords": jd_keywords,
        "resume_keywords": resume_keywords,
        "recommendations": recommendations,
        "match_grade": match_grade,
    }


def extract_jd_requirements(jd_text: str) -> dict:
    """
    Parse a job description to extract structured requirements.

    Useful for displaying "What this job is looking for" on the UI.

    Returns
    -------
    dict with keys:
        "skills"        : list[str]  — extracted skills from JD
        "experience"    : str        — years of experience or level
        "keywords"      : list[str]  — top keywords
        "education"     : str        — education requirements (if mentioned)
        "title"         : str        — job title (if in JD)
    """
    from utils.skill_extractor import extract_skills_from_jd

    jd_skills = extract_skills_from_jd(jd_text)
    jd_skill_names = [s["name"] for s in jd_skills]

    keywords = _extract_top_keywords(jd_text, n=15)

    # Try to extract years of experience
    experience_match = re.search(
        r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)",
        jd_text,
        re.IGNORECASE
    )
    experience = experience_match.group(0) if experience_match else "Not specified"

    # Try to extract education requirement
    education_keywords = ["bachelor", "master", "phd", "degree", "diploma", "certification"]
    education_found = [kw for kw in education_keywords if kw in jd_text.lower()]
    education = education_found[0].title() if education_found else "Not specified"

    # Extract job title (usually in first line)
    lines = jd_text.split("\n")
    title = lines[0].strip() if lines else "Unknown"

    return {
        "skills": jd_skill_names,
        "experience": experience,
        "keywords": keywords,
        "education": education,
        "title": title,
    }


# ---------------------------------------------------------------------------
# Internal scoring functions
# ---------------------------------------------------------------------------

def _compute_keyword_match_score(resume_text: str, jd_text: str) -> float:
    """
    TF-IDF based keyword matching between resume and JD.

    Returns 0-1 score indicating keyword overlap.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=100,
            ngram_range=(1, 2),
        )

        # Fit on both documents
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Similarity typically ranges 0.2-0.8 for related docs
        # Scale to 0-1: if sim is 0.4, return 0.4; if 0.8, return closer to 1.0
        score = min(similarity / 0.8, 1.0)  # normalize to 0-1
        return float(score)
    except Exception:
        return 0.0


def _compute_semantic_match_score(resume_text: str, jd_text: str) -> float:
    """
    Semantic similarity via sentence-transformers embeddings.

    More nuanced than TF-IDF: catches paraphrases and synonyms.
    """
    model = _load_embedding_model()
    if not model:
        return 0.0

    try:
        # Encode full documents as single embeddings
        resume_embedding = model.encode(resume_text)
        jd_embedding = model.encode(jd_text)

        from sklearn.metrics.pairwise import cosine_similarity

        similarity = cosine_similarity(
            [resume_embedding],
            [jd_embedding]
        )[0][0]

        return float(similarity)
    except Exception:
        return 0.0


def _compute_section_relevance(
    resume_text: str,
    jd_text: str,
    sections: Optional[dict] = None,
) -> dict:
    """
    Score relevance of each resume section to the JD.

    Returns dict: {section_name → 0-1 relevance score}
    """
    if not sections:
        return {}

    model = _load_embedding_model()
    jd_sentences = extract_sentences(jd_text)

    relevance = {}

    for section_name, section_text in sections.items():
        if not section_text.strip():
            relevance[section_name] = 0.0
            continue

        if model:
            # Semantic matching: encode section and check similarity to JD sentences
            try:
                section_embedding = model.encode(section_text)
                jd_embeddings = model.encode(jd_sentences)

                from sklearn.metrics.pairwise import cosine_similarity

                similarities = cosine_similarity([section_embedding], jd_embeddings)[0]
                relevance[section_name] = float(max(similarities) if similarities.size > 0 else 0.0)
            except Exception:
                relevance[section_name] = 0.0
        else:
            # Fallback: keyword overlap
            section_tokens = set(get_meaningful_tokens(section_text))
            jd_tokens = set(get_meaningful_tokens(jd_text))
            overlap = len(section_tokens & jd_tokens)
            total = len(jd_tokens) if jd_tokens else 1
            relevance[section_name] = min(overlap / total, 1.0)

    return relevance


def _extract_top_keywords(text: str, n: int = 10) -> list[str]:
    """
    Extract top N meaningful keywords from text using TF-IDF.

    Returns list of keywords, sorted by importance.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=n,
            ngram_range=(1, 2),
        )
        vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return sorted(keywords.tolist())
    except Exception:
        # Fallback: return most common meaningful tokens
        tokens = get_meaningful_tokens(text)
        from collections import Counter

        token_counts = Counter(tokens)
        top_tokens = [t for t, _ in token_counts.most_common(n)]
        return top_tokens


# ---------------------------------------------------------------------------
# Recommendations and grading
# ---------------------------------------------------------------------------

def _generate_jd_recommendations(
    skill_match: dict,
    overall_score: float,
    resume_text: str,
    jd_text: str,
) -> list[str]:
    """
    Generate actionable recommendations for tailoring resume to JD.
    """
    recommendations = []

    matched_count = skill_match["matched_count"]
    jd_total = skill_match["jd_total"]
    missing = skill_match["missing_skills"]

    # Skill recommendations
    if matched_count == 0:
        recommendations.append("❌ No matching skills found. Consider rewriting to emphasize relevant skills.")
    elif matched_count < jd_total * 0.5:
        missing_skills = [s["name"] for s in missing[:3]]
        recommendations.append(
            f"⚠ You're missing key skills: {', '.join(missing_skills)}. "
            "Consider adding these to your Skills section."
        )
    else:
        recommendations.append(f"✓ Good skill match ({matched_count}/{jd_total} skills present)")

    # Overall match recommendations
    if overall_score >= 80:
        recommendations.append("✓ Strong match! This resume is well-tailored to the job description.")
    elif overall_score >= 60:
        recommendations.append(
            "➜ Decent match. Highlight more relevant experience and add missing keywords."
        )
    else:
        recommendations.append(
            "⚠ Weak match. Consider significantly rewriting to better address job requirements."
        )

    # Keyword recommendations
    jd_keywords = _extract_top_keywords(jd_text, n=5)
    resume_keywords = _extract_top_keywords(resume_text, n=5)
    missing_keywords = [kw for kw in jd_keywords if kw not in resume_keywords]
    if missing_keywords:
        recommendations.append(
            f"💡 Add keywords from the job description: {', '.join(missing_keywords)}"
        )

    return recommendations


def _score_to_grade(score: float) -> str:
    """Convert numeric score (0-100) to letter grade (A-F)."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


# ---------------------------------------------------------------------------
# Utility: format match result for display
# ---------------------------------------------------------------------------

def format_jd_match_report(match_result: dict) -> str:
    """
    Format the match result as readable text for Streamlit display.

    Parameters
    ----------
    match_result : dict
        From match_resume_to_jd()

    Returns
    -------
    str — formatted report
    """
    report = []

    # Header
    score = match_result["overall_match_score"]
    grade = match_result["match_grade"]
    report.append(f"{'='*60}")
    report.append(f"JD MATCH REPORT")
    report.append(f"{'='*60}")
    report.append(f"\n📊 Overall Match Score: {score}/100 (Grade: {grade})")

    # Breakdown
    report.append(f"\n{'─'*60}")
    report.append("BREAKDOWN:")
    report.append(f"  • Keyword Match:  {match_result['keyword_match_score']:.1%}")
    report.append(f"  • Semantic Match: {match_result['semantic_match_score']:.1%}")

    skill_match = match_result["skill_match_result"]
    report.append(f"  • Skill Match:    {skill_match['matched_count']}/{skill_match['jd_total']} ({skill_match['match_score']:.1%})")

    # Missing skills
    missing = skill_match["missing_skills"]
    if missing:
        report.append(f"\n{'─'*60}")
        report.append(f"MISSING SKILLS ({len(missing)}):")
        for skill in missing[:5]:  # top 5
            report.append(f"  • {skill['name']}")
        if len(missing) > 5:
            report.append(f"  ... and {len(missing) - 5} more")

    # Recommendations
    if match_result["recommendations"]:
        report.append(f"\n{'─'*60}")
        report.append("RECOMMENDATIONS:")
        for rec in match_result["recommendations"]:
            report.append(f"  {rec}")

    return "\n".join(report)
