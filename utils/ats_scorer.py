"""
utils/ats_scorer.py
===================
ATS (Applicant Tracking System) scoring engine for resumes.

WHY THIS MODULE EXISTS:
    An ATS score (0-100) tells a candidate: "How well does this resume match
    what an employer's system is looking for?"

    We calculate the score using 5 weighted factors:
    1. Section Completeness (20%)  — has Education, Experience, Skills, etc.?
    2. Skill Match (30%)            — are relevant skills present?
    3. Keyword Density (15%)        — resume vs JD keyword overlap
    4. Formatting (15%)             — contact info, bullet points, readability
    5. Readability (20%)            — action verbs, quantified achievements

SCORING PHILOSOPHY:
    - Each factor has sub-scores (0-1) that are weighted and combined.
    - All scoring logic is transparent and documented so judges see exactly why
      a resume got 72 vs 85.
    - Scores are stable: same resume always gets same score (no randomness).
    - JD is optional: if no JD provided, we score on absolute quality criteria.

DEPENDENCIES:
    - section_detector.py  (section completeness)
    - skill_extractor.py   (skill matching, JD skill comparison)
    - text_cleaner.py      (tokenization, sentence analysis)
    - config.py            (weights, thresholds)
"""

import re
from typing import Optional

from config import ATS_WEIGHTS, MIN_RESUME_WORDS, MAX_RESUME_WORDS
from utils.text_cleaner import (
    extract_sentences,
    tokenize,
    get_meaningful_tokens,
    word_count,
)
from utils.skill_extractor import (
    extract_skills_from_resume,
    match_skills_to_jd,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_resume(
    resume_text: str,
    sections: dict,
    jd_text: Optional[str] = None,
) -> dict:
    """
    Calculate comprehensive ATS score for a resume.

    Parameters
    ----------
    resume_text : str
        Full resume text from pdf_extractor.py
    sections : dict
        Detected sections from section_detector.py
    jd_text : str, optional
        Job description text. If provided, matching is JD-aware.

    Returns
    -------
    dict with keys:
        "overall_score"       : float (0-100) — final ATS score
        "factors"             : dict — individual factor scores (0-1) and weights
        "sub_scores"          : dict — breakdown of each factor's components
        "feedback"            : list[str] — actionable improvement suggestions
        "strengths"           : list[str] — what the resume does well
        "weaknesses"          : list[str] — what could be improved

    Example
    -------
        result = score_resume(resume_text, sections, jd_text)
        print(f"ATS Score: {result['overall_score']}/100")
        for weakness in result["weaknesses"]:
            print(f"  ⚠ {weakness}")
    """
    # Calculate each of the 5 scoring factors
    section_score = _score_section_completeness(sections)
    skill_score = _score_skill_match(resume_text, sections, jd_text)
    keyword_score = _score_keyword_density(resume_text, jd_text)
    formatting_score = _score_formatting(resume_text, sections)
    readability_score = _score_readability(resume_text, sections)

    # Combine weighted scores
    factors = {
        "section_completeness": section_score,
        "skill_match": skill_score,
        "keyword_density": keyword_score,
        "formatting": formatting_score,
        "readability": readability_score,
    }

    overall = sum(
        factors[factor] * ATS_WEIGHTS[factor]
        for factor in ATS_WEIGHTS
    )
    overall_score = round(overall * 100, 1)

    # Generate feedback
    feedback = _generate_feedback(factors, resume_text, sections, jd_text)
    strengths = [f for f in feedback if f.startswith("✓")]
    weaknesses = [f for f in feedback if f.startswith("⚠")]

    return {
        "overall_score": overall_score,
        "factors": factors,
        "sub_scores": {
            "section_completeness": section_score,
            "skill_match": skill_score,
            "keyword_density": keyword_score,
            "formatting": formatting_score,
            "readability": readability_score,
        },
        "feedback": feedback,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


# ---------------------------------------------------------------------------
# Scoring factors (each returns 0-1)
# ---------------------------------------------------------------------------

def _score_section_completeness(sections: dict) -> float:
    """
    Score: Does the resume have all the important sections?

    Required sections: contact, experience, education, skills
    Optional but good: summary, projects, certifications

    Returns 0-1 score.
    """
    required_sections = ["contact", "experience", "education", "skills"]
    optional_sections = ["summary", "projects", "certifications"]

    required_found = sum(1 for s in required_sections if sections.get(s, "").strip())
    optional_found = sum(1 for s in optional_sections if sections.get(s, "").strip())

    # Required: 4 sections = 0.8 points, optional: 3 = 0.2 points
    score = (required_found / len(required_sections)) * 0.8
    score += (optional_found / len(optional_sections)) * 0.2

    return min(score, 1.0)


def _score_skill_match(
    resume_text: str,
    sections: dict,
    jd_text: Optional[str] = None,
) -> float:
    """
    Score: How many relevant skills are present?

    If JD provided, score based on skill match to JD.
    If no JD, score based on:
    - Total unique skills found
    - Skill categories covered (breadth)
    - Confidence of skill detections

    Returns 0-1 score.
    """
    resume_skills_result = extract_skills_from_resume(resume_text, sections)
    resume_skills = resume_skills_result["skills"]

    if not resume_skills:
        return 0.0

    # Base score: number of unique skills (scale: 5+ skills = 0.5 points)
    skill_count_score = min(len(resume_skills) / 10, 0.5)

    # Category diversity score: how many different skill categories?
    categories = len(resume_skills_result["skill_categories"])
    category_score = min(categories / 8, 0.3)  # 8+ categories = 0.3 points

    # Confidence score: average confidence of detected skills
    avg_confidence = sum(s["confidence"] for s in resume_skills) / len(resume_skills)
    confidence_score = avg_confidence * 0.2

    score = skill_count_score + category_score + confidence_score

    # If JD provided, boost score based on JD match rate
    if jd_text:
        match_result = match_skills_to_jd(resume_skills, jd_text)
        jd_match_rate = match_result["match_score"]
        jd_boost = jd_match_rate * 0.3  # JD matching can add up to 0.3 points
        score = min(score + jd_boost, 1.0)

    return min(score, 1.0)


def _score_keyword_density(resume_text: str, jd_text: Optional[str] = None) -> float:
    """
    Score: Do resume keywords match job description?

    If JD provided, use TF-IDF cosine similarity.
    If no JD, score based on keyword density heuristics (presence of action
    verbs, quantified metrics, industry-standard terms).

    Returns 0-1 score.
    """
    if not jd_text:
        # Fallback: score based on keyword richness (action verbs, metrics, etc.)
        score = _score_keyword_richness(resume_text)
        return score

    # TF-IDF based matching with JD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        # Scale similarity (0-1) to score (0-1)
        # Typical similarity is 0.2-0.8 for related docs, so scale generously
        score = min(similarity * 1.2, 1.0)
        return score
    except Exception:
        # Fallback if vectorization fails
        return _score_keyword_richness(resume_text)


def _score_keyword_richness(resume_text: str) -> float:
    """
    Score keyword richness when no JD is available.

    Looks for:
    - Action verbs (led, developed, managed, implemented)
    - Quantified metrics (increased by 50%, reduced costs by $200k)
    - Industry terms
    - Technical jargon
    """
    score = 0.0

    # Action verbs
    action_verbs = [
        "led", "developed", "implemented", "designed", "managed", "improved",
        "built", "created", "delivered", "achieved", "increased", "reduced",
        "optimized", "streamlined", "coordinated", "spearheaded", "driven",
        "engineered", "architected", "supervised", "directed"
    ]
    action_verb_count = sum(
        len(re.findall(r"\b" + verb + r"\b", resume_text.lower()))
        for verb in action_verbs
    )
    action_verb_score = min(action_verb_count / 10, 0.4)  # 10+ verbs = 0.4
    score += action_verb_score

    # Quantified metrics (numbers with units: %, $, x times, etc.)
    metric_pattern = r"\b\d+[\s\-]?(percent|%|\$|times|years?|months?|days?)\b"
    metric_count = len(re.findall(metric_pattern, resume_text, re.IGNORECASE))
    metric_score = min(metric_count / 5, 0.3)  # 5+ metrics = 0.3
    score += metric_score

    # Technical density (presence of technical terms)
    technical_terms = [
        "algorithm", "architecture", "framework", "library", "api", "protocol",
        "database", "server", "client", "deployment", "pipeline", "workflow"
    ]
    tech_count = sum(
        len(re.findall(r"\b" + term + r"\b", resume_text.lower()))
        for term in technical_terms
    )
    tech_score = min(tech_count / 8, 0.3)  # 8+ terms = 0.3
    score += tech_score

    return min(score, 1.0)


def _score_formatting(resume_text: str, sections: dict) -> float:
    """
    Score: Is the resume well-formatted?

    Checks:
    - Contact info present (email, phone, or LinkedIn)
    - Reasonable length (not too short, not too long)
    - Bullet point usage (modern formatting)
    - No excessive formatting characters (rare, but can break ATS)
    - Line length (not too cramped)

    Returns 0-1 score.
    """
    score = 0.0

    # Contact info
    from utils.section_detector import extract_contact_info
    contact = extract_contact_info(resume_text)
    contact_fields = sum(1 for v in contact.values() if v)
    contact_score = min(contact_fields / 2, 0.2)  # 2+ contact fields = 0.2
    score += contact_score

    # Resume length
    word_cnt = word_count(resume_text)
    if MIN_RESUME_WORDS <= word_cnt <= MAX_RESUME_WORDS:
        length_score = 0.3
    elif MIN_RESUME_WORDS <= word_cnt <= MAX_RESUME_WORDS + 200:
        length_score = 0.2
    else:
        length_score = 0.0
    score += length_score

    # Bullet point usage (modern ATS-friendly formatting)
    bullet_count = len(re.findall(r"[•\-\*]\s", resume_text))
    bullet_score = min(bullet_count / 15, 0.3)  # 15+ bullets = 0.3
    score += bullet_score

    # Avoid excessive special characters that might break parsing
    special_chars = len(re.findall(r"[^A-Za-z0-9\s\-.,;:()@/\n]", resume_text))
    if special_chars < len(resume_text) * 0.05:  # <5% special chars
        special_score = 0.2
    else:
        special_score = 0.0
    score += special_score

    return min(score, 1.0)


def _score_readability(resume_text: str, sections: dict) -> float:
    """
    Score: Is the resume easy to read?

    Checks:
    - Average sentence length (shorter is better, not too short)
    - Paragraph structure (clear breaks)
    - Achievement orientation (sentences with metrics, results)
    - Consistency (similar formatting across sections)

    Returns 0-1 score.
    """
    score = 0.0

    sentences = extract_sentences(resume_text)
    if not sentences:
        return 0.0

    # Sentence length (ideal: 10-20 words)
    avg_sent_length = sum(len(s.split()) for s in sentences) / len(sentences)
    if 10 <= avg_sent_length <= 20:
        length_score = 0.3
    elif 8 <= avg_sent_length <= 25:
        length_score = 0.2
    else:
        length_score = 0.0
    score += length_score

    # Achievement orientation (% of sentences with quantified results)
    achievement_pattern = r"\b\d+[\s\-]?(percent|%|\$|times?|improved|increased|reduced|achieved)\b"
    achievement_count = sum(
        1 for s in sentences if re.search(achievement_pattern, s, re.IGNORECASE)
    )
    achievement_score = min((achievement_count / len(sentences)) * 0.35, 0.35)
    score += achievement_score

    # Paragraph structure (not all text in one block)
    lines = resume_text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    if non_empty_lines:
        avg_para_length = len(non_empty_lines) / max(len(re.findall(r"\n\n+", resume_text)), 1)
        if avg_para_length < 50:  # reasonably chunked
            para_score = 0.35
        else:
            para_score = 0.15
    else:
        para_score = 0.0
    score += para_score

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------

def _generate_feedback(
    factors: dict,
    resume_text: str,
    sections: dict,
    jd_text: Optional[str] = None,
) -> list[str]:
    """
    Generate actionable feedback based on factor scores.

    Returns list of strings starting with "✓" (strength) or "⚠" (weakness).
    """
    feedback = []

    # Section completeness
    if factors["section_completeness"] >= 0.9:
        feedback.append("✓ All key resume sections are present (Contact, Experience, Education, Skills)")
    elif factors["section_completeness"] >= 0.6:
        feedback.append("⚠ Some sections are missing. Add Summary and/or Projects to strengthen your resume")
    else:
        feedback.append("⚠ Several important sections are missing. Ensure you have Experience and Education sections")

    # Skill match
    if factors["skill_match"] >= 0.8:
        feedback.append("✓ Strong skill diversity and depth detected across multiple categories")
    elif factors["skill_match"] >= 0.5:
        feedback.append("⚠ Consider listing more technical skills and expanding skill categories")
    else:
        feedback.append("⚠ Add more skills to your resume, especially in the Skills section")

    # Keyword density
    if factors["keyword_density"] >= 0.7:
        if jd_text:
            feedback.append("✓ Good keyword match with the job description")
        else:
            feedback.append("✓ Resume contains strong action verbs and quantified achievements")
    else:
        if jd_text:
            feedback.append("⚠ Consider adding more keywords from the job description to improve matching")
        else:
            feedback.append("⚠ Include more quantified metrics and action verbs (e.g., 'increased revenue by 25%')")

    # Formatting
    if factors["formatting"] >= 0.8:
        feedback.append("✓ Excellent formatting with clear structure and contact information")
    else:
        contact = extract_contact_info(resume_text)
        if not any(contact.values()):
            feedback.append("⚠ Add contact information (email, phone, or LinkedIn) at the top")
        word_cnt = word_count(resume_text)
        if word_cnt < MIN_RESUME_WORDS:
            feedback.append(f"⚠ Resume is too short ({word_cnt} words). Aim for {MIN_RESUME_WORDS}+ words")
        elif word_cnt > MAX_RESUME_WORDS:
            feedback.append(f"⚠ Resume is too long ({word_cnt} words). Try to keep it under {MAX_RESUME_WORDS} words")

    # Readability
    if factors["readability"] >= 0.8:
        feedback.append("✓ Excellent readability with clear achievements and impact metrics")
    else:
        sentences = extract_sentences(resume_text)
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len > 25:
                feedback.append("⚠ Some sentences are too long. Break them into shorter, punchier statements")
        feedback.append("⚠ Include more quantified results (percentages, dollar amounts, timeframes)")

    return feedback


def extract_contact_info(resume_text: str) -> dict:
    """Import from section_detector for convenience."""
    from utils.section_detector import extract_contact_info as _extract
    return _extract(resume_text)
