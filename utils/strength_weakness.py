"""
utils/strength_weakness.py
==========================
Analyze resume strengths and weaknesses, generate improvement suggestions.

WHY THIS MODULE EXISTS:
    Beyond scoring, candidates need *actionable feedback*.
    This module answers:
    - "What am I doing well?"
    - "What should I fix?"
    - "How do I rewrite this bullet?"

ANALYSIS CATEGORIES:
    Strengths:
        - Achievement orientation (uses quantified metrics)
        - Action verb usage (led, developed, improved, etc.)
        - Technical depth (diverse skills, depth per skill)
        - Career progression (clear growth trajectory)
        - Formatting quality (clear structure, no clutter)

    Weaknesses:
        - Vague descriptions ("responsible for stuff")
        - Lack of metrics ("built a system" vs "built a system handling 1M+ users/day")
        - Employment gaps (6+ month unexplained gaps)
        - Weak verbs ("worked on", "assisted with")
        - Inconsistent formatting
        - Missing sections

SUGGESTIONS:
    For each weakness, we offer specific rewrites and examples.
    All logic is offline and rule-based — no external APIs.
"""

import re
from typing import Optional
from datetime import datetime

from utils.text_cleaner import extract_sentences, get_meaningful_tokens
from utils.skill_extractor import extract_skills_from_resume
from utils.section_detector import get_detected_sections_summary


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRONG_ACTION_VERBS = {
    "led", "managed", "developed", "designed", "implemented", "built",
    "created", "achieved", "improved", "increased", "delivered", "reduced",
    "optimized", "engineered", "architected", "spearheaded", "directed",
    "established", "launched", "transformed", "revolutionized", "pioneered",
    "streamlined", "accelerated", "automated", "coordinated", "coordinated",
}

WEAK_ACTION_VERBS = {
    "worked", "helped", "assisted", "involved", "participated", "responsible",
    "was", "had", "did", "made", "got", "used", "tried", "attempted",
}

METRIC_PATTERNS = [
    r"\b\d+\s*(?:percent|%)\b",           # 50%, 25 percent
    r"\$\s*\d+(?:k|m|b)?(?:\+)?\b",       # $50k, $2M
    r"\b\d+x\s+(?:faster|better|more)\b", # 3x faster
    r"\b\d+\s*(?:years?|months?)\b",      # 5 years, 12 months
    r"\b(?:increased|decreased|reduced|grew|cut)\s+(?:by\s+)?\d+",  # increased by 40%
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_resume_strengths_weaknesses(
    resume_text: str,
    sections: Optional[dict] = None,
) -> dict:
    """
    Comprehensive analysis of resume strengths and weaknesses.

    Parameters
    ----------
    resume_text : str
        Full resume text from pdf_extractor.py
    sections : dict, optional
        Sections from section_detector.py

    Returns
    -------
    dict with keys:
        "strengths"         : list[str]  — what the resume does well
        "weaknesses"        : list[str]  — areas for improvement
        "suggestions"       : list[dict] — detailed improvement suggestions
        "opportunity_score" : float      — 0-1 score for improvement potential
        "rewrite_examples"  : list[dict] — before/after examples

    Example
    -------
        analysis = analyze_resume_strengths_weaknesses(resume_text, sections)
        for strength in analysis["strengths"]:
            print(f"✓ {strength}")
        for suggestion in analysis["suggestions"]:
            print(f"⚠ {suggestion['issue']}")
            print(f"  💡 {suggestion['suggestion']}")
    """
    strengths = []
    weaknesses = []
    suggestions = []
    rewrite_examples = []

    # -----------------------------------------------------------------------
    # 1. Analyze action verb usage
    # -----------------------------------------------------------------------
    verb_analysis = _analyze_action_verbs(resume_text)
    if verb_analysis["strong_verb_percentage"] >= 0.6:
        strengths.append("✓ Excellent use of strong action verbs throughout")
    elif verb_analysis["strong_verb_percentage"] >= 0.4:
        strengths.append("✓ Good action verb usage in most bullet points")
    else:
        weaknesses.append("⚠ Many weak action verbs (worked, helped, responsible)")
        suggestions.append({
            "issue": "Weak action verbs reduce impact",
            "suggestion": "Replace weak verbs with strong ones: 'worked on' → 'led', 'helped' → 'spearheaded'",
            "severity": "medium",
        })

    # -----------------------------------------------------------------------
    # 2. Analyze achievement orientation (quantified metrics)
    # -----------------------------------------------------------------------
    metric_analysis = _analyze_metrics(resume_text)
    if metric_analysis["metric_count"] >= 8:
        strengths.append("✓ Strong use of quantified achievements and metrics")
    elif metric_analysis["metric_count"] >= 4:
        strengths.append("✓ Good use of quantified results in some sections")
    else:
        weaknesses.append("⚠ Few quantified metrics or achievements")
        suggestions.append({
            "issue": "Lack of quantified results weakens impact",
            "suggestion": "Add specific numbers: 'Improved performance' → 'Improved API response time by 40%'",
            "severity": "high",
            "examples": metric_analysis["example_bad_bullets"][:2],
        })

    # -----------------------------------------------------------------------
    # 3. Analyze technical depth
    # -----------------------------------------------------------------------
    if sections:
        skills_result = extract_skills_from_resume(resume_text, sections)
        skills_count = len(skills_result["skills"])
        categories = len(skills_result["skill_categories"])

        if skills_count >= 15 and categories >= 5:
            strengths.append("✓ Impressive range of technical skills across multiple domains")
        elif skills_count >= 8:
            strengths.append("✓ Solid technical skill set")
        else:
            weaknesses.append("⚠ Limited technical skills listed")
            suggestions.append({
                "issue": "Insufficient skill diversity or depth",
                "suggestion": "Expand your Skills section with relevant technologies and frameworks",
                "severity": "medium",
            })

    # -----------------------------------------------------------------------
    # 4. Analyze section completeness
    # -----------------------------------------------------------------------
    if sections:
        section_summary = get_detected_sections_summary(sections)
        required_sections = ["contact", "experience", "education", "skills"]
        found_required = sum(1 for s in required_sections if section_summary[s]["found"])

        if found_required == len(required_sections):
            strengths.append("✓ All essential sections present (Contact, Experience, Education, Skills)")
        else:
            missing = [s for s in required_sections if not section_summary[s]["found"]]
            weaknesses.append(f"⚠ Missing sections: {', '.join(missing)}")
            suggestions.append({
                "issue": f"Missing required sections: {', '.join(missing)}",
                "suggestion": "Add all key sections. At minimum: Contact, Experience, Education, Skills",
                "severity": "high",
            })

    # -----------------------------------------------------------------------
    # 5. Analyze for employment gaps
    # -----------------------------------------------------------------------
    gap_analysis = _analyze_employment_gaps(resume_text)
    if gap_analysis["has_large_gaps"]:
        weaknesses.append("⚠ Unexplained employment gaps (6+ months)")
        suggestions.append({
            "issue": "Large employment gaps may raise questions",
            "suggestion": "Add brief explanations for gaps: 'Freelance consulting', 'Personal development', etc.",
            "severity": "medium",
        })
    else:
        strengths.append("✓ Clear, continuous employment history")

    # -----------------------------------------------------------------------
    # 6. Analyze consistency and formatting
    # -----------------------------------------------------------------------
    consistency = _analyze_consistency(resume_text)
    if consistency["score"] >= 0.8:
        strengths.append("✓ Consistent formatting and structure throughout")
    else:
        weaknesses.append("⚠ Inconsistent formatting (bullets, spacing, date styles)")
        suggestions.append({
            "issue": "Inconsistent formatting hurts readability",
            "suggestion": "Use consistent bullet styles, date formats (MM/YYYY), and indentation",
            "severity": "low",
        })

    # -----------------------------------------------------------------------
    # 7. Generate rewrite examples
    # -----------------------------------------------------------------------
    rewrite_examples = _generate_rewrite_examples(resume_text)

    # -----------------------------------------------------------------------
    # Calculate opportunity score (how much room for improvement)
    # -----------------------------------------------------------------------
    total_issues = len(weaknesses)
    opportunity_score = min(total_issues / 5, 1.0)  # 5+ issues = 1.0 (high opportunity)

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "opportunity_score": opportunity_score,
        "rewrite_examples": rewrite_examples,
    }


def generate_improvement_plan(analysis: dict) -> list[str]:
    """
    Generate a prioritized action plan based on analysis.

    Returns list of actionable steps, highest priority first.
    """
    plan = []

    # Sort suggestions by severity
    high_severity = [s for s in analysis["suggestions"] if s.get("severity") == "high"]
    medium_severity = [s for s in analysis["suggestions"] if s.get("severity") == "medium"]
    low_severity = [s for s in analysis["suggestions"] if s.get("severity") == "low"]

    all_sorted = high_severity + medium_severity + low_severity

    for i, suggestion in enumerate(all_sorted[:5], 1):
        plan.append(f"{i}. {suggestion['issue']}")
        plan.append(f"   → {suggestion['suggestion']}")

    if len(all_sorted) > 5:
        plan.append(f"... and {len(all_sorted) - 5} more improvements")

    return plan


# ---------------------------------------------------------------------------
# Internal analysis functions
# ---------------------------------------------------------------------------

def _analyze_action_verbs(resume_text: str) -> dict:
    """
    Analyze usage of strong vs weak action verbs.

    Returns dict with:
        - strong_verb_count
        - weak_verb_count
        - strong_verb_percentage (0-1)
    """
    sentences = extract_sentences(resume_text)
    strong_count = 0
    weak_count = 0

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for verb in STRONG_ACTION_VERBS:
            if re.search(r"\b" + verb + r"\b", sentence_lower):
                strong_count += 1
                break
        for verb in WEAK_ACTION_VERBS:
            if re.search(r"\b" + verb + r"\b", sentence_lower):
                weak_count += 1
                break

    total = strong_count + weak_count if strong_count + weak_count > 0 else 1
    return {
        "strong_verb_count": strong_count,
        "weak_verb_count": weak_count,
        "strong_verb_percentage": strong_count / total,
    }


def _analyze_metrics(resume_text: str) -> dict:
    """
    Find quantified metrics and achievements in resume.

    Returns dict with:
        - metric_count: number of metrics found
        - example_bad_bullets: sentences that could use metrics
    """
    sentences = extract_sentences(resume_text)
    metric_count = 0
    bad_bullets = []

    for sentence in sentences:
        # Check if this sentence has metrics
        metric_found = False
        for pattern in METRIC_PATTERNS:
            if re.search(pattern, sentence):
                metric_count += 1
                metric_found = True
                break

        # If no metric and it's an achievement sentence, it's a "bad bullet"
        if not metric_found and any(
            verb in sentence.lower() for verb in STRONG_ACTION_VERBS
        ):
            bad_bullets.append(sentence)

    return {
        "metric_count": metric_count,
        "example_bad_bullets": bad_bullets[:3],
    }


def _analyze_employment_gaps(resume_text: str) -> dict:
    """
    Detect unexplained employment gaps (6+ months with no context).

    Looks for date patterns like "Jan 2020 - Sept 2020" and checks for gaps.
    """
    # Pattern: Month YYYY - Month YYYY
    date_pattern = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*[-–]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}"

    matches = re.finditer(date_pattern, resume_text, re.IGNORECASE)
    dates = []

    for match in matches:
        date_str = match.group(0)
        dates.append(date_str)

    # Simple heuristic: if there are date ranges but they don't cover most of
    # a typical career span, there might be gaps
    has_large_gaps = len(dates) > 0 and len(dates) < 3  # few entries = possibly gaps

    return {
        "has_large_gaps": has_large_gaps,
        "date_entries_found": len(dates),
    }


def _analyze_consistency(resume_text: str) -> dict:
    """
    Check for consistent formatting (bullets, spacing, case, etc.).

    Returns dict with:
        - score: 0-1 consistency score
    """
    lines = resume_text.split("\n")

    # Count bullet types
    bullet_types = {
        "dash": len([l for l in lines if l.strip().startswith("-")]),
        "asterisk": len([l for l in lines if l.strip().startswith("*")]),
        "dot": len([l for l in lines if l.strip().startswith("•")]),
        "number": len([l for l in lines if re.match(r"^\s*\d+\.", l)]),
    }

    # If mostly one type, that's consistent
    total_bullets = sum(bullet_types.values())
    if total_bullets == 0:
        return {"score": 1.0}

    max_type = max(bullet_types.values())
    consistency = max_type / total_bullets if total_bullets > 0 else 0

    return {"score": consistency}


def _generate_rewrite_examples(resume_text: str) -> list[dict]:
    """
    Find actual sentences from resume that could be rewritten and provide examples.

    Returns list of dicts with:
        - before: original sentence
        - after: improved version
        - reason: why the improvement helps
    """
    examples = []
    sentences = extract_sentences(resume_text)

    for sentence in sentences[:10]:  # check first 10 sentences
        # Find "bad" bullets that need metrics
        if any(verb in sentence.lower() for verb in WEAK_ACTION_VERBS):
            # Suggest strengthening with better verb
            improved = sentence
            for weak_verb in ["worked on", "helped with", "was responsible"]:
                if weak_verb in improved.lower():
                    improved = improved.replace(weak_verb, "led", 1)
                    break

            if improved != sentence:
                examples.append({
                    "before": sentence.strip(),
                    "after": improved.strip(),
                    "reason": "Use stronger action verbs for greater impact",
                })

        # Find sentences with achievements but no metrics
        if (
            any(verb in sentence.lower() for verb in ["improved", "increased", "reduced"])
            and not any(re.search(p, sentence) for p in METRIC_PATTERNS)
        ):
            examples.append({
                "before": sentence.strip(),
                "after": sentence.strip() + " (add specific metrics: %, $, or timeframe)",
                "reason": "Quantify results for greater credibility and impact",
            })

            if len(examples) >= 5:
                break

    return examples[:5]


# ---------------------------------------------------------------------------
# Utility: format analysis for display
# ---------------------------------------------------------------------------

def format_strength_weakness_report(analysis: dict) -> str:
    """
    Format analysis as readable text for Streamlit display.

    Parameters
    ----------
    analysis : dict
        From analyze_resume_strengths_weaknesses()

    Returns
    -------
    str — formatted report
    """
    report = []

    report.append("=" * 60)
    report.append("RESUME ANALYSIS")
    report.append("=" * 60)

    # Opportunity score
    score = analysis["opportunity_score"]
    if score <= 0.3:
        score_text = "✓ Limited improvement opportunities (high quality)"
    elif score <= 0.6:
        score_text = "→ Moderate improvement opportunities"
    else:
        score_text = "⚠ Significant improvement opportunities"

    report.append(f"\n{score_text}")

    # Strengths
    if analysis["strengths"]:
        report.append(f"\n{'─'*60}")
        report.append("STRENGTHS:")
        for strength in analysis["strengths"]:
            report.append(f"  {strength}")

    # Weaknesses
    if analysis["weaknesses"]:
        report.append(f"\n{'─'*60}")
        report.append("WEAKNESSES:")
        for weakness in analysis["weaknesses"]:
            report.append(f"  {weakness}")

    # Suggestions
    if analysis["suggestions"]:
        report.append(f"\n{'─'*60}")
        report.append("HOW TO IMPROVE:")
        for suggestion in analysis["suggestions"][:5]:
            report.append(f"  • {suggestion['issue']}")
            report.append(f"    → {suggestion['suggestion']}")

    # Rewrite examples
    if analysis["rewrite_examples"]:
        report.append(f"\n{'─'*60}")
        report.append("REWRITE EXAMPLES:")
        for ex in analysis["rewrite_examples"][:3]:
            report.append(f"  BEFORE: {ex['before']}")
            report.append(f"  AFTER:  {ex['after']}")
            report.append(f"  WHY:    {ex['reason']}")
            report.append("")

    return "\n".join(report)
