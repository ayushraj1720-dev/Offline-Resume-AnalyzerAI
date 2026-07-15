"""
utils/report_generator.py
=========================
Generate professional PDF reports of resume analysis.

WHY THIS MODULE EXISTS:
    Candidates want a downloadable report they can save, print, or share.
    This module generates a clean, professional PDF containing:
    - ATS score and breakdown
    - Detected sections
    - Extracted skills
    - Strengths and weaknesses
    - Improvement suggestions
    - JD match (if provided)

TECHNOLOGY:
    We use fpdf2 (lightweight, offline-friendly) instead of reportlab.
    No external services, no internet required.
    Reports are generated on-device in seconds.
"""

from datetime import datetime
from typing import Optional

from fpdf import FPDF


# ---------------------------------------------------------------------------
# PDF report generator class
# ---------------------------------------------------------------------------

class ResumeAnalysisReport:
    """Generate a professional PDF report of resume analysis."""

    def __init__(self, filename: str = "resume_analysis.pdf"):
        """
        Initialize report generator.

        Parameters
        ----------
        filename : str
            Output PDF filename
        """
        self.filename = filename
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_font("Helvetica", size=11)

        # Color scheme (from config)
        self.primary_color = (124, 58, 237)      # violet
        self.accent_color = (59, 130, 246)       # blue
        self.success_color = (34, 197, 94)       # green
        self.warning_color = (234, 179, 8)       # amber
        self.text_color = (31, 41, 55)           # slate-900

    def add_header(self, title: str, subtitle: str = ""):
        """Add a header section with title and optional subtitle."""
        # Title
        self.pdf.set_font("Helvetica", "B", size=20)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.multi_cell(0, 10, title, align="C")

        # Subtitle
        if subtitle:
            self.pdf.set_font("Helvetica", "I", size=11)
            self.pdf.set_text_color(100, 100, 100)
            self.pdf.multi_cell(0, 8, subtitle, align="C")

        self.pdf.ln(5)

    def add_section_title(self, title: str):
        """Add a section title with underline."""
        self.pdf.set_font("Helvetica", "B", size=14)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.multi_cell(0, 10, title)
        self.pdf.set_line_width(0.5)
        self.pdf.set_draw_color(*self.primary_color)
        self.pdf.line(10, self.pdf.get_y(), 200, self.pdf.get_y())
        self.pdf.ln(3)

    def add_subsection_title(self, title: str):
        """Add a subsection title."""
        self.pdf.set_font("Helvetica", "B", size=12)
        self.pdf.set_text_color(*self.accent_color)
        self.pdf.multi_cell(0, 8, title)
        self.pdf.ln(2)

    def add_text(self, text: str, bold: bool = False, color: tuple = None):
        """Add body text."""
        style = "B" if bold else ""
        self.pdf.set_font("Helvetica", style, size=10)
        if color:
            self.pdf.set_text_color(*color)
        else:
            self.pdf.set_text_color(*self.text_color)
        self.pdf.multi_cell(0, 6, text)
        self.pdf.ln(2)

    def add_key_value_pair(self, key: str, value: str):
        """Add a key-value pair (like Score: 75/100)."""
        self.pdf.set_font("Helvetica", "B", size=11)
        self.pdf.set_text_color(*self.primary_color)
        self.pdf.cell(0, 8, f"{key}:", ln=False)

        self.pdf.set_font("Helvetica", size=11)
        self.pdf.set_text_color(*self.text_color)
        self.pdf.cell(0, 8, str(value), ln=True)

    def add_score_box(self, score: float, max_score: float = 100):
        """Add a prominent score box."""
        percentage = (score / max_score) * 100
        color = self.success_color if score >= 70 else self.warning_color

        self.pdf.set_font("Helvetica", "B", size=24)
        self.pdf.set_text_color(*color)
        self.pdf.multi_cell(0, 12, f"{score}/{max_score}", align="C")

        self.pdf.set_font("Helvetica", size=10)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.multi_cell(0, 8, f"({percentage:.0f}%)", align="C")

    def add_bullet_list(self, items: list):
        """Add a bulleted list of items."""
        self.pdf.set_font("Helvetica", size=10)
        self.pdf.set_text_color(*self.text_color)

        for item in items:
            self.pdf.multi_cell(0, 6, f"• {item}")

        self.pdf.ln(2)

    def add_table(self, headers: list, data: list):
        """Add a simple table."""
        self.pdf.set_font("Helvetica", "B", size=10)
        self.pdf.set_text_color(*self.primary_color)

        # Headers
        col_width = 190 / len(headers)
        for header in headers:
            self.pdf.cell(col_width, 8, header, border=1)
        self.pdf.ln()

        # Data rows
        self.pdf.set_font("Helvetica", size=9)
        self.pdf.set_text_color(*self.text_color)

        for row in data:
            for i, cell in enumerate(row):
                self.pdf.cell(col_width, 8, str(cell), border=1)
            self.pdf.ln()

        self.pdf.ln(3)

    def generate_full_report(
        self,
        resume_text: str,
        ats_score_result: dict,
        sections: dict,
        skills_result: dict,
        strength_weakness_result: dict,
        jd_match_result: Optional[dict] = None,
    ) -> str:
        """
        Generate complete analysis report.

        Parameters
        ----------
        resume_text : str
        ats_score_result : dict
            From ats_scorer.score_resume()
        sections : dict
            From section_detector.detect_resume_sections()
        skills_result : dict
            From skill_extractor.extract_skills_from_resume()
        strength_weakness_result : dict
            From strength_weakness.analyze_resume_strengths_weaknesses()
        jd_match_result : dict, optional
            From jd_matcher.match_resume_to_jd()

        Returns
        -------
        str — path to generated PDF file
        """
        # ===================================================================
        # PAGE 1: Cover + ATS Score
        # ===================================================================
        self.add_header(
            "Resume Analysis Report",
            "Powered by ResumeSense AI",
        )

        self.add_text(f"Generated: {datetime.now().strftime('%B %d, %Y')}", color=(100, 100, 100))
        self.pdf.ln(10)

        # ATS Score
        self.add_section_title("ATS Score")
        self.add_score_box(ats_score_result["overall_score"], 100)
        self.pdf.ln(5)

        # ATS breakdown
        self.add_subsection_title("Score Breakdown")
        factors = ats_score_result["factors"]
        table_data = [
            [name.replace("_", " ").title(), f"{value:.0%}"]
            for name, value in factors.items()
        ]
        self.add_table(["Factor", "Score"], table_data)

        # Key feedback
        if ats_score_result["strengths"]:
            self.add_subsection_title("Strengths")
            self.add_bullet_list(ats_score_result["strengths"][:3])

        if ats_score_result["weaknesses"]:
            self.add_subsection_title("Areas for Improvement")
            self.add_bullet_list(ats_score_result["weaknesses"][:3])

        # ===================================================================
        # PAGE 2: Sections & Skills
        # ===================================================================
        self.pdf.add_page()

        self.add_section_title("Resume Sections")
        from utils.section_detector import get_detected_sections_summary

        section_summary = get_detected_sections_summary(sections)
        for section_name, summary in section_summary.items():
            status = "✓" if summary["found"] else "✗"
            self.add_text(
                f"{status} {section_name.title()}: {summary['word_count']} words",
                color=self.success_color if summary["found"] else self.warning_color,
            )

        # Skills
        self.add_section_title("Technical Skills")
        self.add_key_value_pair(
            "Skills Found",
            f"{skills_result['total_unique']} unique skills",
        )

        # Top skills by category
        for category, skills in list(skills_result["skill_categories"].items())[:3]:
            skills_str = ", ".join([s["name"] for s in skills[:5]])
            if len(skills) > 5:
                skills_str += f", +{len(skills) - 5} more"
            self.add_text(f"{category.replace('_', ' ').title()}: {skills_str}")

        # ===================================================================
        # PAGE 3: Strengths & Weaknesses
        # ===================================================================
        self.pdf.add_page()

        self.add_section_title("Detailed Analysis")

        self.add_subsection_title("Strengths")
        self.add_bullet_list(strength_weakness_result["strengths"])

        self.add_subsection_title("Weaknesses & Opportunities")
        self.add_bullet_list(strength_weakness_result["weaknesses"])

        # Improvement suggestions
        self.add_subsection_title("How to Improve")
        for suggestion in strength_weakness_result["suggestions"][:4]:
            self.add_text(f"• {suggestion['issue']}", bold=True)
            self.add_text(f"  → {suggestion['suggestion']}")
            self.pdf.ln(1)

        # ===================================================================
        # PAGE 4: JD Match (if provided)
        # ===================================================================
        if jd_match_result:
            self.pdf.add_page()
            self.add_section_title("Job Description Match")

            score = jd_match_result["overall_match_score"]
            grade = jd_match_result["match_grade"]
            self.add_score_box(score, 100)
            self.add_text(f"Grade: {grade}", align="C")
            self.pdf.ln(5)

            # Match breakdown
            self.add_subsection_title("Match Breakdown")
            breakdown_data = [
                ["Keyword Match", f"{jd_match_result['keyword_match_score']:.0%}"],
                ["Semantic Match", f"{jd_match_result['semantic_match_score']:.0%}"],
                [
                    "Skill Match",
                    f"{jd_match_result['skill_match_result']['matched_count']}/{jd_match_result['skill_match_result']['jd_total']}",
                ],
            ]
            self.add_table(["Metric", "Score"], breakdown_data)

            # Missing skills
            missing = jd_match_result["skill_match_result"]["missing_skills"]
            if missing:
                self.add_subsection_title("Missing Skills")
                missing_names = [s["name"] for s in missing[:5]]
                self.add_bullet_list(missing_names)

            # Recommendations
            if jd_match_result["recommendations"]:
                self.add_subsection_title("Recommendations")
                self.add_bullet_list(jd_match_result["recommendations"][:3])

        # ===================================================================
        # Final page: Footer & contact info
        # ===================================================================
        self.pdf.add_page()
        self.add_section_title("Next Steps")

        self.add_text(
            "1. Review the feedback above and prioritize improvements.",
            bold=True,
        )
        self.add_text(
            "2. Implement at least 2-3 suggestions from the 'How to Improve' section.",
            bold=True,
        )
        self.add_text(
            "3. Re-upload your updated resume to see how your score improves.",
            bold=True,
        )

        self.pdf.ln(10)
        self.add_text(
            "ResumeSense AI — 100% Offline Resume Analyzer\n"
            "No cloud services. No data storage. Your resume stays on your device.",
            color=(100, 100, 100),
        )

        # Save and return path
        self.pdf.output(self.filename)
        return self.filename


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_analysis_report(
    output_filename: str,
    resume_text: str,
    ats_score_result: dict,
    sections: dict,
    skills_result: dict,
    strength_weakness_result: dict,
    jd_match_result: Optional[dict] = None,
) -> str:
    """
    Generate a complete PDF analysis report.

    Parameters
    ----------
    output_filename : str
        Path/filename for output PDF (e.g., "resume_analysis.pdf")
    resume_text : str
    ats_score_result : dict
        From ats_scorer.score_resume()
    sections : dict
        From section_detector.detect_resume_sections()
    skills_result : dict
        From skill_extractor.extract_skills_from_resume()
    strength_weakness_result : dict
        From strength_weakness.analyze_resume_strengths_weaknesses()
    jd_match_result : dict, optional

    Returns
    -------
    str — path to generated PDF
    """
    report = ResumeAnalysisReport(output_filename)
    return report.generate_full_report(
        resume_text=resume_text,
        ats_score_result=ats_score_result,
        sections=sections,
        skills_result=skills_result,
        strength_weakness_result=strength_weakness_result,
        jd_match_result=jd_match_result,
    )
