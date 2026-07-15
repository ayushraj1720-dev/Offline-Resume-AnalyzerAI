"""
app.py
======
Main Streamlit application for ResumeSense AI.

This is the user-facing interface that orchestrates the entire analysis pipeline:
1. Upload PDF resume
2. Extract text (pdf_extractor)
3. Analyze sections (section_detector)
4. Score ATS (ats_scorer)
5. Extract skills (skill_extractor)
6. Analyze strengths/weaknesses (strength_weakness)
7. Optionally match to JD (jd_matcher)
8. Generate report (report_generator)

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import tempfile

# Import all analysis modules
from config import (
    APP_TITLE,
    APP_TAGLINE,
    PRIMARY_COLOR,
    BG_COLOR,
    CARD_COLOR,
    TEXT_COLOR,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
)
from utils.pdf_extractor import extract_text_from_pdf, get_pdf_metadata
from utils.section_detector import (
    detect_resume_sections,
    extract_contact_info,
    format_sections_for_display,
)
from utils.skill_extractor import extract_skills_from_resume
from utils.ats_scorer import score_resume
from utils.jd_matcher import match_resume_to_jd, extract_jd_requirements
from utils.strength_weakness import (
    analyze_resume_strengths_weaknesses,
    generate_improvement_plan,
    format_strength_weakness_report,
)
from utils.report_generator import generate_analysis_report


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark theme CSS
st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: #{PRIMARY_COLOR[0]:02x}{PRIMARY_COLOR[1]:02x}{PRIMARY_COLOR[2]:02x};
        --text-color: #{TEXT_COLOR[0]:02x}{TEXT_COLOR[1]:02x}{TEXT_COLOR[2]:02x};
    }}
    
    .main {{
        background-color: #{BG_COLOR[0]:02x}{BG_COLOR[1]:02x}{BG_COLOR[2]:02x};
        color: #{TEXT_COLOR[0]:02x}{TEXT_COLOR[1]:02x}{TEXT_COLOR[2]:02x};
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    
    .metric-card {{
        background-color: #{CARD_COLOR[0]:02x}{CARD_COLOR[1]:02x}{CARD_COLOR[2]:02x};
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "sections" not in st.session_state:
    st.session_state.sections = None
if "ats_result" not in st.session_state:
    st.session_state.ats_result = None
if "skills_result" not in st.session_state:
    st.session_state.skills_result = None
if "strength_weakness_result" not in st.session_state:
    st.session_state.strength_weakness_result = None
if "jd_match_result" not in st.session_state:
    st.session_state.jd_match_result = None
if "pdf_metadata" not in st.session_state:
    st.session_state.pdf_metadata = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def reset_analysis():
    """Clear all analysis results."""
    st.session_state.resume_text = None
    st.session_state.sections = None
    st.session_state.ats_result = None
    st.session_state.skills_result = None
    st.session_state.strength_weakness_result = None
    st.session_state.jd_match_result = None
    st.session_state.pdf_metadata = None


def run_full_analysis(resume_text, sections, jd_text=None):
    """Run the complete analysis pipeline."""
    # Extract skills
    skills_result = extract_skills_from_resume(resume_text, sections)

    # Score ATS
    ats_result = score_resume(resume_text, sections, jd_text)

    # Analyze strengths/weaknesses
    strength_weakness_result = analyze_resume_strengths_weaknesses(
        resume_text, sections
    )

    # Optional: JD matching
    jd_match_result = None
    if jd_text:
        jd_match_result = match_resume_to_jd(resume_text, jd_text, sections)

    return {
        "skills_result": skills_result,
        "ats_result": ats_result,
        "strength_weakness_result": strength_weakness_result,
        "jd_match_result": jd_match_result,
    }


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

# Header
st.markdown(
    f"<h1 style='text-align: center; color: #{PRIMARY_COLOR[0]:02x}{PRIMARY_COLOR[1]:02x}{PRIMARY_COLOR[2]:02x};'>{APP_TITLE}</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align: center; color: {TEXT_COLOR}; font-size: 14px;'>{APP_TAGLINE}</p>",
    unsafe_allow_html=True,
)
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload a PDF resume",
        type="pdf",
        help="Upload your resume as a PDF file (max 10 MB)",
    )

    if uploaded_file:
        st.success("✓ File uploaded successfully!")
        st.markdown(f"**File name:** {uploaded_file.name}")
        st.markdown(f"**File size:** {uploaded_file.size / 1024:.1f} KB")

        # Extract on upload
        with st.spinner("Extracting text from PDF..."):
            pdf_bytes = uploaded_file.getvalue()

            pdf_result = extract_text_from_pdf(pdf_bytes)

            if pdf_result["error"]:
                st.error(f"❌ {pdf_result['error']}")
                uploaded_file = None
            else:
                st.session_state.resume_text = pdf_result["text"]
                st.session_state.pdf_metadata = get_pdf_metadata(pdf_bytes)

                st.markdown(f"**Pages:** {pdf_result['pages']}")
                st.markdown(f"**Words:** {pdf_result['word_count']}")
    if uploaded_file and st.session_state.resume_text:
        st.divider()
        st.markdown("### 🚀 Analysis")

        if st.button("Run Full Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing resume..."):
                # Detect sections
                sections = detect_resume_sections(st.session_state.resume_text)
                st.session_state.sections = sections

                # Run analysis
                results = run_full_analysis(
                    st.session_state.resume_text,
                    sections,
                    jd_text=None,  # JD can be added in the main UI
                )

                st.session_state.skills_result = results["skills_result"]
                st.session_state.ats_result = results["ats_result"]
                st.session_state.strength_weakness_result = (
                    results["strength_weakness_result"]
                )

            st.success("✓ Analysis complete!")

        if st.button("Reset", use_container_width=True):
            reset_analysis()
            st.rerun()

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        **ResumeSense AI** analyzes your resume offline using:
        - ATS scoring
        - Skill extraction
        - Job matching
        - Strength/weakness analysis
        
        Your resume never leaves your device. 100% private. 100% offline.
        """
    )


# Main content area
if not st.session_state.resume_text:
    # Empty state
    st.info(
        "👈 **Upload a resume to get started!**\n\n"
        "ResumeSense AI will analyze your resume and provide:\n"
        "- ATS score with detailed breakdown\n"
        "- Extracted skills and categories\n"
        "- Strengths and weaknesses\n"
        "- Actionable improvement suggestions\n"
        "- (Optional) Job description matching"
    )
else:
    # Analysis results
    if not st.session_state.ats_result:
        st.warning("⚠️ Run analysis from the sidebar to see results.")
    else:
        # ===================================================================
        # TAB 1: ATS SCORE
        # ===================================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🎯 ATS Score", "🛠️ Skills", "💪 Analysis", "📊 JD Match", "📥 Download"]
        )

        with tab1:
            col1, col2 = st.columns([1, 2])

            with col1:
                score = st.session_state.ats_result["overall_score"]
                score_color = "🟢" if score >= 70 else "🟡" if score >= 60 else "🔴"

                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 20px;'>
                        <p style='font-size: 48px; font-weight: bold; color: #{PRIMARY_COLOR[0]:02x}{PRIMARY_COLOR[1]:02x}{PRIMARY_COLOR[2]:02x};'>
                            {score}
                        </p>
                        <p style='font-size: 18px; color: gray;'>/ 100</p>
                        <p style='font-size: 24px;'>{score_color}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                factors = st.session_state.ats_result["factors"]

                for factor_name, factor_score in factors.items():
                    label = factor_name.replace("_", " ").title()
                    percentage = factor_score * 100

                    st.metric(label, f"{percentage:.0f}%", border=True)

            st.divider()

            # Feedback
            st.subheader("💡 Key Feedback")

            col1, col2 = st.columns(2)

            with col1:
                if st.session_state.ats_result["strengths"]:
                    st.markdown("**✅ Strengths:**")
                    for strength in st.session_state.ats_result["strengths"][:3]:
                        st.markdown(f"- {strength}")

            with col2:
                if st.session_state.ats_result["weaknesses"]:
                    st.markdown("**⚠️ Areas to Improve:**")
                    for weakness in st.session_state.ats_result["weaknesses"][:3]:
                        st.markdown(f"- {weakness}")

        # ===================================================================
        # TAB 2: SKILLS
        # ===================================================================
        with tab2:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Unique Skills",
                    st.session_state.skills_result["total_unique"],
                    border=True,
                )

            with col2:
                st.metric(
                    "Total Mentions",
                    st.session_state.skills_result["total_mentions"],
                    border=True,
                )

            with col3:
                st.metric(
                    "Categories",
                    len(st.session_state.skills_result["skill_categories"]),
                    border=True,
                )

            st.divider()

            # Skills by category
            for category, skills in st.session_state.skills_result[
                "skill_categories"
            ].items():
                with st.expander(
                    f"{category.replace('_', ' ').title()} ({len(skills)})"
                ):
                    skills_df = pd.DataFrame(
                        [
                            {
                                "Skill": s["name"],
                                "Confidence": f"{s['confidence']:.0%}",
                                "Count": s["count"],
                                "Source": s["source"],
                            }
                            for s in sorted(skills, key=lambda x: x["confidence"], reverse=True)
                        ]
                    )
                    st.dataframe(skills_df, use_container_width=True, hide_index=True)

        # ===================================================================
        # TAB 3: STRENGTHS & WEAKNESSES
        # ===================================================================
        with tab3:
            analysis = st.session_state.strength_weakness_result

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✅ Strengths")
                for strength in analysis["strengths"]:
                    st.markdown(f"- {strength}")

            with col2:
                st.markdown("### ⚠️ Weaknesses")
                for weakness in analysis["weaknesses"]:
                    st.markdown(f"- {weakness}")

            st.divider()

            # Improvement plan
            st.markdown("### 🎯 Improvement Plan")
            plan = generate_improvement_plan(analysis)
            for item in plan:
                st.markdown(item)

            st.divider()

            # Rewrite examples
            if analysis["rewrite_examples"]:
                st.markdown("### ✏️ Rewrite Examples")
                for ex in analysis["rewrite_examples"]:
                    with st.expander(
                        f"Example: {ex['before'][:50]}..."
                    ):
                        st.markdown(f"**Before:** {ex['before']}")
                        st.markdown(f"**After:** {ex['after']}")
                        st.markdown(f"**Why:** {ex['reason']}")

        # ===================================================================
        # TAB 4: JD MATCHING
        # ===================================================================
        with tab4:
            st.markdown("### 📄 Job Description Matching")

            jd_text = st.text_area(
                "Paste job description here (optional)",
                placeholder="Paste the job description you want to match against...",
                height=150,
            )

            if jd_text:
                if st.button("🔍 Match Resume to JD", type="primary"):
                    with st.spinner("Analyzing job description match..."):
                        jd_match = match_resume_to_jd(
                            st.session_state.resume_text,
                            jd_text,
                            st.session_state.sections,
                        )
                        st.session_state.jd_match_result = jd_match

            if st.session_state.jd_match_result:
                match_result = st.session_state.jd_match_result

                # Score
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Match Score",
                        f"{match_result['overall_match_score']:.0f}/100",
                        border=True,
                    )

                with col2:
                    st.metric(
                        "Grade",
                        match_result["match_grade"],
                        border=True,
                    )

                with col3:
                    skill_match = match_result["skill_match_result"]
                    st.metric(
                        "Skills Match",
                        f"{skill_match['matched_count']}/{skill_match['jd_total']}",
                        border=True,
                    )

                st.divider()

                # Missing skills
                missing = match_result["skill_match_result"]["missing_skills"]
                if missing:
                    st.markdown("### ❌ Missing Skills")
                    missing_names = [s["name"] for s in missing[:10]]
                    st.markdown(", ".join(missing_names))

                # Recommendations
                if match_result["recommendations"]:
                    st.markdown("### 💡 Recommendations")
                    for rec in match_result["recommendations"]:
                        st.markdown(f"- {rec}")

        # ===================================================================
        # TAB 5: DOWNLOAD
        # ===================================================================
        with tab5:
            st.markdown("### 📥 Download Report")

            if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):
                    # Create temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        pdf_path = generate_analysis_report(
                            output_filename=tmp_file.name,
                            resume_text=st.session_state.resume_text,
                            ats_score_result=st.session_state.ats_result,
                            sections=st.session_state.sections,
                            skills_result=st.session_state.skills_result,
                            strength_weakness_result=st.session_state.strength_weakness_result,
                            jd_match_result=st.session_state.jd_match_result,
                        )

                    # Read and offer download
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()

                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_data,
                        file_name=f"Resume_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

                    st.success("✓ Report generated successfully!")

                    # Cleanup
                    os.unlink(pdf_path)

        # ===================================================================
        # View raw extracted text
        # ===================================================================
        with st.expander("📝 View Extracted Resume Text"):
            st.text_area(
                "Extracted text (read-only):",
                value=st.session_state.resume_text,
                height=300,
                disabled=True,
            )

        # ===================================================================
        # View detected sections
        # ===================================================================
        with st.expander("🔍 View Detected Sections"):
            if st.session_state.sections:
                for section_name, section_text in st.session_state.sections.items():
                    if section_text.strip():
                        with st.expander(f"{section_name.upper()}"):
                            st.text(section_text[:500] + "..." if len(section_text) > 500 else section_text)
