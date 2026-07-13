"""
utils/pdf_extractor.py
======================
Extracts clean text from a PDF resume — 100% locally, using PyMuPDF (fitz).

WHY PyMuPDF?
    - Pure C library wrapped in Python, no internet needed.
    - Handles multi-column layouts, headers/footers, and Unicode fonts better
      than pdfminer or pypdf.
    - Extremely fast (a 2-page resume extracts in under 0.1 seconds).
    - MIT-compatible license (AGPL for open-source use, which covers hackathons).

WHAT THIS MODULE DOES:
    1. Accepts a PDF as a file path OR raw bytes (Streamlit uploads give bytes).
    2. Iterates through every page and extracts text block-by-block.
    3. Cleans up excessive whitespace.
    4. Returns the raw text string + metadata (page count, word count, file size).

WHAT IT DOES NOT DO:
    - OCR (scanned image-only PDFs won't work — we detect and warn the user).
    - Form fields (most resumes don't use fillable PDF forms).
"""

import io
import re
from typing import Union

import fitz  # PyMuPDF — installed via `pip install PyMuPDF`

from config import MAX_UPLOAD_SIZE_MB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_input: Union[str, bytes]) -> dict:
    """
    Extract all text from a PDF file.

    Parameters
    ----------
    pdf_input : str or bytes
        Either a filesystem path to a .pdf file, or raw PDF bytes
        (as returned by Streamlit's st.file_uploader via .read()).

    Returns
    -------
    dict with keys:
        "text"       : str   — full extracted text, pages joined by newlines
        "pages"      : int   — number of pages in the PDF
        "word_count" : int   — approximate word count of extracted text
        "char_count" : int   — character count
        "is_scanned" : bool  — True if the PDF appears to be image-only (no selectable text)
        "error"      : str or None — error message if extraction failed

    Example
    -------
        result = extract_text_from_pdf(uploaded_file.read())
        if result["error"]:
            st.error(result["error"])
        else:
            resume_text = result["text"]
    """
    result = {
        "text": "",
        "pages": 0,
        "word_count": 0,
        "char_count": 0,
        "is_scanned": False,
        "error": None,
    }

    # ------------------------------------------------------------------
    # 1. Validate input size (before opening — saves time on huge files)
    # ------------------------------------------------------------------
    if isinstance(pdf_input, bytes):
        size_mb = len(pdf_input) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            result["error"] = (
                f"File is {size_mb:.1f} MB — max allowed is {MAX_UPLOAD_SIZE_MB} MB. "
                "Please compress your PDF or remove embedded images."
            )
            return result

    # ------------------------------------------------------------------
    # 2. Open the PDF with PyMuPDF
    # ------------------------------------------------------------------
    try:
        if isinstance(pdf_input, bytes):
            # fitz can open from a bytes stream directly
            pdf_stream = io.BytesIO(pdf_input)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
        else:
            # It's a filesystem path string
            doc = fitz.open(pdf_input)
    except Exception as e:
        result["error"] = f"Could not open PDF: {e}"
        return result

    result["pages"] = len(doc)

    # ------------------------------------------------------------------
    # 3. Extract text page by page
    # ------------------------------------------------------------------
    all_text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_text("text") returns plain text with layout-aware line breaks.
        # "blocks" mode would give bounding boxes — we don't need those here.
        page_text = page.get_text("text")
        all_text_parts.append(page_text)

    doc.close()

    # ------------------------------------------------------------------
    # 4. Join pages and clean whitespace
    # ------------------------------------------------------------------
    raw_text = "\n".join(all_text_parts)
    cleaned_text = _clean_extracted_text(raw_text)

    # ------------------------------------------------------------------
    # 5. Detect scanned / image-only PDFs
    #    Heuristic: if we got fewer than 50 characters across all pages,
    #    the PDF is likely a scanned image (no selectable text layer).
    # ------------------------------------------------------------------
    if len(cleaned_text.strip()) < 50:
        result["is_scanned"] = True
        result["error"] = (
            "This PDF appears to be a scanned image — no selectable text was found. "
            "Please export your resume as a text-based PDF (not a scanned/photo copy)."
        )
        return result

    # ------------------------------------------------------------------
    # 6. Populate result
    # ------------------------------------------------------------------
    result["text"] = cleaned_text
    result["word_count"] = len(cleaned_text.split())
    result["char_count"] = len(cleaned_text)

    return result


def extract_text_by_page(pdf_input: Union[str, bytes]) -> list[dict]:
    """
    Extract text page by page, returning a list of dicts.

    Useful for advanced features like "which page has the Skills section?"

    Parameters
    ----------
    pdf_input : str or bytes

    Returns
    -------
    list of dicts, each with:
        "page"  : int  — 1-indexed page number
        "text"  : str  — cleaned text for that page
    """
    pages = []

    try:
        if isinstance(pdf_input, bytes):
            doc = fitz.open(stream=io.BytesIO(pdf_input), filetype="pdf")
        else:
            doc = fitz.open(pdf_input)
    except Exception:
        return pages

    for i, page in enumerate(doc):
        page_text = _clean_extracted_text(page.get_text("text"))
        pages.append({"page": i + 1, "text": page_text})

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_extracted_text(raw_text: str) -> str:
    """
    Clean up raw text extracted from a PDF.

    Steps:
        1. Replace Windows-style line endings (\r\n) with \n.
        2. Collapse runs of 3+ blank lines into a single blank line.
           (PDFs often insert blank lines between every element.)
        3. Strip leading/trailing whitespace from each line.
        4. Remove lines that are just whitespace or a single non-word character.
        5. Strip the whole text.

    Parameters
    ----------
    raw_text : str

    Returns
    -------
    str — cleaned text, still with meaningful paragraph breaks preserved.
    """
    # Step 1: Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Step 2: Strip each line
    lines = [line.strip() for line in text.split("\n")]

    # Step 3: Remove noise lines (empty or single punctuation)
    cleaned_lines = []
    for line in lines:
        # keep the line if it has at least one letter or digit
        if re.search(r"[A-Za-z0-9]", line):
            cleaned_lines.append(line)
        elif line == "":
            cleaned_lines.append("")  # preserve paragraph breaks

    # Step 4: Collapse 3+ consecutive blank lines into 2
    result_lines = []
    blank_count = 0
    for line in cleaned_lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result_lines.append("")
        else:
            blank_count = 0
            result_lines.append(line)

    return "\n".join(result_lines).strip()


def get_pdf_metadata(pdf_input: Union[str, bytes]) -> dict:
    """
    Return basic metadata from the PDF's document properties.

    Returns
    -------
    dict with keys: title, author, creator, pages
    """
    metadata = {"title": "", "author": "", "creator": "", "pages": 0}

    try:
        if isinstance(pdf_input, bytes):
            doc = fitz.open(stream=io.BytesIO(pdf_input), filetype="pdf")
        else:
            doc = fitz.open(pdf_input)

        meta = doc.metadata  # dict from PyMuPDF
        metadata["title"] = meta.get("title", "") or ""
        metadata["author"] = meta.get("author", "") or ""
        metadata["creator"] = meta.get("creator", "") or ""
        metadata["pages"] = len(doc)
        doc.close()
    except Exception:
        pass

    return metadata
