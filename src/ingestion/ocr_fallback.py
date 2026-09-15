"""
OCR fallback for scanned PDFs with no text layer.

Some grantee documents are submitted as scanned images rather than
digital PDFs, which pdfplumber cannot read. Each page is rasterised and
run through Tesseract. The resulting text is noisier than native
extraction, which is recorded in the document's ocr_quality_score.
"""

import os
from pathlib import Path

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


CACHE_DIR = Path("data/ocr_cache")


def has_text_layer(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    Check whether a PDF has an extractable text layer.
    Samples the first few pages rather than the whole document.
    """
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        n = min(sample_pages, len(pdf.pages))
        for i in range(n):
            text = pdf.pages[i].extract_text()
            if text and len(text.strip()) > 50:
                return True
    return False


def ocr_document(pdf_path: str, dpi: int = 150) -> str:
    """
    Run OCR on a scanned PDF and return the extracted text.
    Results are cached, since OCR is expensive.

    Args:
        pdf_path: path to the scanned PDF
        dpi:      rasterisation resolution (150 is a good balance)

    Returns:
        extracted text as a single string
    """
    if not OCR_AVAILABLE:
        raise ImportError(
            "OCR requires pytesseract and pdf2image. Install with:\n"
            "  brew install tesseract poppler\n"
            "  pip install pytesseract pdf2image"
        )

    pdf_path = Path(pdf_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{pdf_path.stem}.txt"

    if cache_file.exists():
        print(f"    Using cached OCR for {pdf_path.stem}")
        return cache_file.read_text(encoding="utf-8")

    print(f"    Running OCR on {pdf_path.stem} — this may take a few minutes")

    images = convert_from_path(str(pdf_path), dpi=dpi)
    pages_text = []

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        pages_text.append(text)
        if (i + 1) % 10 == 0:
            print(f"      OCR progress: {i + 1}/{len(images)} pages")

    full_text = "\n".join(pages_text)

    cache_file.write_text(full_text, encoding="utf-8")
    print(f"    OCR complete — {len(full_text):,} characters extracted")

    return full_text
