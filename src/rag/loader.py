from pathlib import Path

from pypdf import PdfReader

from src.rag.models import PdfPage

def load_pdf(file) -> list[PdfPage]:
    """
    Extract all text from an uploaded PDF.

    Raises:
        ValueError: If the PDF cannot be read or is invalid
        Exception: For other PDF processing errors
    """

    try:
        reader = PdfReader(file)
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {str(e)}") from e

    if not reader.pages:
        raise ValueError("PDF has no pages or could not be parsed")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            pages.append(
                PdfPage(
                    page=page_number,
                    text=text,
                )
            )
        except Exception as e:
            raise ValueError(
                f"Failed to extract text from PDF page {page_number}: {str(e)}"
            ) from e

    return pages
