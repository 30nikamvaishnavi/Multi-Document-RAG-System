"""
Document ingestion module.
Extracts text from supported document formats while retaining source locations.
"""
import csv
import html
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List

from docx import Document as WordDocument
from openpyxl import load_workbook
from pypdf import PdfReader
from pptx import Presentation


class PDFIngestion:
    """Reads common office and text documents into citation-aware text units."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm",
        ".csv", ".xlsx", ".pptx",
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _unit(source: str, text: str, location: str, page: int = 1) -> Dict[str, Any]:
        return {"source": source, "page": page, "location": location, "text": text}

    def _extract_pdf(self, path: Path) -> List[Dict[str, Any]]:
        reader = PdfReader(str(path))
        return [
            self._unit(path.name, page.extract_text() or "", f"Page {i + 1}", i + 1)
            for i, page in enumerate(reader.pages)
        ]

    @staticmethod
    def _extract_docx(path: Path) -> List[Dict[str, Any]]:
        document = WordDocument(path)
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            parts.extend(
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in table.rows
            )
        return [PDFIngestion._unit(path.name, "\n".join(parts), "Document")]

    @staticmethod
    def _extract_xlsx(path: Path) -> List[Dict[str, Any]]:
        workbook = load_workbook(path, read_only=True, data_only=True)
        units = []
        try:
            for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
                rows = []
                for row in sheet.iter_rows(values_only=True):
                    values = [str(value) for value in row if value is not None]
                    if values:
                        rows.append(" | ".join(values))
                units.append(
                    PDFIngestion._unit(
                        path.name,
                        "\n".join(rows),
                        f"Sheet: {sheet.title}",
                        sheet_index,
                    )
                )
        finally:
            workbook.close()
        return units

    @staticmethod
    def _extract_pptx(path: Path) -> List[Dict[str, Any]]:
        presentation = Presentation(path)
        units = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text.strip():
                    parts.append(shape.text.strip())
                if shape.has_table:
                    parts.extend(
                        " | ".join(cell.text.strip() for cell in row.cells)
                        for row in shape.table.rows
                    )
            units.append(
                PDFIngestion._unit(
                    path.name, "\n".join(parts), f"Slide {slide_index}", slide_index
                )
            )
        return units

    @staticmethod
    def _extract_csv(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = [
                " | ".join(cell.strip() for cell in row)
                for row in csv.reader(file)
                if any(cell.strip() for cell in row)
            ]
        return [PDFIngestion._unit(path.name, "\n".join(rows), "Document")]

    @staticmethod
    def _extract_html(path: Path) -> List[Dict[str, Any]]:
        class TextExtractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.parts: List[str] = []
                self.hidden_depth = 0

            def handle_starttag(self, tag: str, attrs: Any) -> None:
                if tag.lower() in {"script", "style"}:
                    self.hidden_depth += 1

            def handle_endtag(self, tag: str) -> None:
                if tag.lower() in {"script", "style"} and self.hidden_depth:
                    self.hidden_depth -= 1

            def handle_data(self, data: str) -> None:
                if not self.hidden_depth and data.strip():
                    self.parts.append(data.strip())

        parser = TextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return [
            PDFIngestion._unit(
                path.name, html.unescape("\n".join(parser.parts)), "Document"
            )
        ]

    @staticmethod
    def _extract_text_file(path: Path) -> List[Dict[str, Any]]:
        return [
            PDFIngestion._unit(
                path.name, path.read_text(encoding="utf-8", errors="replace"), "Document"
            )
        ]

    def extract_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text units from one supported file."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix == ".docx":
            return self._extract_docx(path)
        if suffix == ".xlsx":
            return self._extract_xlsx(path)
        if suffix == ".pptx":
            return self._extract_pptx(path)
        if suffix == ".csv":
            return self._extract_csv(path)
        if suffix in {".html", ".htm"}:
            return self._extract_html(path)
        return self._extract_text_file(path)

    def extract_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Compatibility wrapper for callers that used the original PDF-only API."""
        return self._extract_pdf(Path(pdf_path))

    def extract_all(self, document_dir: str) -> List[Dict[str, Any]]:
        """Extract text from all supported files in a directory."""
        if not os.path.isdir(document_dir):
            raise FileNotFoundError(f"Document directory does not exist: {document_dir}")

        all_pages = []
        for path in sorted(Path(document_dir).iterdir()):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                all_pages.extend(self.extract_file(str(path)))
        return all_pages

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split extracted text into overlapping chunks, preserving source locations."""
        chunks = []
        for page in pages:
            text = page["text"]
            source = page["source"]
            page_num = page["page"]
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "source": source,
                        "page": page_num,
                        "location": page.get("location", f"Page {page_num}"),
                        "text": chunk_text,
                    })
                start += self.chunk_size - self.chunk_overlap
        return chunks

    def load_and_chunk(self, document_dir: str) -> List[Dict[str, Any]]:
        """Full pipeline: read supported documents -> chunk -> return documents."""
        return self.chunk_pages(self.extract_all(document_dir))
