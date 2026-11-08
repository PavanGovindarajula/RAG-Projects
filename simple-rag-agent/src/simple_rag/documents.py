"""Load explicitly configured files under a trusted document root."""
from dataclasses import dataclass
from pathlib import Path


class DocumentError(ValueError):
    """A configured document could not be safely loaded."""


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def load_documents(root: Path, names: list[str], max_bytes: int = 10_000_000) -> list[Document]:
    root = root.resolve(strict=True)
    if not root.is_dir() or not names:
        raise DocumentError("Provide a document directory and at least one file")
    result = []
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise DocumentError(f"Document is missing or outside the configured root: {name}")
        if path.stat().st_size > max_bytes:
            raise DocumentError(f"Document exceeds {max_bytes} bytes: {name}")
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise DocumentError(f"Encrypted PDF is not supported: {name}")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            raise DocumentError(f"Unsupported document type: {name}")
        if not text.strip():
            raise DocumentError(f"No extractable text in document: {name}")
        result.append(Document(path.relative_to(root).as_posix(), text))
    return result
