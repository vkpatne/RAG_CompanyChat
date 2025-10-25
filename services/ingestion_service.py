from sentence_transformers import SentenceTransformer
from storage.utils import chunk_text
from storage.index_manager import IndexManager
from core.config import EMBED_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from core.logger import logger
from core.errors import IngestionError
import os

# Use SentenceTransformer to provide consistent formatting; embeddings are handled by LangChain later
embed_model = SentenceTransformer(EMBED_MODEL_PATH)

class IngestionService:
    def __init__(self):
        self.index_manager = IndexManager()

    def ingest_text(self, text):
        try:
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            self.index_manager.add_chunks(chunks)
            logger.info(f'Ingested {len(chunks)} chunks into IndexManager')
            return {'status': 'ingested', 'chunks': len(chunks)}
        except Exception as e:
            logger.error(f'Ingestion failed: {e}')
            raise IngestionError(str(e))

    def ingest_file(self, path: str):
        """
        Extract text from a file (PDF or TXT) and ingest it.
        Raises IngestionError on problems (missing libs, empty content, read errors).
        """
        if not path or not os.path.exists(path):
            raise IngestionError("File not found: " + str(path))

        ext = os.path.splitext(path)[1].lower()
        text = ""
        try:
            if ext == ".pdf":
                try:
                    # pypdf is recommended (package name: pypdf)
                    from pypdf import PdfReader
                except Exception:
                    raise IngestionError("PDF support requires 'pypdf'. Install with: pip install pypdf")
                try:
                    reader = PdfReader(path)
                    pages = []
                    for p in reader.pages:
                        pages.append(p.extract_text() or "")
                    text = "\n\n".join(pages)
                except Exception as e:
                    raise IngestionError(f"Failed to extract text from PDF: {e}")
            else:
                # treat as plain text
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # fallback to latin-1 if utf-8 fails
                    try:
                        with open(path, "r", encoding="latin-1") as f:
                            text = f.read()
                    except Exception as e:
                        raise IngestionError(f"Failed to read text file: {e}")
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(f"Unexpected error reading file: {e}")

        if not text or not text.strip():
            raise IngestionError("No text extracted from file.")

        # reuse existing ingest_text to chunk & save
        return self.ingest_text(text)
