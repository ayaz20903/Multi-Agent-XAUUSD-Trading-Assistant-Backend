import fcntl
import logging
import tempfile
from pathlib import Path

from gold_ai.vector_store import CHROMA_DIR

logger = logging.getLogger(__name__)

_LOCK_DIR = Path(tempfile.gettempdir()) / "gold_ai_locks"


def ensure_vector_store():
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _LOCK_DIR / "chroma_init.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
                logger.info("Vector store already exists at %s", CHROMA_DIR)
                return

            logger.info("Vector store not found. Running ingestion...")

            from gold_ai.ingest import ingest_documents

            ingest_documents()

            logger.info("Ingestion complete.")
        except Exception:
            logger.exception("Vector store initialization failed")
            raise
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
