"""
Indexing tracker for the Agentic RAG system.

Prevents re-indexing the same document twice within a single running
session (based on file content, not just filename), while allowing
multiple distinct documents to coexist in the Qdrant store.

NOTE: This tracker is intentionally IN-MEMORY ONLY (not persisted to
disk). The vector store (Qdrant) is also in-memory and resets on every
app restart — if this tracker's log persisted to disk across restarts
while Qdrant did not, the tracker would incorrectly believe files are
already indexed when the vector store is actually empty, silently
causing zero search results. Keeping both in-memory keeps them in sync.

Usage:
    tracker = IndexingTracker()  # lives for this process only

    if tracker.is_indexed(file_path):
        print(f"Skipping {file_path} — already indexed this session")
    else:
        # ... run your existing indexing pipeline ...
        tracker.mark_indexed(file_path)
"""

import hashlib
from datetime import datetime


class IndexingTracker:
    def __init__(self, log_path: str = None):
        # log_path kept as an accepted (ignored) argument for backward
        # compatibility with existing call sites; tracker is in-memory only.
        self.log = {}  # {file_hash: {"filename": ..., "indexed_at": ...}}

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """
        Content-based hash (MD5). Using content rather than filename
        means: renaming a file won't cause a false re-index, and
        editing a file's content WILL correctly trigger re-indexing.
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_indexed(self, file_path: str) -> bool:
        """Check if this exact file content has already been indexed this session."""
        file_hash = self._compute_file_hash(file_path)
        return file_hash in self.log

    def mark_indexed(self, file_path: str):
        """Record a file as indexed for this session, keyed by its content hash."""
        file_hash = self._compute_file_hash(file_path)
        self.log[file_hash] = {
            "filename": file_path,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        }

    def get_indexed_files(self) -> list:
        """Return a list of all currently-indexed filenames this session (for UI display)."""
        return [entry["filename"] for entry in self.log.values()]

    def remove_entry(self, file_path: str):
        """Remove a file's tracking entry, making it eligible for re-indexing again."""
        file_hash = self._compute_file_hash(file_path)
        if file_hash in self.log:
            del self.log[file_hash]