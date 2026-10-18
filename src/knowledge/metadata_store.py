import json
import os
import tempfile

from src.core.settings import KNOWLEDGE_INDEX_FILE


class MetadataStore:

    def __init__(self):

        KNOWLEDGE_INDEX_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self):

        if not KNOWLEDGE_INDEX_FILE.exists():
            return {}

        with KNOWLEDGE_INDEX_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    def save(
        self,
        data: dict,
    ):
        KNOWLEDGE_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=KNOWLEDGE_INDEX_FILE.parent,
                prefix=f"{KNOWLEDGE_INDEX_FILE.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = stream.name
                json.dump(data, stream, indent=4)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, KNOWLEDGE_INDEX_FILE)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
