from pathlib import Path

APP_NAME = "Bridge AI"

APP_DESCRIPTION = "Enterprise Modernization Assistant"

PAGE_ICON = "🌉"

PAGE_LAYOUT = "wide"

OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

MAX_HISTORY = 20

THINKING_MESSAGE = "Thinking..."

APP_VERSION = "0.1.0"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CHAT_PROVIDER = "ollama"

OLLAMA_CHAT_MODEL = OLLAMA_MODEL

EMBED_MODEL = "nomic-embed-text"

# ChromaDB
CHROMA_PATH = "data/chroma"
CHROMA_COLLECTION = "bridge_documents"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = PROJECT_ROOT / "docs" / "knowledge"

DATA_PATH = PROJECT_ROOT / "data"

METADATA_PATH = DATA_PATH / "metadata"

KNOWLEDGE_INDEX_FILE = METADATA_PATH / "knowledge_index.json"
