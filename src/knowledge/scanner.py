from src.core.logger import logger
from src.core.settings import KNOWLEDGE_PATH
from src.knowledge.hashing import calculate_file_hash
from src.knowledge.models import KnowledgeDocument


def scan_library() -> list[KnowledgeDocument]:
    """
    Scan the local knowledge library for PDF documents.

    Returns
    -------
    list[KnowledgeDocument]
        Documents discovered in the knowledge library.
    """

    documents: list[KnowledgeDocument] = []

    for pdf in sorted(KNOWLEDGE_PATH.rglob("*.pdf")):
        relative_path = pdf.relative_to(KNOWLEDGE_PATH)

        document = KnowledgeDocument(
            path=pdf,
            name=pdf.name,
            category=relative_path.parts[0],
            size=pdf.stat().st_size,
            hash=calculate_file_hash(pdf),
        )

        documents.append(document)

    categories = {
        document.category
        for document in documents
    }

    logger.info(
        "Knowledge library scan complete "
        "(documents=%d categories=%d)",
        len(documents),
        len(categories),
    )

    return documents