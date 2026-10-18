from src.rag.embeddings import embed_text
from src.rag.vectordb import VectorDatabase


def retrieve_context(
    db,
    question: str,
    top_k: int = 5,
):

    question_embedding = embed_text(question)

    return db.search(
        embedding=question_embedding,
        top_k=top_k,
    )

    return results