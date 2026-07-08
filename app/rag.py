from pathlib import Path
import re
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


SKIP_SECTION_PREFIXES = (
    "14. Как ассистент",
    "15. Когда ассистент",
    "16. Источники",
)


def load_markdown_documents(docs_dir: Path) -> List[Dict[str, str]]:
    documents = []

    for file_path in docs_dir.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


def tokenize(text: str) -> set[str]:
    """
    Простая токенизация для keyword-части поиска.
    Нужна, чтобы лучше находить короткие вопросы и числа.
    """
    return set(re.findall(r"[a-zа-яё0-9]{2,}", text.lower()))


def split_markdown_by_sections(
    text: str,
    source: str,
) -> List[Dict[str, Any]]:
    """
    Делит markdown-документ по заголовкам второго уровня: ## ...
    Каждый раздел становится отдельным chunk.
    """
    lines = text.splitlines()

    sections = []
    current_title = None
    current_lines = []

    for line in lines:
        if line.startswith("## "):
            if current_title and current_lines:
                sections.append(
                    {
                        "title": current_title,
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            current_title = line.replace("## ", "").strip()
            current_lines = [line]
        else:
            if current_title:
                current_lines.append(line)

    if current_title and current_lines:
        sections.append(
            {
                "title": current_title,
                "text": "\n".join(current_lines).strip(),
            }
        )

    chunks = []

    for section in sections:
        title = section["title"]

        if title.startswith(SKIP_SECTION_PREFIXES):
            continue

        chunks.append(
            {
                "source": source,
                "title": title,
                "text": section["text"],
            }
        )

    return chunks


def build_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    all_chunks = []

    for document in documents:
        chunks = split_markdown_by_sections(
            text=document["text"],
            source=document["source"],
        )

        for chunk in chunks:
            chunk["chunk_id"] = len(all_chunks)
            all_chunks.append(chunk)

    return all_chunks


def build_vector_store(docs_dir: Path) -> Dict[str, Any]:
    documents = load_markdown_documents(docs_dir)

    if not documents:
        raise FileNotFoundError(
            f"В папке {docs_dir} не найдено markdown-файлов."
        )

    chunks = build_chunks(documents)

    texts_for_embedding = [
        f"{chunk['title']}\n\n{chunk['text']}"
        for chunk in chunks
    ]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts_for_embedding,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return {
        "model": model,
        "chunks": chunks,
        "embeddings": np.array(embeddings),
    }


def keyword_score(query: str, chunk: Dict[str, Any]) -> float:
    """
    Небольшая keyword-добавка к semantic search.
    Помогает вопросам вроде:
    - "Какие есть направления?"
    - "Мне 20"
    - "7 класс"
    """
    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    title_tokens = tokenize(chunk["title"])
    text_tokens = tokenize(chunk["text"])

    title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
    text_overlap = len(query_tokens & text_tokens) / len(query_tokens)

    return min(1.0, 0.6 * title_overlap + 0.4 * text_overlap)


def retrieve_relevant_chunks(
    query: str,
    vector_store: Dict[str, Any],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    model = vector_store["model"]
    chunks = vector_store["chunks"]
    embeddings = vector_store["embeddings"]

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]

    final_scores = []

    for index, chunk in enumerate(chunks):
        semantic = float(semantic_scores[index])
        keyword = keyword_score(query, chunk)

        final = 0.75 * semantic + 0.25 * keyword
        final_scores.append(final)

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        chunk = chunks[index]

        results.append(
            {
                "source": chunk["source"],
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(final_scores[index]),
                "semantic_score": float(semantic_scores[index]),
                "keyword_score": float(keyword_score(query, chunk)),
            }
        )

    return results