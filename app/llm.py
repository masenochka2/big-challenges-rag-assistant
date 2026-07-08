import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from prompts import build_rag_prompt


load_dotenv()


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Получает секрет сначала из .env / переменных окружения,
    а если их нет — из Streamlit secrets.

    Локально:
    .env

    На Streamlit Cloud:
    App settings -> Secrets
    """
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def get_hf_client() -> Optional[InferenceClient]:
    """
    Создаёт Hugging Face InferenceClient, если найден HF_TOKEN.
    """
    token = get_secret("HF_TOKEN")

    if not token:
        return None

    return InferenceClient(token=token)


def generate_rag_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    coordinator_settings: Dict[str, str],
) -> str:
    """
    Генерирует финальный ответ ассистента через Hugging Face Inference API.

    Если токена нет или API недоступен, приложение не падает,
    а возвращает понятное сообщение.
    """
    client = get_hf_client()

    if client is None:
        return (
            "LLM API пока не подключён: не найден HF_TOKEN. "
            "Ниже можно посмотреть найденные фрагменты из базы знаний."
        )

    prompt = build_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
        coordinator_settings=coordinator_settings,
    )

    model_name = get_secret(
        "HF_MODEL",
        "Qwen/Qwen2.5-7B-Instruct",
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=700,
        )

        return response.choices[0].message.content.strip()

    except Exception as error:
        return (
            "Не получилось получить ответ от Hugging Face API. "
            "Ниже можно посмотреть найденные фрагменты из базы знаний.\n\n"
            f"Техническая ошибка: {error}"
        )