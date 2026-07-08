from pathlib import Path
from typing import Optional

import streamlit as st

from rag import build_vector_store, retrieve_relevant_chunks
from llm import generate_rag_answer
from config import ASSISTANT_CONFIG, EXAMPLE_QUESTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"

MIN_RETRIEVAL_SCORE = 0.18
TOP_K = 3


st.set_page_config(
    page_title="БВ Помощник",
    layout="centered",
)


@st.cache_resource
def load_vector_store():
    return build_vector_store(DOCS_DIR)


def build_retrieval_query(
    current_question: str,
    previous_question: Optional[str],
) -> str:

    # Для коротких уточнений используем предыдущий вопрос как контекст.
    if not previous_question:
        return current_question

    question_lower = current_question.lower().strip()

    follow_up_starts = (
        "а если",
        "а можно",
        "а мне",
        "а для",
        "а когда",
        "а где",
        "а какие",
        "а что",
        "это",
        "тогда",
    )

    is_short_question = len(current_question.split()) <= 5
    is_follow_up = question_lower.startswith(follow_up_starts)

    if is_short_question or is_follow_up:
        return f"{previous_question}\n{current_question}"

    return current_question


def get_previous_user_question() -> Optional[str]:
    previous_questions = [
        message["content"]
        for message in st.session_state.messages
        if message["role"] == "user"
    ]

    if not previous_questions:
        return None

    return previous_questions[-1]


def clean_source_text(text: str) -> str:
    lines = text.strip().splitlines()

    if lines and lines[0].startswith("## "):
        lines = lines[1:]

    cleaned_text = "\n".join(lines).strip()

    return cleaned_text


def render_sources(results) -> None:
    if not results:
        return

    with st.expander("Источники"):
        st.caption(
            "Фрагменты базы знаний, на которые опирался ответ."
        )

        for index, result in enumerate(results, start=1):
            title = result.get("title", "Раздел базы знаний")
            source_text = clean_source_text(result.get("text", ""))

            st.markdown(f"**{index}. {title}**")

            if source_text:
                st.markdown(source_text)
            else:
                st.markdown("Текст источника не найден.")

            if index != len(results):
                st.divider()


def process_question(question: str) -> None:
    previous_question = get_previous_user_question()

    retrieval_query = build_retrieval_query(
        current_question=question,
        previous_question=previous_question,
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.spinner("Ищу информацию..."):
        vector_store = load_vector_store()

        results = retrieve_relevant_chunks(
            query=retrieval_query,
            vector_store=vector_store,
            top_k=TOP_K,
        )

    if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
        assistant_message = (
            "Я не нашёл достаточно точной информации в базе знаний, "
            "чтобы уверенно ответить. Лучше проверить официальный сайт конкурса "
            "или уточнить вопрос у координатора."
        )
    else:
        with st.spinner("Готовлю ответ..."):
            assistant_message = generate_rag_answer(
                question=question,
                retrieved_chunks=results,
                coordinator_settings=ASSISTANT_CONFIG,
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_message,
            "results": results,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("БВ Помощник")

st.caption(
    "Помощник по конкурсу «Большие вызовы» в Новосибирской области"
)

tab_chat, tab_about = st.tabs(["Чат", "О проекте"])


with tab_chat:
    st.markdown(
        """
        Задайте вопрос о конкурсе. Помощник найдёт информацию в базе знаний,
        ответит простым языком и покажет источники.
        """
    )

    st.info(
        "Помощник не заменяет официальный сайт конкурса. "
        "Актуальные сроки, регистрацию и требования лучше проверять на официальных ресурсах."
    )

    st.markdown("### Быстрые вопросы")

    columns = st.columns(2)
    selected_question = None

    for index, example_question in enumerate(EXAMPLE_QUESTIONS):
        with columns[index % 2]:
            if st.button(example_question, use_container_width=True):
                selected_question = example_question

    if st.session_state.messages:
        st.divider()

        if st.button("Очистить чат"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("results", []))

    typed_question = st.chat_input(
        "Напишите вопрос о конкурсе..."
    )

    question = typed_question or selected_question

    if question:
        process_question(question)
        st.rerun()


with tab_about:
    st.markdown(
        """
        ### О проекте

        **БВ Помощник** — это справочный ассистент по конкурсу
        **«Большие вызовы»** в Новосибирской области.

        Он отвечает на вопросы по подготовленной базе знаний и показывает,
        на какие фрагменты опирался при ответе.

        ### Что умеет помощник

        Помощник может рассказать, кто может участвовать в конкурсе,
        какие есть направления, что нужно указать в заявке, чем основной конкурс
        отличается от «ПРО Больших вызовов», где проходит региональный этап
        и в каких случаях лучше уточнить информацию на официальных ресурсах.

        ### Ограничения

        Помощник не является официальным источником информации.
        Актуальные сроки, регистрацию, контакты и официальные требования
        нужно проверять на сайте конкурса или у координатора.
        """
    )
