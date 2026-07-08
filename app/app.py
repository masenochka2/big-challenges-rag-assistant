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
SHOW_DEBUG_INFO = False


st.set_page_config(
    page_title="Помощник БВ",
    layout="wide",
)


@st.cache_resource
def load_vector_store():
    return build_vector_store(DOCS_DIR)


def build_retrieval_query(
    current_question: str,
    previous_question: Optional[str],
) -> str:
    """
    Если вопрос короткий или похож на уточнение, добавляем предыдущий вопрос.
    Это помогает для фраз вроде: "А если мне 20?"
    """
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
    previous_user_questions = [
        message["content"]
        for message in st.session_state.messages
        if message["role"] == "user"
    ]

    if not previous_user_questions:
        return None

    return previous_user_questions[-1]


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

    with st.spinner("Ищу информацию в базе знаний..."):
        vector_store = load_vector_store()

        results = retrieve_relevant_chunks(
            query=retrieval_query,
            vector_store=vector_store,
            top_k=TOP_K,
        )

    if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
        assistant_message = (
            "Я не нашёл достаточно точной информации в базе знаний, чтобы уверенно ответить. "
            "Лучше проверить официальный сайт конкурса или уточнить вопрос у координатора."
        )
    else:
        with st.spinner("Формирую ответ..."):
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
            "retrieval_query": retrieval_query,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("Помощник по Большим вызовам")
st.caption(
    "RAG-ассистент по конкурсу «Большие вызовы» в Новосибирской области"
)

tab_chat, tab_about = st.tabs(["Чат", "О проекте"])


with tab_chat:
    st.markdown(
        """
        Задайте вопрос о конкурсе **«Большие вызовы»** в Новосибирской области.  
        Ассистент ищет информацию в базе знаний, отвечает простым языком и показывает источники.
        """
    )

    st.info(
        "Ассистент не заменяет официальный сайт конкурса и координатора. "
        "Сроки, регистрацию и актуальные требования лучше проверять на официальных ресурсах."
    )

    st.markdown("### Быстрые вопросы")

    cols = st.columns(3)
    selected_question = None

    for index, example_question in enumerate(EXAMPLE_QUESTIONS):
        with cols[index % 3]:
            if st.button(example_question, use_container_width=True):
                selected_question = example_question

    st.divider()

    if st.session_state.messages:
        if st.button("Очистить историю"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "results" in message:
                st.markdown("**Источники:**")

                for i, result in enumerate(message["results"], start=1):
                    st.markdown(
                        f"{i}. `{result['source']}` — "
                        f"**{result['title']}**"
                    )

                with st.expander("Показать найденные фрагменты"):
                    for i, result in enumerate(message["results"], start=1):
                        st.markdown(f"### {i}. {result['title']}")
                        st.caption(
                            f"Источник: {result['source']} | "
                            f"score={result['score']:.3f}"
                        )
                        st.write(result["text"])

                        if SHOW_DEBUG_INFO:
                            st.caption(
                                f"semantic_score={result['semantic_score']:.3f}, "
                                f"keyword_score={result['keyword_score']:.3f}"
                            )

                        st.divider()

                if SHOW_DEBUG_INFO and "retrieval_query" in message:
                    with st.expander("Технический запрос для поиска"):
                        st.code(message["retrieval_query"])

    typed_question = st.chat_input(
        "Задайте вопрос про «Большие вызовы»..."
    )

    question = typed_question or selected_question

    if question:
        process_question(question)
        st.rerun()


with tab_about:
    st.markdown(
        """
        ### О проекте

        **БВ Помощник** — это RAG-ассистент для ответов на вопросы о конкурсе
        **«Большие вызовы»** в Новосибирской области.

        Проект предназначен для демонстрации того, как можно использовать RAG-подход
        для справочного ассистента по образовательной программе.

        ### Что умеет ассистент

        - отвечает на вопросы по базе знаний;
        - показывает источники ответа;
        - различает основной конкурс и «ПРО Большие вызовы»;
        - аккуратно отвечает на вопросы про возраст, классы, сроки и регистрацию;
        - честно сообщает, если информации недостаточно;
        - не пишет проект за участника;
        - не обещает победу или прохождение отбора.

    
        ### Ограничения

        Ассистент не является официальным источником информации.  
        Актуальные сроки, форму регистрации, контакты и требования нужно проверять
        на официальных ресурсах конкурса или уточнять у координатора.
        """
    )