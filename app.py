from pathlib import Path

from app.loader import load_documents
from app.vector_store import FaissVectorStore
from app.retriever import Retriever
from app.context_builder import build_context
from app.llm import LLMClient
from app.prompts import RAG_PROMPT


def answer_question(question: str):

    # 1. Load vector store
    store = FaissVectorStore(
        persist_dir="faiss_index",
        embedding_model_name="all-MiniLM-L6-v2"
    )

    store.load()

    # 2. Create retriever
    retriever = Retriever(store)

    # 3. Retrieve relevant documents
    results = retriever.retrieve(
        question,
        top_k=3
    )

    # 4. Build context
    context = build_context(results)

    # 5. Build RAG prompt
    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    # 6. Generate answer
    llm = LLMClient()

    answer = llm.generate(prompt)

    return answer, results


if __name__ == "__main__":

    question = (
        "Who are the members of the Ganesh committee?"
    )

    answer, results = answer_question(question)

    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ================\n")

    for result in results:

        source = result["metadata"].get(
            "source",
            "Unknown"
        )

        print(f"- {source}")