from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.model = SentenceTransformer(model_name)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        print(
            f"Initialized EmbeddingPipeline with model: {model_name}"
        )

    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:

        print(f"Chunking {len(documents)} documents...")

        chunks = self.text_splitter.split_documents(documents)

        print(
            f"Created {len(chunks)} chunks from documents."
        )

        return chunks

    def embed_chunks(
        self,
        chunks: List[Document]
    ) -> np.ndarray:

        print(f"Embedding {len(chunks)} chunks...")

        # Document -> text
        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print(
            f"Generated embeddings with shape: {embeddings.shape}"
        )

        return embeddings