import os
import faiss
import numpy as np
import pickle
from typing import List, Any

from app.embeddings import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_index",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model_name

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create embedding pipeline only once
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.index = None
        self.embeddings = None
        self.metadata = []

        os.makedirs(self.persist_dir, exist_ok=True)

        print(
            f"Initialized FaissVectorStore with "
            f"persist directory: {self.persist_dir}"
        )

    def build_from_documents(self, documents: List[Any]):

        print(
            f"Building FAISS index from "
            f"{len(documents)} documents..."
        )

        # 1. Split documents into chunks
        chunks = self.embedding_pipeline.chunk_documents(
            documents
        )

        # 2. Convert chunks into embeddings
        embeddings = self.embedding_pipeline.embed_chunks(
            chunks
        )

        # 3. Keep text + metadata for retrieval
        metadata = [
            {
                "text": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

        # 4. Add vectors to FAISS
        self.add_embeddings(
            embeddings,
            metadata
        )

        # 5. Persist index
        self.save()

        print(
            f"FAISS index built and saved with "
            f"{len(chunks)} chunks and embeddings."
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[dict]
    ):

        # FAISS expects float32 vectors
        embeddings = embeddings.astype("float32")

        if self.index is None:

            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatL2(
                dimension
            )

            print(
                f"Created new FAISS index "
                f"with dimension: {dimension}"
            )

        self.index.add(embeddings)

        self.embeddings = (
            embeddings
            if self.embeddings is None
            else np.vstack(
                (self.embeddings, embeddings)
            )
        )

        self.metadata.extend(metadata)

        print(
            f"Added {len(embeddings)} embeddings "
            f"to the FAISS index. "
            f"Total embeddings: {self.index.ntotal}"
        )

    def save(self):

        index_path = os.path.join(
            self.persist_dir,
            "faiss_index.bin"
        )

        metadata_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            metadata_path,
            "wb"
        ) as f:
            pickle.dump(
                self.metadata,
                f
            )

        print(
            f"FAISS index and metadata "
            f"saved to {self.persist_dir}"
        )

    def load(self):

        index_path = os.path.join(
            self.persist_dir,
            "faiss_index.bin"
        )

        metadata_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        if (
            os.path.exists(index_path)
            and os.path.exists(metadata_path)
        ):

            self.index = faiss.read_index(
                index_path
            )

            with open(
                metadata_path,
                "rb"
            ) as f:
                self.metadata = pickle.load(f)

            print(
                f"FAISS index and metadata "
                f"loaded from {self.persist_dir}"
            )

        else:

            print(
                f"No existing FAISS index or "
                f"metadata found in {self.persist_dir}. "
                f"Starting fresh."
            )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[dict]:

        if self.index is None:

            raise ValueError(
                "FAISS index is not built. "
                "Please build the index before searching."
            )

        # Convert query into the SAME embedding space
        query_embedding = self.embedding_pipeline.model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx, dist in zip(
            indices[0],
            distances[0]
        ):

            if idx == -1:
                continue

            if idx < len(self.metadata):

                result = self.metadata[idx].copy()

                result["distance"] = float(dist)

                results.append(result)

        print(
            f"Search completed for query: "
            f"'{query}'. "
            f"Found {len(results)} results."
        )

        return results

    def clear(self):

        self.index = None
        self.embeddings = None
        self.metadata = []

        index_path = os.path.join(
            self.persist_dir,
            "faiss_index.bin"
        )

        metadata_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        if os.path.exists(index_path):
            os.remove(index_path)

        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        print(
            f"Cleared FAISS index and metadata "
            f"from {self.persist_dir}"
        )