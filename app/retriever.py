class Retriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3):
        return self.vector_store.search(
            query,
            top_k
        )