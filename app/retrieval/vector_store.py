import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add(self, embeddings: list[list[float]], chunks: list[str]):
        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match the number of chunks."
            )

        if not embeddings:
            return

        vectors = np.array(embeddings, dtype="float32")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for distance, index in zip(distances[0], indices[0]):
            if index == -1:
                continue

            results.append({
                "chunk": self.chunks[index],
                "distance": float(distance)
            })

        return results
    