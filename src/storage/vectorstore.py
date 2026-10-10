from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_lancedb import LanceDB
from pathlib import Path

class LanceDBStore:
    def __init__(self, embeddings: str | Embeddings,
                 uri: str | Path = 'data/lancedb',
                 table_name='obsidian') -> None:
        if isinstance(embeddings, str):
            embeddings = HuggingFaceEmbeddings(
                model_name=embeddings
            )

        self.db = LanceDB(
            embedding_function=embeddings,
            uri=uri,
            table_name=table_name
        )

    def add_notes(self, chunks: list[Document]) -> None:
        self.db.add_documents(chunks)

    def similarity_search(self, query, k, **kwargs):
        return self.db.similarity_search(query, k, **kwargs)

# Maybe create a base class and inherit different vectorstores?
# Just for studying purpose, I think lancedb suffices.
