import lancedb
import torch
import numpy as np
import pyarrow as pa
import json
import hashlib
from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents import Document
from pathlib import Path

class LanceDBStore:
    def __init__(
        self, embeddings: str = 'BAAI/bge-m3',
        uri: str | Path = 'data/lancedb',
        table_name='obsidian',
        *,
        device: str = 'cpu',
        use_fp16: bool = False,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.uri = str(uri)
        self.table_name = table_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = BGEM3FlagModel(
            embeddings,
            use_fp16=use_fp16 and torch.cuda.is_available(),
            device=device,
        )

    def _connect(self):
        return lancedb.connect(self.uri)

    def _open_table(self):
        return self._connect().open_table(self.table_name)

    def _row(self, doc: Document, dense, sparse, doc_id: str) -> dict:
        return {
            'id': doc_id,
            'dense': np.asarray(dense, dtype=np.float32).tolist(),
            'sparse': json.dumps({str(k): float(v) for k, v in sparse.items()}),
            'text': doc.page_content,
            'metadata': json.dumps(doc.metadata, ensure_ascii=False),
        }

    @staticmethod
    def _schema(dim: int) -> pa.Schema:
        return pa.schema([
            pa.field('id', pa.string()),
            pa.field('dense', pa.list_(pa.float32(), dim)),
            pa.field('sparse', pa.string()),
            pa.field('text', pa.string()),
            pa.field('metadata', pa.string()),
        ])

    def _encode(self, texts: list[str]) -> dict:
        return self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False, # Might use later, need eval
            batch_size = self.batch_size,
            max_length=self.max_length
        )

    def add_notes(self, chunks: list[Document]) -> None:
        if not chunks:
            return

        out = self._encode([c.page_content for c in chunks])
        rows = [
            self._row(
                doc,
                dense,
                sparse,
                hashlib.sha256(doc.page_content.encode()).hexdigest()
            )
            for doc, dense, sparse in zip(
                chunks, out['dense_vecs'], out['lexical_weights']
            )
        ]
        db = self._connect()
        if self.table_name in db.list_tables().tables:
            db.open_table(self.table_name).add(rows)
        else:
            table = pa.Table.from_pylist(
                rows, schema=self._schema(len(rows[0]['dense']))
            )
            db.create_table(self.table_name, data=table)

    def _rows_to_documents(self, rows: list[dict]) -> list[Document]:
        return [
            Document(
                page_content=row['text'],
                metadata=json.loads(row['metadata'])
            )
            for row in rows
        ]

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        q = self._encode([query])['dense_vecs'][0]
        rows = self._open_table().search(q).limit(k).to_list()
        return self._rows_to_documents(rows)

    def sparse_search(self, query: str, k: int = 5) -> list[Document]:
        q = self._encode([query])['lexical_weights'][0]
        rows = self._open_table().to_arrow().to_pylist()
        scored = [
            (
                self.model.compute_lexical_matching_score(
                    q, json.loads(row['sparse'])
                ),
                row,
            )
            for row in rows
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return self._rows_to_documents([row for score, row in scored[:k] if score > 0])

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        c: int = 60,
        candidates: int = 50,
    ) -> list[Document]:
        '''
        Run hybrid search using BGE-m3 and fuse the results with RRF.

        The dense and sparse ranked lists are combined with weighted reciprocal
        rank fusion, and documents appearing in both lists are returned once.

        Args:
            query: Query to search for.
            k: Number of fused documents to return.
            dense_weight: RRF weight for the dense result list.
            sparse_weight: RRF weight for the sparse result list.
            c: RRF smoothing constant. Larger values flatten the advantage of
                top ranks.
            candidates: Number of candidates retrieved from each leg before
                fusion. Should be at least ``k``.

        Returns:
            Up to ``k`` documents ordered by descending fused RRF score.

        Raises:
            ValueError: If ``candidates`` is smaller than ``k``.
        '''
        if candidates < k:
            raise ValueError('``candidates`` should be at least ``k``')

        q = self._encode([query])
        q_dense = q['dense_vecs'][0]
        q_sparse = q['lexical_weights'][0]

        table = self._open_table()
        dense_rows = table.search(q_dense).limit(candidates).to_list()

        scored = [
            (
                self.model.compute_lexical_matching_score(
                    q_sparse, json.loads(row['sparse'])
                ),
                row,
            )
            for row in table.to_arrow().to_pylist()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        sparse_rows = [row for score, row in scored[:candidates] if score > 0]

        # by_id ensures a doc appearing in both lists is returned once
        rrf = {}
        by_id = {}
        for weight, rows in ((dense_weight, dense_rows), (sparse_weight, sparse_rows)):
            for rank, row in enumerate(rows, start=1):
                row_id = row['id']
                by_id[row_id] = row
                rrf[row_id] = rrf.get(row_id, 0.0) + weight / (c + rank)

        top = sorted(rrf, key=rrf.__getitem__, reverse=True)[:k]
        return self._rows_to_documents([by_id[doc_id] for doc_id in top])
