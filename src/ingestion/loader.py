from __future__ import annotations
from tqdm import tqdm
from pathlib import Path
from ingestion import Parser, Chunker
from storage import LanceDBStore, ObsidianGraph, KnowledgeTriple
from core import Config

class Loader:
    def __init__(self,
                 parser: Parser,
                 chunker: Chunker,
                 vectorstore: LanceDBStore,
                 graph: ObsidianGraph,
                 graph_path: str | Path = 'gr.gml',
                 batch_size: int = 100) -> None:
        self.parser = parser
        self.chunker = chunker
        self.vectorstore = vectorstore
        self.graph = graph
        self.graph_path = Path(graph_path)
        self.batch_size = batch_size

    @classmethod
    def from_cfg(cls, cfg: Config) -> Loader:
        parser = Parser(cfg.obsidian_path)
        chunker = Chunker()
        vectorstore = LanceDBStore(
            uri=cfg.lance_uri, 
            table_name=cfg.lance_table
        )
        if cfg.graphml_path.is_file():
            graph = ObsidianGraph.from_gml(cfg.graphml_path)
        else:
            graph = ObsidianGraph()
        graph_path = cfg.graphml_path
        batch_size = cfg.loader_batch_size
        return cls(
            parser, chunker, vectorstore, graph, graph_path, batch_size
        )

    def load(self, dir_path='') -> None:
        dir_path = dir_path if dir_path else self.parser.dir_path # fix

        documents = self.parser.load_notes()

        for i in tqdm(
            range(0, len(documents), self.batch_size),
            desc='Loading batches'
        ):
            batch = documents[i:i + self.batch_size]
            chunks = self.chunker.split_documents(batch)
            try:
                self.vectorstore.add_notes(chunks)
            except Exception as e:
                print(f'Batch {i} error!')
                print(e)
                break

            # TODO: Links to sub-chunks
            # Links not from source, but from chunks. Better to another chunks.
            for chunk in chunks:
                source = chunk.metadata['source'] # REDO
                links = chunk.metadata['links']
                for link in links:
                    self.graph.add_triple(
                        KnowledgeTriple(
                            source,
                            'links_to',
                            link
                        )
                    )
        self.graph.write_to_gml(str(self.graph_path))
