'''
Networkx wrapper based on langchain-community implementation.

See https://github.com/langchain-ai/langchain-community/blob/7c10a5fa327f6aaaf7c932822a9e5d144891406e/libs/community/langchain_community/graphs/networkx_graph.py
'''

from __future__ import annotations

from typing import Any, Optional, List
import networkx as nx
from dataclasses import dataclass
from pathlib import Path

@dataclass
class KnowledgeTriple:
    subject: str
    predicate: str
    object_: str

class ObsidianGraph:
    def __init__(self, graph: Optional[Any] = None) -> None:
        if graph is not None:
            if not isinstance(graph, nx.DiGraph):
                raise ValueError('Passed in graph is not of correct shape')
            self._graph = graph
        else:
            self._graph = nx.DiGraph()

    @classmethod
    def from_gml(cls, gml_path: str | Path) -> ObsidianGraph:
        graph = nx.read_gml(gml_path)
        return cls(graph)

    def write_to_gml(self, path: str) -> None:
        nx.write_gml(self._graph, path)

    def add_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        '''Add triple to the graph.'''
        if not self._graph.has_node(knowledge_triple.subject):
            self._graph.add_node(knowledge_triple.subject)
        if not self._graph.has_node(knowledge_triple.object_):
            self._graph.add_node(knowledge_triple.object_)
        self._graph.add_edge(
            knowledge_triple.subject,
            knowledge_triple.object_,
            relation=knowledge_triple.predicate
        )

    def delete_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        '''Delete a triple from the graph.'''
        if self._graph.has_edge(knowledge_triple.subject, knowledge_triple.object_):
            self._graph.remove_edge(knowledge_triple.subject, knowledge_triple.object_)

    def get_entity_knowledge(self, entity: str, depth: int = 1) -> List[str]:
        '''Get information about an entity.'''

        # TODO: Have more information-specific retrieval methods
        if not self._graph.has_node(entity):
            return []

        results = []
        for src, sink in nx.dfs_edges(self._graph, entity, depth_limit=depth):
            relation = self._graph[src][sink]["relation"]
            results.append(f"{src} {relation} {sink}")
        return results

    # def build_from_notes(self, notes):
    #     self._graph.add_triple(
    #         KnowledgeTriple(
    #             "1",
    #             "links_to",
    #             "2"
    #         )
    #     )
