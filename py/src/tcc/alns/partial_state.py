from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tcc.solution import Solution, TreeEdge


@dataclass
class PartialState:
    """
    Estado parcial após uma destruição (destroy).

    """

    # solução original (antes de destruir)
    base_solution: Solution

    local_edges: List[TreeEdge]

    # arestas globais após remoções
    global_edges_remaining: List[TreeEdge]

    # arestas globais removidas pelo destroy
    global_edges_removed: List[TreeEdge]

    # componentes no nível dos clusters: cada componente é uma lista de ids de cluster
    components: List[List[int]]

    # para cada cluster k, qual componente ele pertence (id = índice em components)
    cluster_to_component: List[int]

    # se o destroy escolheu um cluster específico, guarda aqui
    destroyed_cluster: Optional[int] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_components(self) -> int:
        return len(self.components)

    def current_edges(self) -> List[TreeEdge]:
        return list(self.local_edges) + list(self.global_edges_remaining)
