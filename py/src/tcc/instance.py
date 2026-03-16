from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set


Edge = Tuple[int, int, float]


@dataclass
class Instance:

    name: str                  # nome da instância, ex: "EUC_Type1_Small/10berlin52"
    n: int                     # número de vértices
    m: int                     # número de arestas
    edges: List[Edge]          # lista de arestas (u, v, w), 0-based
    terminals: List[int]       # lista de vértices requeridos (conjunto R)
    clusters: List[List[int]]  # clusters R_0, ..., R_{h-1}
    cluster_of: List[int]      # tamanho n; -1 para não-requeridos
    is_euclidean: bool = False # flag opcional

    def validate(self) -> None:
        """
        Checa invariantes básicos da instância.

        Se algo estiver errado, lança ValueError com uma mensagem explicando.
        """

        # 1) m bate com len(edges)
        if self.m != len(self.edges):
            raise ValueError(f"m={self.m} mas len(edges)={len(self.edges)}")

        # 2) limites das arestas e pesos
        for (u, v, w) in self.edges:
            if not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError(
                    f"Aresta ({u}, {v}) fora do range [0, {self.n - 1}]"
                )
            if w < 0:
                raise ValueError(f"Peso não-positivo na aresta ({u}, {v}): w={w}")

        # 3) tamanho de cluster_of
        if len(self.cluster_of) != self.n:
            raise ValueError(
                f"cluster_of deve ter tamanho n={self.n}, "
                f"mas tem len={len(self.cluster_of)}"
            )

        # 4) montar união dos clusters e checar disjunção
        union_clusters: Set[int] = set()
        for k, Ck in enumerate(self.clusters):
            if not Ck:
                raise ValueError(f"Cluster {k} está vazio")

            for v in Ck:
                if not (0 <= v < self.n):
                    raise ValueError(f"Vértice {v} inválido no cluster {k}")

                if v in union_clusters:
                    raise ValueError(
                        f"Vértice {v} aparece em mais de um cluster "
                        f"(violação de disjunção)"
                    )
                union_clusters.add(v)

                # coerência com cluster_of
                if self.cluster_of[v] != k:
                    raise ValueError(
                        f"cluster_of[{v}]={self.cluster_of[v]}, "
                        f"mas esperava {k} (cluster {k})"
                    )

        # 5) terminals bate com união dos clusters
        if set(self.terminals) != union_clusters:
            raise ValueError(
                "terminals difere da união dos clusters: "
                f"terminals={sorted(self.terminals)}, "
                f"union={sorted(union_clusters)}"
            )

        # 6) não-requeridos têm cluster_of = -1
        for v in range(self.n):
            if v not in union_clusters and self.cluster_of[v] != -1:
                raise ValueError(
                    f"Vértice {v} não é requerido, mas cluster_of[{v}]="
                    f"{self.cluster_of[v]} (esperado -1)"
                )
