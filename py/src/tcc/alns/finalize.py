from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from tcc.instance import Instance
from tcc.solution import Solution, TreeEdge


def _norm_edge(u: int, v: int) -> TreeEdge:
    return (u, v) if u < v else (v, u)


def build_weight_lookup(inst: Instance) -> Dict[Tuple[int, int], float]:
    w: Dict[Tuple[int, int], float] = {}
    for u, v, c in inst.edges:
        cc = float(c)
        w[(u, v)] = cc
        w[(v, u)] = cc
    return w


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def finalize_with_kruskal(
    inst: Instance,
    local_edges: Iterable[TreeEdge],
    extra_edges: Iterable[TreeEdge],
) -> Solution:
    """
    Garante que a solução final é uma ÁRVORE e preserva as local trees.
    """
    w = build_weight_lookup(inst)
    dsu = DSU(inst.n)

    local_norm = [_norm_edge(u, v) for (u, v) in local_edges]
    extra_norm = [_norm_edge(u, v) for (u, v) in extra_edges]

    chosen: List[TreeEdge] = []
    chosen_set: Set[TreeEdge] = set()

    # 1) admite local edges primeiro
    for (u, v) in local_norm:
        e = _norm_edge(u, v)
        if e in chosen_set:
            continue
        chosen_set.add(e)
        chosen.append(e)
        dsu.union(u, v)

    reps = [inst.clusters[k][0] for k in range(len(inst.clusters))]

    def clusters_connected() -> bool:
        r0 = dsu.find(reps[0])
        for r in reps[1:]:
            if dsu.find(r) != r0:
                return False
        return True

    if clusters_connected():
        cost = sum(w[(u, v)] for (u, v) in chosen)
        return Solution(instance_name=inst.name, cost=cost, edges=chosen)

    # 2) kruskal nas extras
    extra_sorted = sorted(
        (e for e in extra_norm if e not in chosen_set),
        key=lambda e: w[(e[0], e[1])],
    )

    for (u, v) in extra_sorted:
        if dsu.union(u, v):
            chosen.append((u, v))
            chosen_set.add((u, v))
            if clusters_connected():
                break

    cost = sum(w[(u, v)] for (u, v) in chosen)
    return Solution(instance_name=inst.name, cost=cost, edges=chosen)
