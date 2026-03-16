from __future__ import annotations

import random
from typing import Dict, List, Tuple

from tcc.instance import Instance
from tcc.solution import Solution

Edge = Tuple[int, int]


def _weight_lookup(inst: Instance) -> Dict[Tuple[int, int], float]:
    w: Dict[Tuple[int, int], float] = {}
    for u, v, c in inst.edges:
        w[(u, v)] = float(c)
        w[(v, u)] = float(c)
    return w


def _mst_prim(nodes: List[int], w: Dict[Tuple[int, int], float]) -> List[Edge]:
    if len(nodes) <= 1:
        return []

    in_tree = {nodes[0]}
    remaining = set(nodes[1:])
    edges: List[Edge] = []

    while remaining:
        best = None
        best_u = best_v = None
        for u in in_tree:
            for v in remaining:
                cost = w[(u, v)]
                if best is None or cost < best:
                    best = cost
                    best_u, best_v = u, v
        edges.append((best_u, best_v))
        in_tree.add(best_v)
        remaining.remove(best_v)

    return edges


def solve_two_level_mst(inst: Instance) -> Tuple[float, List[Edge]]:
    """

    1) MST dentro de cada cluster (somente terminais)
    2) MST entre clusters usando custo(i,j)=min_{u in Ci, v in Cj} w(u,v)

    OBS: não usa Steiner.
    """
    w = _weight_lookup(inst)

    all_edges: List[Edge] = []
    for ck in inst.clusters:
        all_edges.extend(_mst_prim(ck, w))

    h = len(inst.clusters)
    if h <= 1:
        cost = sum(w[(u, v)] for (u, v) in all_edges)
        return cost, all_edges

    best_pair: Dict[Tuple[int, int], Tuple[float, int, int]] = {}
    for i in range(h):
        for j in range(i + 1, h):
            best = None
            bu = bv = None
            for u in inst.clusters[i]:
                for v in inst.clusters[j]:
                    c = w[(u, v)]
                    if best is None or c < best:
                        best = c
                        bu, bv = u, v
            best_pair[(i, j)] = (best, bu, bv)

    in_tree = {0}
    remaining = set(range(1, h))
    while remaining:
        best = None
        best_j = None
        best_u = best_v = None
        for i in in_tree:
            for j in remaining:
                a, b = (i, j) if i < j else (j, i)
                c, u, v = best_pair[(a, b)]
                if best is None or c < best:
                    best = c
                    best_j = j
                    best_u, best_v = u, v

        all_edges.append((best_u, best_v))
        in_tree.add(best_j)
        remaining.remove(best_j)

    cost = sum(w[(u, v)] for (u, v) in all_edges)
    return cost, all_edges


def build_initial_solution(inst: Instance, rng: random.Random | None = None) -> Solution:

    from .operators_repair_two_level import repair_r8_full_decode_sph
    from .partial_state import PartialState
    from tcc.solution import Solution as Sol

    if inst.is_euclidean:
        cost, edges = solve_two_level_mst(inst)
        return Solution(instance_name=inst.name, cost=cost, edges=edges)

    if rng is None:
        rng = random.Random(0)

    dummy = PartialState(
        base_solution=Sol(instance_name=inst.name, cost=0.0, edges=[]),
        local_edges=[],
        global_edges_remaining=[],
        global_edges_removed=[],
        components=[[i for i in range(len(inst.clusters))]],
        cluster_to_component=[0 for _ in range(len(inst.clusters))],
        destroyed_cluster=None,
        meta={"destroy_op": "none"},
    )
    return repair_r8_full_decode_sph(inst, dummy, rng)


def build_reset_solution(inst: Instance, rng: random.Random) -> Solution:
    return build_initial_solution(inst, rng=rng)