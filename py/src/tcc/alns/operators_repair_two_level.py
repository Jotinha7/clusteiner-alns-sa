from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

from tcc.instance import Instance
from tcc.solution import Solution, TreeEdge

from .partial_state import PartialState
from .finalize import finalize_with_kruskal
from .operators_repair import build_adj, dijkstra_all, reconstruct_path_edges


def _norm_edge(u: int, v: int) -> TreeEdge:
    return (u, v) if u < v else (v, u)


def _cluster_local_vertices_from_edges(inst: Instance, local_edges: List[TreeEdge]) -> List[Set[int]]:
    """Vértices de cada local tree olhando apenas `local_edges` (que são disjuntas)."""
    adj: Dict[int, List[int]] = {}
    for (u, v) in local_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    per: List[Set[int]] = []
    for terminals in inst.clusters:
        start = terminals[0]
        seen: Set[int] = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        per.append(seen)
    return per


def _sph_tree_on_allowed(
    inst: Instance,
    required: List[int],
    allowed: Set[int],
    rng: random.Random,
    adj: List[List[Tuple[int, float]]],
) -> List[TreeEdge]:
    if len(required) <= 1:
        return []

    remaining = set(required)
    root = rng.choice(required)
    remaining.remove(root)

    tree_vertices: Set[int] = {root}
    edges: List[TreeEdge] = []

    blocked = set(range(inst.n)) - allowed

    while remaining:
        dist, parent = dijkstra_all(inst, adj, list(tree_vertices), blocked=blocked)

        best_t = None
        best_d = 10**30
        for t in remaining:
            if dist[t] < best_d:
                best_d = dist[t]
                best_t = t
        if best_t is None or best_d >= 10**29:
            break

        path = reconstruct_path_edges(parent, best_t)
        for (u, v) in path:
            edges.append(_norm_edge(u, v))
            tree_vertices.add(u)
            tree_vertices.add(v)

        remaining.remove(best_t)

    return list({e for e in edges})


def repair_r7_global_sph(inst: Instance, ps: PartialState, rng: random.Random) -> Solution:
    local_edges = [_norm_edge(u, v) for (u, v) in ps.local_edges]
    per_local_v = _cluster_local_vertices_from_edges(inst, local_edges)

    # pool de steiners globais = steiners que não estão dentro de nenhuma local tree
    used_local: Set[int] = set().union(*per_local_v)
    free_global = {v for v in range(inst.n) if inst.cluster_of[v] == -1 and v not in used_local}

    adj = build_adj(inst)

    # começa conectando a partir de um cluster aleatório
    h = len(inst.clusters)
    start = rng.randrange(h)
    connected = {start}
    tree_vertices: Set[int] = set(per_local_v[start])
    global_edges: List[TreeEdge] = []

    all_not_connected_vertices: Set[int] = set()
    for k in range(h):
        if k != start:
            all_not_connected_vertices |= per_local_v[k]

    while len(connected) < h:
        # não deixe vértices de clusters ainda não conectados virarem intermediários
        ban_expand = all_not_connected_vertices - tree_vertices

        allowed = set(tree_vertices) | set(free_global) | set(all_not_connected_vertices)
        blocked = set(range(inst.n)) - allowed
        dist, parent = dijkstra_all(inst, adj, list(tree_vertices), blocked=blocked, ban_expand=ban_expand)

        # escolhe o cluster não-conectado cuja local tree tem um vértice mais próximo
        best_k = None
        best_v = None
        best_d = 10**30
        for k in range(h):
            if k in connected:
                continue
            for v in per_local_v[k]:
                if dist[v] < best_d:
                    best_d = dist[v]
                    best_k = k
                    best_v = v

        if best_k is None or best_v is None or best_d >= 10**29:
            break

        path = reconstruct_path_edges(parent, best_v)
        for e in path:
            global_edges.append(_norm_edge(*e))
            tree_vertices.add(e[0])
            tree_vertices.add(e[1])
            if inst.cluster_of[e[0]] == -1:
                free_global.discard(e[0])
            if inst.cluster_of[e[1]] == -1:
                free_global.discard(e[1])

        # agora esse cluster virou "conectado"
        connected.add(best_k)
        tree_vertices |= per_local_v[best_k]
        # remove os vértices dele do conjunto "not connected"
        all_not_connected_vertices -= per_local_v[best_k]

    return finalize_with_kruskal(inst, local_edges, global_edges)


def repair_r8_full_decode_sph(inst: Instance, ps: PartialState, rng: random.Random) -> Solution:
    adj = build_adj(inst)

    h = len(inst.clusters)
    order = list(range(h))
    rng.shuffle(order)

    free_steiners: Set[int] = {v for v in range(inst.n) if inst.cluster_of[v] == -1}
    local_edges: List[TreeEdge] = []
    used_local_vertices: List[Set[int]] = [set() for _ in range(h)]

    for k in order:
        terms = list(inst.clusters[k])
        allowed = set(terms) | set(free_steiners)
        edges_k = _sph_tree_on_allowed(inst, terms, allowed, rng, adj)
        local_edges.extend(edges_k)

        # marca vertices usados e remove steiners do pool
        used: Set[int] = set(terms)
        for (u, v) in edges_k:
            used.add(u)
            used.add(v)
        used_local_vertices[k] = used
        for v in used:
            if inst.cluster_of[v] == -1:
                free_steiners.discard(v)

    # Agora, global SPH conectando as local trees
    # Monta per_local_v a partir do local_edges construído
    per_local_v = _cluster_local_vertices_from_edges(inst, local_edges)
    used_local_all = set().union(*per_local_v)
    free_global = {v for v in range(inst.n) if inst.cluster_of[v] == -1 and v not in used_local_all}

    # global SPH
    start = rng.randrange(h)
    connected = {start}
    tree_vertices: Set[int] = set(per_local_v[start])
    global_edges: List[TreeEdge] = []

    all_not_connected_vertices: Set[int] = set()
    for k in range(h):
        if k != start:
            all_not_connected_vertices |= per_local_v[k]

    while len(connected) < h:
        ban_expand = all_not_connected_vertices - tree_vertices
        allowed = set(tree_vertices) | set(free_global) | set(all_not_connected_vertices)
        blocked = set(range(inst.n)) - allowed
        dist, parent = dijkstra_all(inst, adj, list(tree_vertices), blocked=blocked, ban_expand=ban_expand)

        best_k = None
        best_v = None
        best_d = 10**30
        for k in range(h):
            if k in connected:
                continue
            for v in per_local_v[k]:
                if dist[v] < best_d:
                    best_d = dist[v]
                    best_k = k
                    best_v = v

        if best_k is None or best_v is None or best_d >= 10**29:
            break

        path = reconstruct_path_edges(parent, best_v)
        for e in path:
            global_edges.append(_norm_edge(*e))
            tree_vertices.add(e[0])
            tree_vertices.add(e[1])
            if inst.cluster_of[e[0]] == -1:
                free_global.discard(e[0])
            if inst.cluster_of[e[1]] == -1:
                free_global.discard(e[1])

        connected.add(best_k)
        tree_vertices |= per_local_v[best_k]
        all_not_connected_vertices -= per_local_v[best_k]

    return finalize_with_kruskal(inst, local_edges, global_edges)
