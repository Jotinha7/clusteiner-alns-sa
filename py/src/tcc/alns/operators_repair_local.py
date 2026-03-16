from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, List, Set, Tuple

from tcc.instance import Instance
from tcc.solution import TreeEdge

from .partial_state import PartialState
from .operators_destroy import compute_local_vertices_per_cluster

import heapq


def _norm_edge(u: int, v: int) -> TreeEdge:
    return (u, v) if u < v else (v, u)


def _weight_map(instance: Instance) -> Dict[Tuple[int, int], float]:
    wm = getattr(instance, "_wm_cache", None)
    if wm is not None:
        return wm

    wm2: Dict[Tuple[int, int], float] = {}
    for (u, v, w) in instance.edges:
        a, b = (u, v) if u < v else (v, u)
        wm2[(a, b)] = float(w)
    setattr(instance, "_wm_cache", wm2)
    return wm2


def _build_adj(instance: Instance) -> List[List[Tuple[int, float]]]:
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(instance.n)]
    for (u, v, w) in instance.edges:
        ww = float(w)
        adj[u].append((v, ww))
        adj[v].append((u, ww))
    return adj


def _dijkstra_restricted(
    instance: Instance,
    adj: List[List[Tuple[int, float]]],
    sources: List[int],
    *,
    allowed: Set[int],
) -> Tuple[List[float], List[int]]:
    INF = 10**30
    dist = [INF] * instance.n
    parent = [-1] * instance.n
    pq: List[Tuple[float, int]] = []

    for s in sources:
        if s not in allowed:
            continue
        dist[s] = 0.0
        parent[s] = -1
        heapq.heappush(pq, (0.0, s))

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w_uv in adj[u]:
            if v not in allowed:
                continue
            nd = d + w_uv
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, parent


def _reconstruct_path(parent: List[int], target: int) -> List[TreeEdge]:
    edges: List[TreeEdge] = []
    cur = target
    while parent[cur] != -1:
        p = parent[cur]
        edges.append(_norm_edge(p, cur))
        cur = p
    edges.reverse()
    return edges


def _sph_local_tree(
    instance: Instance,
    terminals: List[int],
    *,
    allowed_vertices: Set[int],
    rng: random.Random,
    adj: List[List[Tuple[int, float]]],
) -> List[TreeEdge]:
    """
    SPH (Shortest Path Heuristic) para construir uma Steiner tree do cluster.
    """
    if len(terminals) <= 1:
        return []

    remaining = set(terminals)
    root = rng.choice(terminals)
    remaining.remove(root)

    tree_vertices: Set[int] = {root}
    tree_edges: List[TreeEdge] = []

    while remaining:
        dist, parent = _dijkstra_restricted(instance, adj, list(tree_vertices), allowed=allowed_vertices)

        # pega terminal restante mais próximo da árvore
        best_t = None
        best_d = 10**30
        for t in remaining:
            if dist[t] < best_d:
                best_d = dist[t]
                best_t = t

        if best_t is None or best_d >= 10**29:
            # não conseguiu conectar
            break

        path_edges = _reconstruct_path(parent, best_t)
        for (u, v) in path_edges:
            tree_edges.append((u, v))
            tree_vertices.add(u)
            tree_vertices.add(v)

        remaining.remove(best_t)

    # poda Steiner folhas (mantém terminais sempre)
    steiner_set = {v for v in tree_vertices if instance.cluster_of[v] == -1}
    pruned = _prune_steiner_leaves([_norm_edge(u, v) for (u, v) in tree_edges], steiner_set)
    return list({e for e in pruned})


def _prim_mst_complete(vertices: List[int], wm: Dict[Tuple[int, int], float]) -> List[TreeEdge]:
    n = len(vertices)
    if n <= 1:
        return []

    INF = 1e30
    in_mst = [False] * n
    key = [INF] * n
    parent = [-1] * n
    key[0] = 0.0

    for _ in range(n):
        u = -1
        best = INF
        for i in range(n):
            if not in_mst[i] and key[i] < best:
                best = key[i]
                u = i
        if u == -1:
            break
        in_mst[u] = True

        vu = vertices[u]
        for v in range(n):
            if in_mst[v] or v == u:
                continue
            vv = vertices[v]
            a, b = (vu, vv) if vu < vv else (vv, vu)
            w = wm[(a, b)]
            if w < key[v]:
                key[v] = w
                parent[v] = u

    edges: List[TreeEdge] = []
    for v in range(1, n):
        if parent[v] == -1:
            continue
        edges.append(_norm_edge(vertices[parent[v]], vertices[v]))
    return edges


def _prune_steiner_leaves(edges: List[TreeEdge], steiner_set: Set[int]) -> List[TreeEdge]:
    """Remove Steiner folhas da árvore local."""
    if not edges:
        return edges

    adj: Dict[int, List[int]] = {}
    deg: Dict[int, int] = {}
    for (u, v) in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    q = [s for s in steiner_set if deg.get(s, 0) == 1]
    edge_set = set(edges)

    while q:
        s = q.pop()
        if deg.get(s, 0) != 1:
            continue
        if s not in adj or not adj[s]:
            continue

        nei = adj[s][0]
        e = _norm_edge(s, nei)
        if e in edge_set:
            edge_set.remove(e)

        deg[s] = 0
        if nei in adj:
            try:
                adj[nei].remove(s)
            except ValueError:
                pass
            deg[nei] = max(0, deg.get(nei, 0) - 1)
            if nei in steiner_set and deg.get(nei, 0) == 1:
                q.append(nei)

        adj[s] = []

    return list(edge_set)


def ensure_local_rebuilt(
    instance: Instance,
    ps: PartialState,
    rng: random.Random,
    max_steiners: int = 15,
    diversify_pool_mult: int = 3,
) -> PartialState:
    """
    Se o destroy foi D3, reconstrói a *local tree* do cluster destruído.
    """
    if ps.meta.get("destroy_op") != "D3_break_local_tree":
        return ps

    c = ps.destroyed_cluster
    if c is None:
        return ps

    terms = list(instance.clusters[c])
    if len(terms) <= 1:
        return ps

    # proibidos = vértices que pertencem às local trees dos outros clusters (na solução base)
    per_v = compute_local_vertices_per_cluster(instance, ps.base_solution.edges)
    forbidden: Set[int] = set()
    for k, vk in enumerate(per_v):
        if k == c:
            continue
        forbidden |= vk

    # evita steiners já usados no GLOBAL atual (mantém “global liga em terminais”)
    used_global: Set[int] = set()
    for (u, v) in ps.global_edges_remaining:
        used_global.add(u)
        used_global.add(v)

    # Conjunto permitido para este cluster:
    #   terminais do cluster + steiners que não são proibidos e não aparecem no global.
    allowed: Set[int] = set(terms)
    for v in range(instance.n):
        if instance.cluster_of[v] == -1 and v not in forbidden and v not in used_global:
            allowed.add(v)

    if instance.is_euclidean:
        wm = _weight_map(instance)
        steiners = [v for v in allowed if instance.cluster_of[v] == -1]
        if not steiners:
            return ps

        scored: List[Tuple[float, int]] = []
        for s in steiners:
            best = 1e30
            for t in terms:
                a, b = (s, t) if s < t else (t, s)
                w = wm[(a, b)]
                if w < best:
                    best = w
            scored.append((best, s))
        scored.sort(key=lambda x: x[0])
        pool_sz = min(len(scored), max_steiners * max(1, diversify_pool_mult))
        pool = [s for (_, s) in scored[:pool_sz]]
        chosen = pool if len(pool) <= max_steiners else rng.sample(pool, max_steiners)
        steiner_set = set(chosen)
        verts = terms + chosen
        mst_edges = _prim_mst_complete(verts, wm)
        pruned_edges = _prune_steiner_leaves(mst_edges, steiner_set)
    else:
        adj = _build_adj(instance)
        pruned_edges = _sph_local_tree(instance, terms, allowed_vertices=allowed, rng=rng, adj=adj)

    new_local = list(ps.local_edges) + [_norm_edge(u, v) for (u, v) in pruned_edges]
    new_local = list({e for e in new_local})

    new_meta = dict(ps.meta)
    new_meta.update({
        "local_repair": "R0_mst_prune" if instance.is_euclidean else "R0_sph",
        "local_cluster": c,
        "local_steiners_used": len({v for e in pruned_edges for v in e if instance.cluster_of[v] == -1}),
    })

    return replace(ps, local_edges=new_local, meta=new_meta)
