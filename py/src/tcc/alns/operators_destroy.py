from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from tcc.instance import Instance
from tcc.solution import Solution, TreeEdge

from .partial_state import PartialState


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


def _tree_path_edges(edges: List[TreeEdge], a: int, b: int) -> List[TreeEdge]:
    adj: Dict[int, List[int]] = {}
    for (u, v) in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    from collections import deque
    q = deque([a])
    parent: Dict[int, int] = {a: -1}
    while q:
        u = q.popleft()
        if u == b:
            break
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)

    if b not in parent:
        return []

    path_nodes: List[int] = []
    cur = b
    while cur != -1:
        path_nodes.append(cur)
        cur = parent[cur]
    path_nodes.reverse()

    out: List[TreeEdge] = []
    for i in range(len(path_nodes) - 1):
        out.append(_norm_edge(path_nodes[i], path_nodes[i + 1]))
    return out


def _norm_edge(u: int, v: int) -> TreeEdge:
    return (u, v) if u < v else (v, u)


def _root_tree(n: int, edges: List[TreeEdge]) -> Tuple[int, List[int], List[List[int]], Set[int]]:
    """

    Retorna:
      - root
      - parent[v] (root tem parent=-1; vertices não-usados ficam com parent=-2)
      - children[u] (lista de filhos)
      - used_vertices (vértices que aparecem em alguma aresta)
    """
    used: Set[int] = set()
    adj: List[List[int]] = [[] for _ in range(n)]
    for (u, v) in edges:
        used.add(u)
        used.add(v)
        adj[u].append(v)
        adj[v].append(u)

    if not used:
        # degenerate
        return 0, [-2] * n, [[] for _ in range(n)], used

    root = min(used)
    parent = [-2] * n
    parent[root] = -1
    children: List[List[int]] = [[] for _ in range(n)]

    q = [root]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        for v in adj[u]:
            if parent[v] != -2:
                continue
            parent[v] = u
            children[u].append(v)
            q.append(v)

    return root, parent, children, used


def compute_local_edges_per_cluster(instance: Instance, edges: List[TreeEdge]) -> List[Set[TreeEdge]]:
    """Computa, para cada cluster k, o conjunto de arestas que pertencem à *local tree* de k.

      cnt[v] = #terminais de R_k na subárvore de v.
      (parent[v],v) é local de k se 0 < cnt[v] < |R_k|.
    """
    norm_edges = [_norm_edge(u, v) for (u, v) in edges]
    root, parent, children, used = _root_tree(instance.n, norm_edges)

    # ordem pós-ordem
    stack = [root]
    order: List[int] = []
    while stack:
        u = stack.pop()
        order.append(u)
        for ch in children[u]:
            stack.append(ch)
    postorder = list(reversed(order))

    per_cluster: List[Set[TreeEdge]] = [set() for _ in range(len(instance.clusters))]

    for k, terms in enumerate(instance.clusters):
        sz = len(terms)
        if sz <= 1:
            continue

        is_term = [0] * instance.n
        for t in terms:
            is_term[t] = 1

        cnt = [0] * instance.n
        for v in postorder:
            if parent[v] == -2:
                continue
            s = is_term[v]
            for ch in children[v]:
                s += cnt[ch]
            cnt[v] = s

        # marca arestas locais
        for v in used:
            p = parent[v]
            if p < 0:
                continue
            if 0 < cnt[v] < sz:
                per_cluster[k].add(_norm_edge(p, v))

    return per_cluster


def compute_local_vertices_per_cluster(instance: Instance, edges: List[TreeEdge]) -> List[Set[int]]:
    """Conjunto de vértices de cada local tree (inclui terminais + steiners internos)."""
    per_edges = compute_local_edges_per_cluster(instance, edges)
    per_v: List[Set[int]] = []
    for k, terms in enumerate(instance.clusters):
        vs: Set[int] = set(terms)
        for (u, v) in per_edges[k]:
            vs.add(u)
            vs.add(v)
        per_v.append(vs)
    return per_v


def split_local_global_edges(instance: Instance, edges: List[TreeEdge]) -> Tuple[List[TreeEdge], List[TreeEdge]]:
    """

    LOCAL = união das arestas que pertencem às local trees dos clusters.
    GLOBAL = todas as demais arestas.
    """
    norm_edges = [_norm_edge(u, v) for (u, v) in edges]
    per_cluster_local = compute_local_edges_per_cluster(instance, norm_edges)
    local_set: Set[TreeEdge] = set()
    for s in per_cluster_local:
        local_set |= s

    local: List[TreeEdge] = []
    global_: List[TreeEdge] = []
    for e in norm_edges:
        if e in local_set:
            local.append(e)
        else:
            global_.append(e)
    return local, global_


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def compute_cluster_components(
    instance: Instance,
    global_edges: List[TreeEdge],
    local_edges: Optional[List[TreeEdge]] = None,
) -> List[List[int]]:
    """
    Repairs podem conectar no GLOBAL via vértices Steiner que estão na local tree.
    Então precisamos unir também local_edges no DSU, senão componentes nunca se fundem.
    """
    dsu = DSU(instance.n)

    # 1) contrai cada cluster (terminais)
    for terminals in instance.clusters:
        if not terminals:
            continue
        t0 = terminals[0]
        for t in terminals[1:]:
            dsu.union(t0, t)

    # 2) une arestas LOCAIS (faz Steiner local pertencer ao cluster)
    if local_edges:
        for (u, v) in local_edges:
            dsu.union(u, v)

    # 3) une arestas GLOBAIS
    for (u, v) in global_edges:
        dsu.union(u, v)

    root_to_clusters: Dict[int, List[int]] = {}
    for cid, terminals in enumerate(instance.clusters):
        t0 = terminals[0]
        r = dsu.find(t0)
        root_to_clusters.setdefault(r, []).append(cid)

    return list(root_to_clusters.values())


def _build_cluster_to_component(num_clusters: int, components: List[List[int]]) -> List[int]:
    out = [-1] * num_clusters
    for comp_id, comp in enumerate(components):
        for c in comp:
            out[c] = comp_id
    return out


def destroy_d1_remove_k_global_edges(instance: Instance, solution: Solution, rng: random.Random, k: int = 2) -> PartialState:
    local_edges, global_edges = split_local_global_edges(instance, solution.edges)

    if len(global_edges) == 0:
        components = compute_cluster_components(instance, global_edges, local_edges)
        cluster_to_component = _build_cluster_to_component(len(instance.clusters), components)
        return PartialState(
            base_solution=solution,
            local_edges=local_edges,
            global_edges_remaining=global_edges,
            global_edges_removed=[],
            components=components,
            cluster_to_component=cluster_to_component,
            destroyed_cluster=None,
            meta={"destroy_op": "D1_remove_k_global_edges", "k": 0},
        )

    kk = min(k, len(global_edges))
    removed = rng.sample(global_edges, kk)
    removed_set = set(removed)
    remaining = [e for e in global_edges if e not in removed_set]

    components = compute_cluster_components(instance, remaining, local_edges)
    cluster_to_component = _build_cluster_to_component(len(instance.clusters), components)

    return PartialState(
        base_solution=solution,
        local_edges=local_edges,
        global_edges_remaining=remaining,
        global_edges_removed=removed,
        components=components,
        cluster_to_component=cluster_to_component,
        destroyed_cluster=None,
        meta={"destroy_op": "D1_remove_k_global_edges", "k": kk},
    )


def destroy_d2_disconnect_cluster(instance: Instance, solution: Solution, rng: random.Random) -> PartialState:
    local_edges, global_edges = split_local_global_edges(instance, solution.edges)

    num_clusters = len(instance.clusters)
    c = rng.randrange(num_clusters)
    terminals = set(instance.clusters[c])

    incident = [e for e in global_edges if (e[0] in terminals) or (e[1] in terminals)]

    if not incident:
        components = compute_cluster_components(instance, global_edges, local_edges)
        cluster_to_component = _build_cluster_to_component(num_clusters, components)
        return PartialState(
            base_solution=solution,
            local_edges=local_edges,
            global_edges_remaining=global_edges,
            global_edges_removed=[],
            components=components,
            cluster_to_component=cluster_to_component,
            destroyed_cluster=c,
            meta={"destroy_op": "D2_disconnect_cluster", "note": "no incident global edge"},
        )

    removed_edge = rng.choice(incident)
    remaining = [e for e in global_edges if e != removed_edge]

    components = compute_cluster_components(instance, remaining, local_edges)
    cluster_to_component = _build_cluster_to_component(num_clusters, components)

    return PartialState(
        base_solution=solution,
        local_edges=local_edges,
        global_edges_remaining=remaining,
        global_edges_removed=[removed_edge],
        components=components,
        cluster_to_component=cluster_to_component,
        destroyed_cluster=c,
        meta={"destroy_op": "D2_disconnect_cluster"},
    )


def destroy_d3_break_local_tree(instance: Instance, solution: Solution, rng: random.Random) -> PartialState:
    """D3: escolhe um cluster c e REMOVE toda a local tree dele.

    - força o repair a reconstruir a árvore local do cluster.
    - arestas globais ficam intactas.
    """
    num_clusters = len(instance.clusters)
    c = rng.randrange(num_clusters)

    per_local = compute_local_edges_per_cluster(instance, solution.edges)
    removed_local = per_local[c]

    remaining_edges = [_norm_edge(u, v) for (u, v) in solution.edges if _norm_edge(u, v) not in removed_local]
    local_edges, global_edges = split_local_global_edges(instance, remaining_edges)

    components = compute_cluster_components(instance, global_edges, local_edges)
    cluster_to_component = _build_cluster_to_component(num_clusters, components)

    return PartialState(
        base_solution=solution,
        local_edges=local_edges,
        global_edges_remaining=global_edges,
        global_edges_removed=[],
        components=components,
        cluster_to_component=cluster_to_component,
        destroyed_cluster=c,
        meta={
            "destroy_op": "D3_break_local_tree",
            "cluster": c,
            "removed_local_edges": len(removed_local),
        },
    )


destroy_remove_k_global_edges = destroy_d1_remove_k_global_edges
destroy_disconnect_cluster = destroy_d2_disconnect_cluster
destroy_break_local_tree = destroy_d3_break_local_tree


def destroy_d4_remove_worst_k_global_edges(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    k: int = 2,
    top_frac: float = 0.30,
) -> PartialState:
    """D4: remove k arestas globais "caras" (worst removal)."""
    local_edges, global_edges = split_local_global_edges(instance, solution.edges)
    if not global_edges:
        components = compute_cluster_components(instance, global_edges, local_edges)
        cluster_to_component = _build_cluster_to_component(len(instance.clusters), components)
        return PartialState(
            base_solution=solution,
            local_edges=local_edges,
            global_edges_remaining=global_edges,
            global_edges_removed=[],
            components=components,
            cluster_to_component=cluster_to_component,
            destroyed_cluster=None,
            meta={"destroy_op": "D4_remove_worst_k_global_edges", "k": 0},
        )

    wm = _weight_map(instance)
    scored = []
    for (u, v) in global_edges:
        a, b = (u, v) if u < v else (v, u)
        scored.append((wm[(a, b)], (u, v)))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = max(1, int(len(scored) * top_frac))
    pool = scored[:top]
    kk = min(k, len(pool))

    # pool: lista de (cost, edge) já ordenada (pior -> melhor)
    pool_edges = [e for (_, e) in pool]
    weights = [max(0.0, float(w)) for (w, _) in pool]

    chosen: List[TreeEdge] = []
    for _ in range(kk):
        if not pool_edges:
            break

        # pode virar tudo 0 depois de alguns pops
        if sum(weights) <= 0.0:
            # fallback: escolha uniforme (qualquer aresta)
            idx = rng.randrange(len(pool_edges))
        else:
            idx = rng.choices(range(len(pool_edges)), weights=weights, k=1)[0]

        chosen.append(pool_edges.pop(idx))
        weights.pop(idx)

    removed_set = set(chosen)
    remaining = [e for e in global_edges if e not in removed_set]

    components = compute_cluster_components(instance, remaining, local_edges)
    cluster_to_component = _build_cluster_to_component(len(instance.clusters), components)
    return PartialState(
        base_solution=solution,
        local_edges=local_edges,
        global_edges_remaining=remaining,
        global_edges_removed=chosen,
        components=components,
        cluster_to_component=cluster_to_component,
        destroyed_cluster=None,
        meta={"destroy_op": "D4_remove_worst_k_global_edges", "k": kk, "top_frac": top_frac},
    )


def destroy_d5_remove_global_path_segment(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    max_remove: int = 4,
) -> PartialState:
    """D5: remove um segmento de arestas GLOBAIS do caminho entre 2 clusters."""
    local_edges, global_edges = split_local_global_edges(instance, solution.edges)
    if len(instance.clusters) <= 1 or not global_edges:
        return destroy_d1_remove_k_global_edges(instance, solution, rng, k=min(2, len(global_edges)))

    c1, c2 = rng.sample(range(len(instance.clusters)), 2)
    a = rng.choice(instance.clusters[c1])
    b = rng.choice(instance.clusters[c2])

    path = _tree_path_edges([_norm_edge(u, v) for (u, v) in solution.edges], a, b)
    if not path:
        return destroy_d1_remove_k_global_edges(instance, solution, rng, k=min(2, len(global_edges)))

    global_set = set(global_edges)
    path_global = [e for e in path if e in global_set]
    if not path_global:
        return destroy_d1_remove_k_global_edges(instance, solution, rng, k=min(2, len(global_edges)))

    kk = min(max_remove, len(path_global))
    start = rng.randrange(0, len(path_global) - kk + 1)
    removed = path_global[start:start + kk]
    removed_set = set(removed)
    remaining = [e for e in global_edges if e not in removed_set]

    components = compute_cluster_components(instance, remaining, local_edges)
    cluster_to_component = _build_cluster_to_component(len(instance.clusters), components)
    return PartialState(
        base_solution=solution,
        local_edges=local_edges,
        global_edges_remaining=remaining,
        global_edges_removed=removed,
        components=components,
        cluster_to_component=cluster_to_component,
        destroyed_cluster=None,
        meta={"destroy_op": "D5_remove_global_path_segment", "max_remove": kk, "c1": c1, "c2": c2},
    )


def destroy_d6_remove_global_steiner_star(
    instance: Instance,
    solution: Solution,
    rng: random.Random,
    max_remove: int = 6,
) -> PartialState:
    """D6: escolhe um Steiner que aparece no GLOBAL e remove várias arestas globais incidentes."""
    local_edges, global_edges = split_local_global_edges(instance, solution.edges)
    if not global_edges:
        return destroy_d1_remove_k_global_edges(instance, solution, rng, k=0)

    deg: Dict[int, int] = {}
    for (u, v) in global_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    steiners = [v for v, d in deg.items() if instance.cluster_of[v] == -1 and d >= 2]
    if not steiners:
        return destroy_d4_remove_worst_k_global_edges(instance, solution, rng, k=2)

    s = rng.choice(steiners)
    incident = [e for e in global_edges if e[0] == s or e[1] == s]
    kk = min(max_remove, len(incident))
    removed = rng.sample(incident, kk)
    removed_set = set(removed)
    remaining = [e for e in global_edges if e not in removed_set]

    components = compute_cluster_components(instance, remaining, local_edges)
    cluster_to_component = _build_cluster_to_component(len(instance.clusters), components)
    return PartialState(
        base_solution=solution,
        local_edges=local_edges,
        global_edges_remaining=remaining,
        global_edges_removed=removed,
        components=components,
        cluster_to_component=cluster_to_component,
        destroyed_cluster=None,
        meta={"destroy_op": "D6_remove_global_steiner_star", "steiner": s, "removed": kk},
    )


destroy_remove_worst_k_global_edges = destroy_d4_remove_worst_k_global_edges
destroy_remove_global_path_segment = destroy_d5_remove_global_path_segment
destroy_remove_global_steiner_star = destroy_d6_remove_global_steiner_star