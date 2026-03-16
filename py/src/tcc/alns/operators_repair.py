from __future__ import annotations

import heapq
import random
from typing import Dict, List, Tuple, Optional, Set

from tcc.instance import Instance
from tcc.solution import Solution, TreeEdge

from .partial_state import PartialState
from .operators_destroy import compute_cluster_components  
from .finalize import finalize_with_kruskal, build_weight_lookup


def _norm_edge(e: TreeEdge) -> TreeEdge:
    u, v = e
    return (u, v) if u < v else (v, u)



def build_adj(inst: Instance) -> List[List[Tuple[int, float]]]:
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(inst.n)]
    for u, v, c in inst.edges:
        cc = float(c)
        adj[u].append((v, cc))
        adj[v].append((u, cc))
    return adj


def dijkstra_all(
    inst: Instance,
    adj: List[List[Tuple[int, float]]],
    sources: List[int],
    *,
    blocked: Optional[Set[int]] = None,
    ban_expand: Optional[Set[int]] = None,
) -> Tuple[List[float], List[int]]:
    INF = 10**30
    n = inst.n
    dist = [INF] * n
    parent = [-1] * n
    pq: List[Tuple[float, int]] = []

    source_set = set(sources)
    blocked = blocked or set()
    ban_expand = ban_expand or set()

    for s in sources:
        dist[s] = 0.0
        parent[s] = -1
        heapq.heappush(pq, (0.0, s))

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue

        if u in blocked:
            continue

        if u in ban_expand and u not in source_set:
            continue

        for v, w_uv in adj[u]:
            if v in blocked:
                continue
            nd = d + w_uv
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, parent

def _build_cluster_to_component(num_clusters: int, components: List[List[int]]) -> List[int]:
    out = [-1] * num_clusters
    for comp_id, comp in enumerate(components):
        for c in comp:
            out[c] = comp_id
    return out

def reconstruct_path_edges(parent: List[int], target: int) -> List[TreeEdge]:
    """
    Reconstrói o caminho (lista de arestas) voltando do target até alguma fonte (parent=-1).
    """
    edges: List[TreeEdge] = []
    cur = target
    while parent[cur] != -1:
        p = parent[cur]
        edges.append(_norm_edge((p, cur)))
        cur = p
    edges.reverse()
    return edges


def _cluster_local_vertices_from_local_edges(inst: Instance, local_edges: List[TreeEdge]) -> List[Set[int]]:
    """
    Extrai, para cada cluster k, o conjunto de vértices da local tree atual
    usando apenas as arestas locais (que já são disjuntas).
    """
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


# Repair R1 — reconecta com Dijkstra

def repair_r1_dijkstra(inst: Instance, ps: PartialState, rng: random.Random) -> Solution:
    """
    R1: reconectar componentes usando Dijkstra multi-source repetidamente.
    """
    adj = build_adj(inst)
    w = build_weight_lookup(inst)

    local_edges = [_norm_edge(e) for e in ps.local_edges]
    global_edges = [_norm_edge(e) for e in ps.global_edges_remaining]
    global_set = set(global_edges)

    INF = 10**30

    while True:
        components = compute_cluster_components(inst, global_edges, local_edges)
        cluster_to_component = _build_cluster_to_component(len(inst.clusters), components)
        if len(components) <= 1:
            break

        # componente base: se D2 marcou um cluster, usa o componente dele
        if ps.destroyed_cluster is not None:
            base_component_id = cluster_to_component[ps.destroyed_cluster]
        else:
            base_component_id = rng.randrange(len(components))

        base_clusters = components[base_component_id]

        # fontes = TODOS os vértices das local trees dos clusters da componente base.
        per_local_v = _cluster_local_vertices_from_local_edges(inst, local_edges)
        sources: List[int] = []
        for ck in base_clusters:
            sources.extend(list(per_local_v[ck]))
        source_set = set(sources)

        all_terms = set(inst.terminals)
        ban_expand = all_terms - (source_set & all_terms)

        def pick_target(dist: List[float], parent: List[int]) -> Optional[int]:
            bt = None
            bc = INF
            for k in range(len(inst.clusters)):
                if cluster_to_component[k] == base_component_id:
                    continue
                for v in inst.clusters[k]:
                    # precisa ter um pai (caminho não-vazio)
                    if parent[v] != -1 and dist[v] < bc:
                        bc = dist[v]
                        bt = v
            return bt

        # 1) não usa terminais como intermediário
        dist, parent = dijkstra_all(inst, adj, sources, ban_expand=ban_expand)
        best_target = pick_target(dist, parent)

        # 2) se não achou alvo com parent!=-1, roda Dijkstra sem restrição
        if best_target is None:
            dist2, parent2 = dijkstra_all(inst, adj, sources, ban_expand=set())
            best_target = pick_target(dist2, parent2)
            dist, parent = dist2, parent2

        # 3) Se ainda não achou, tenta fallback por aresta direta source->terminal
        if best_target is None:
            # escolhe o terminal fora do componente base mais barato por aresta direta
            best_u = None
            best_v = None
            best_cost = INF

            for k in range(len(inst.clusters)):
                if cluster_to_component[k] == base_component_id:
                    continue
                for v in inst.clusters[k]:
                    for u in sources:
                        e = _norm_edge((u, v))
                        ww = w.get(e, INF)
                        if ww < best_cost:
                            best_cost = ww
                            best_u = u
                            best_v = v

            if best_u is None:
                raise RuntimeError("R1: não encontrou conexão para outra componente.")

            e = _norm_edge((best_u, best_v))
            if e not in global_set:
                global_set.add(e)
                global_edges.append(e)
            continue

        # Reconstrói caminho 
        path_edges = reconstruct_path_edges(parent, best_target)

        if not path_edges:
            # isso só acontece se best_target acabar sendo source
            bu = None
            bc = INF
            for u in sources:
                e = _norm_edge((u, best_target))
                ww = w.get(e, INF)
                if ww < bc:
                    bc = ww
                    bu = u
            if bu is None:
                raise RuntimeError("R1: caminho vazio.")
            e = _norm_edge((bu, best_target))
            if e not in global_set:
                global_set.add(e)
                global_edges.append(e)
        else:
            for e in path_edges:
                if e not in global_set:
                    global_set.add(e)
                    global_edges.append(e)

    sol = finalize_with_kruskal(inst, local_edges, global_edges)
    return sol


# Repair R3  — MST entre componentes + expandir caminhos

def prim_mst_components(weights: List[List[float]]) -> List[Tuple[int, int]]:
    n = len(weights)
    if n <= 1:
        return []

    INF = 10**30
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

        for v in range(n):
            if in_mst[v] or v == u:
                continue
            w = weights[u][v]
            if w < key[v]:
                key[v] = w
                parent[v] = u

    edges = []
    for v in range(1, n):
        if parent[v] == -1:
            raise RuntimeError("Prim falhou.")
        edges.append((parent[v], v))
    return edges


def repair_r3_mst_components(inst: Instance, ps: PartialState, rng: random.Random) -> Solution:
    adj = build_adj(inst)
    wlookup = build_weight_lookup(inst)

    local_edges = [_norm_edge(e) for e in ps.local_edges]
    global_edges = [_norm_edge(e) for e in ps.global_edges_remaining]
    global_set = set(global_edges)

    components = compute_cluster_components(inst, global_edges, local_edges)
    cluster_to_component = _build_cluster_to_component(len(inst.clusters), components)
    c = len(components)

    if c <= 1:
        final_edges = list(local_edges) + list(global_edges)
        cost = sum(wlookup[(u, v)] for (u, v) in final_edges)
        return Solution(instance_name=inst.name, cost=cost, edges=final_edges)

    # comp_vertices[i] = todos os vértices das local trees que pertencem aos clusters daquela componente
    per_local_v = _cluster_local_vertices_from_local_edges(inst, local_edges)
    comp_vertices: List[List[int]] = []
    for comp in components:
        verts = []
        for ck in comp:
            verts.extend(list(per_local_v[ck]))
        comp_vertices.append(verts)

    # Para cada componente i:
    # - roda Dijkstra de todas fontes em comp_vertices[i]
    # - escolhe melhor terminal em cada componente j como alvo
    parents: List[List[int]] = []
    dist_lists: List[List[float]] = []
    best_target: List[List[Optional[int]]] = [[None] * c for _ in range(c)]
    best_dist: List[List[float]] = [[10**30] * c for _ in range(c)]

    for i in range(c):
        all_terms = set(inst.terminals)
        ban_expand = all_terms - (set(inst.terminals) & set(comp_vertices[i]))
        dist, parent = dijkstra_all(inst, adj, comp_vertices[i], ban_expand=ban_expand)
        parents.append(parent)
        dist_lists.append(dist)

        for j in range(c):
            if i == j:
                best_dist[i][j] = 0.0
                best_target[i][j] = None
                continue

            # melhor terminal dentro da componente j
            t_best = None
            d_best = 10**30
            for v in comp_vertices[j]:
                if dist[v] < d_best:
                    d_best = dist[v]
                    t_best = v

            best_dist[i][j] = d_best
            best_target[i][j] = t_best

    weights = [[0.0] * c for _ in range(c)]
    for i in range(c):
        for j in range(c):
            if i == j:
                weights[i][j] = 0.0
            else:
                weights[i][j] = min(best_dist[i][j], best_dist[j][i])

    # MST no nível de componentes
    mst_edges = prim_mst_components(weights)

    # expandir cada aresta da MST em caminho real
    for a, b in mst_edges:
        # escolhe direção que tem target válido (sempre deve ter)
        ta = best_target[a][b]
        tb = best_target[b][a]

        if ta is not None and best_dist[a][b] <= best_dist[b][a]:
            path_edges = reconstruct_path_edges(parents[a], ta)
        elif tb is not None:
            path_edges = reconstruct_path_edges(parents[b], tb)
        else:
            raise RuntimeError("R3: não encontrou target para reconstruir.")

        if not path_edges:
            raise RuntimeError("R3: caminho reconstruído vazio.")

        for e in path_edges:
            if e not in global_set:
                global_set.add(e)
                global_edges.append(e)

    sol = finalize_with_kruskal(inst, local_edges, global_edges)
    return sol


def repair_r1_dijkstra_topL(inst: Instance, ps: PartialState, rng: random.Random, L: int = 5) -> Solution:
    """
    R1-TopL:
      igual ao R1, mas ao conectar uma componente com outra,
      escolhe aleatoriamente um alvo entre os TOP-L melhores (menor dist).

    """
    adj = build_adj(inst)
    w = build_weight_lookup(inst)

    local_edges = [_norm_edge(e) for e in ps.local_edges]
    global_edges = [_norm_edge(e) for e in ps.global_edges_remaining]
    global_set = set(global_edges)

    while True:
        components = compute_cluster_components(inst, global_edges, local_edges)
        cluster_to_component = _build_cluster_to_component(len(inst.clusters), components)

        if len(components) <= 1:
            break

        # componente base
        if ps.destroyed_cluster is not None:
            base_component_id = cluster_to_component[ps.destroyed_cluster]
        else:
            base_component_id = rng.randrange(len(components))

        base_clusters = components[base_component_id]

        per_local_v = _cluster_local_vertices_from_local_edges(inst, local_edges)
        sources: List[int] = []
        for ck in base_clusters:
            sources.extend(list(per_local_v[ck]))

        all_terms = set(inst.terminals)
        ban_expand = all_terms - (set(inst.terminals) & set(sources))
        dist, parent = dijkstra_all(inst, adj, sources, ban_expand=ban_expand)

        # candidatos: todos terminais fora da componente base
        cand: List[Tuple[float, int]] = []
        for k in range(len(inst.clusters)):
            if cluster_to_component[k] == base_component_id:
                continue
            for v in inst.clusters[k]:
                # Só considera alvos que realmente têm caminho reconstruível (parent!=-1)
                if parent[v] != -1 and dist[v] < 10**29:
                    cand.append((dist[v], v))

        # se não tiver nenhum, roda dijkstra sem ban_expand
        if not cand:
            dist2, parent2 = dijkstra_all(inst, adj, sources, ban_expand=set())
            cand = []
            for k in range(len(inst.clusters)):
                if cluster_to_component[k] == base_component_id:
                    continue
                for v in inst.clusters[k]:
                    if parent2[v] != -1 and dist2[v] < 10**29:
                        cand.append((dist2[v], v))
            dist, parent = dist2, parent2

        cand.sort(key=lambda x: x[0])
        if not cand:
            raise RuntimeError("R1-TopL: não encontrou conexão para outra componente.")

        L_eff = min(L, len(cand))
        _, target = rng.choice(cand[:L_eff])

        path_edges = reconstruct_path_edges(parent, target)
        if not path_edges:
            # conecta source->target por aresta direta
            best_u = None
            best_cost = 10**30
            for u in sources:
                e = _norm_edge((u, target))
                ww = w.get(e, 10**30)
                if ww < best_cost:
                    best_cost = ww
                    best_u = u
            if best_u is None:
                raise RuntimeError("R1-TopL: caminho vazio.")
            path_edges = [_norm_edge((best_u, target))]
        
        for e in path_edges:
            if e not in global_set:
                global_set.add(e)
                global_edges.append(e)

    sol = finalize_with_kruskal(inst, local_edges, global_edges)
    return sol
