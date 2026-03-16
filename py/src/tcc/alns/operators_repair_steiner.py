from __future__ import annotations

import random
from typing import Dict, List, Tuple

from tcc.instance import Instance
from tcc.solution import Solution, TreeEdge

from .partial_state import PartialState
from .operators_repair import repair_r3_mst_components
from .finalize import finalize_with_kruskal


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


def _cost(instance: Instance, edges: List[TreeEdge]) -> float:
    wm = _weight_map(instance)
    total = 0.0
    for (u, v) in edges:
        a, b = (u, v) if u < v else (v, u)
        total += wm[(a, b)]
    return total


def repair_r4_steiner_hub(instance: Instance, ps: PartialState, rng: random.Random, max_candidates: int = 25) -> Solution:
    """
    R4 (Steiner Hub):
    - Se temos C componentes de clusters, escolhemos 1 vértice Steiner s
    - Ligamos s a 1 terminal “mais perto” de cada componente
    - Como s é novo, adicionamos C arestas e 1 vértice => volta a ser árvore

    Observação: se já estiver 1 componente, só retorna a solução reconstruída.
    """
    # se já está tudo conectado no nível de clusters, só "finaliza"
    if len(ps.components) <= 1:
        return finalize_with_kruskal(instance, ps.local_edges, ps.global_edges_remaining)

    used_vertices = set()
    for (u, v) in (ps.local_edges + ps.global_edges_remaining):
        used_vertices.add(u)
        used_vertices.add(v)

    # candidatos Steiner = vertices não-requeridos (-1) e ainda não usados
    steiners = [v for v in range(instance.n) if instance.cluster_of[v] == -1 and v not in used_vertices]
    if not steiners:
        # reconecta do jeito antigo
        return repair_r3_mst_components(instance, ps, rng)

    cand = rng.sample(steiners, min(max_candidates, len(steiners)))
    wm = _weight_map(instance)

    best_s = None
    best_sum = float("inf")
    best_attach: List[int] = []

    # pré-lista: terminais por componente
    comp_terminals: List[List[int]] = []
    for comp in ps.components:
        terminals: List[int] = []
        for cid in comp:
            terminals.extend(instance.clusters[cid])
        comp_terminals.append(terminals)

    for s in cand:
        attach: List[int] = []
        total = 0.0
        for terminals in comp_terminals:
            # escolhe o terminal mais próximo de s
            best_t = None
            best_w = float("inf")
            for t in terminals:
                a, b = (s, t) if s < t else (t, s)
                w = wm[(a, b)]
                if w < best_w:
                    best_w = w
                    best_t = t
            attach.append(best_t)  # type: ignore
            total += best_w

        if total < best_sum:
            best_sum = total
            best_s = s
            best_attach = attach

    assert best_s is not None

    new_edges = [_norm_edge(best_s, t) for t in best_attach]
    extra = list(ps.global_edges_remaining) + new_edges
    return finalize_with_kruskal(instance, ps.local_edges, extra)

def repair_r6_component_bridge_chain(
    instance: Instance,
    ps: PartialState,
    rng: random.Random,
    max_candidates: int = 25,
) -> Solution:
    """R6 (Component-Bridge Chain)

    Ideia:
      - Escolhe dois steiners *novos* s1 e s2
      - Escolhe um componente B ("ponte") que será conectado a AMBOS
      - Todo outro componente conecta a s1 ou a s2 (o mais barato)

    Estrutura típica (no nível de componentes):
      compA -- s1 -- compB -- s2 -- compC
    """
    if len(ps.components) <= 1:
        return finalize_with_kruskal(instance, ps.local_edges, ps.global_edges_remaining)

    used_vertices = set()
    for (u, v) in (ps.local_edges + ps.global_edges_remaining):
        used_vertices.add(u)
        used_vertices.add(v)

    steiners = [v for v in range(instance.n) if instance.cluster_of[v] == -1 and v not in used_vertices]
    if len(steiners) < 2:
        return repair_r3_mst_components(instance, ps, rng)

    cand = rng.sample(steiners, min(max_candidates, len(steiners)))
    if len(cand) < 2:
        return repair_r3_mst_components(instance, ps, rng)

    wm = _weight_map(instance)

    comp_terminals: List[List[int]] = []
    for comp in ps.components:
        terminals: List[int] = []
        for cid in comp:
            terminals.extend(instance.clusters[cid])
        comp_terminals.append(terminals)

    c = len(comp_terminals)
    INF = 1e30

    best_cost: List[List[float]] = [[INF] * c for _ in range(len(cand))]
    best_term: List[List[int]] = [[-1] * c for _ in range(len(cand))]

    for si, s in enumerate(cand):
        for ci, terms in enumerate(comp_terminals):
            bt = -1
            bw = INF
            for t in terms:
                a, b = (s, t) if s < t else (t, s)
                w = wm[(a, b)]
                if w < bw:
                    bw = w
                    bt = t
            best_cost[si][ci] = bw
            best_term[si][ci] = bt

    best_total = INF
    best_pair = None  
    best_bridge = None
    best_assign = None  # list[int] size c: 0->s1, 1->s2

    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            assign = [0] * c
            sum_min = 0.0
            for ci in range(c):
                if best_cost[i][ci] <= best_cost[j][ci]:
                    sum_min += best_cost[i][ci]
                    assign[ci] = 0
                else:
                    sum_min += best_cost[j][ci]
                    assign[ci] = 1

            bridge = 0
            extra_best = INF
            for ci in range(c):
                extra = max(best_cost[i][ci], best_cost[j][ci])
                if extra < extra_best:
                    extra_best = extra
                    bridge = ci

            total = sum_min + extra_best
            if total < best_total:
                best_total = total
                best_pair = (i, j)
                best_bridge = bridge
                best_assign = assign

    assert best_pair is not None and best_bridge is not None and best_assign is not None

    i, j = best_pair
    s1 = cand[i]
    s2 = cand[j]
    B = best_bridge

    base_edges = [_norm_edge(*e) for e in (ps.local_edges + ps.global_edges_remaining)]
    edge_set = set(base_edges)
    new_edges: List[TreeEdge] = []

    tB1 = best_term[i][B]
    tB2 = best_term[j][B]
    e1 = _norm_edge(s1, tB1)
    e2 = _norm_edge(s2, tB2)
    if e1 not in edge_set:
        edge_set.add(e1)
        new_edges.append(e1)
    if e2 not in edge_set:
        edge_set.add(e2)
        new_edges.append(e2)

    for ci in range(c):
        if ci == B:
            continue
        if best_assign[ci] == 0:
            t = best_term[i][ci]
            e = _norm_edge(s1, t)
        else:
            t = best_term[j][ci]
            e = _norm_edge(s2, t)

        if e not in edge_set:
            edge_set.add(e)
            new_edges.append(e)

    extra = list(ps.global_edges_remaining) + new_edges
    return finalize_with_kruskal(instance, ps.local_edges, extra)

def repair_r5_two_hubs_direct(
    instance: Instance,
    ps: PartialState,
    rng: random.Random,
    max_candidates: int = 25,
) -> Solution:
    """R5 (Two Steiner Hubs + ligação direta s1-s2)

    Ideia:
      - escolhe 2 steiners s1 e s2
      - cada componente conecta ao hub mais barato (s1 ou s2)
      - adiciona aresta direta s1-s2 para garantir conectividade entre os lados

    Estrutura macro:
      comps -> s1 -- s2 <- comps

    Obs: conecta hubs diretamente (diferente do R6 que usa componente-ponte).
    """
    if len(ps.components) <= 1:
        return finalize_with_kruskal(instance, ps.local_edges, ps.global_edges_remaining)

    used_vertices = set()
    for (u, v) in (ps.local_edges + ps.global_edges_remaining):
        used_vertices.add(u)
        used_vertices.add(v)

    steiners = [v for v in range(instance.n) if instance.cluster_of[v] == -1 and v not in used_vertices]
    if len(steiners) < 2:
        return repair_r3_mst_components(instance, ps, rng)

    cand = rng.sample(steiners, min(max_candidates, len(steiners)))
    if len(cand) < 2:
        return repair_r3_mst_components(instance, ps, rng)

    wm = _weight_map(instance)

    # terminais de cada componente
    comp_terminals: List[List[int]] = []
    for comp in ps.components:
        terminals: List[int] = []
        for cid in comp:
            terminals.extend(instance.clusters[cid])
        comp_terminals.append(terminals)

    c = len(comp_terminals)
    INF = 1e30

    # best_cost[si][ci] e best_term[si][ci]
    best_cost: List[List[float]] = [[INF] * c for _ in range(len(cand))]
    best_term: List[List[int]] = [[-1] * c for _ in range(len(cand))]

    for si, s in enumerate(cand):
        for ci, terms in enumerate(comp_terminals):
            bt = -1
            bw = INF
            for t in terms:
                a, b = (s, t) if s < t else (t, s)
                w = wm[(a, b)]
                if w < bw:
                    bw = w
                    bt = t
            best_cost[si][ci] = bw
            best_term[si][ci] = bt

    # pré-calc custo s1-s2
    sdist = [[0.0] * len(cand) for _ in range(len(cand))]
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = (cand[i], cand[j])
            a, b = (a, b) if a < b else (b, a)
            sdist[i][j] = wm[(a, b)]
            sdist[j][i] = sdist[i][j]

    best_total = INF
    best_pair = None
    best_assign = None  # 0->s1, 1->s2

    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            total = sdist[i][j]
            assign = [0] * c
            for ci in range(c):
                if best_cost[i][ci] <= best_cost[j][ci]:
                    total += best_cost[i][ci]
                    assign[ci] = 0
                else:
                    total += best_cost[j][ci]
                    assign[ci] = 1

            if total < best_total:
                best_total = total
                best_pair = (i, j)
                best_assign = assign

    assert best_pair is not None and best_assign is not None

    i, j = best_pair
    s1 = cand[i]
    s2 = cand[j]

    base_edges = [_norm_edge(*e) for e in (ps.local_edges + ps.global_edges_remaining)]
    edge_set = set(base_edges)
    new_edges: List[TreeEdge] = []

    # adiciona ligação direta s1-s2
    e12 = _norm_edge(s1, s2)
    if e12 not in edge_set:
        edge_set.add(e12)
        new_edges.append(e12)

    # conecta cada componente ao hub escolhido
    for ci in range(c):
        if best_assign[ci] == 0:
            t = best_term[i][ci]
            e = _norm_edge(s1, t)
        else:
            t = best_term[j][ci]
            e = _norm_edge(s2, t)

        if e not in edge_set:
            edge_set.add(e)
            new_edges.append(e)

    extra = list(ps.global_edges_remaining) + new_edges
    return finalize_with_kruskal(instance, ps.local_edges, extra)
