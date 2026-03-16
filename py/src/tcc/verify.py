from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

from .instance import Instance
from .solution import Solution, TreeEdge


class VerificationResult:
    def __init__(self, feasible: bool, violations: List[str], cost: float | None = None):
        self.feasible = feasible
        self.violations = violations
        self.cost = cost  # custo recalculado (opcional, por enquanto pode ser None)

    def __repr__(self) -> str:
        return f"VerificationResult(feasible={self.feasible}, violations={self.violations}, cost={self.cost})"


def _build_solution_graph(edges: List[TreeEdge]) -> Tuple[Dict[int, List[int]], Set[int]]:
    """Constrói o grafo da solução (não-direcionado) a partir da lista de arestas."""

    adj: Dict[int, List[int]] = {}
    used_vertices: Set[int] = set()

    for u, v in edges:
        used_vertices.add(u)
        used_vertices.add(v)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    return adj, used_vertices


def _check_tree(adj: Dict[int, List[int]], used_vertices: Set[int]) -> List[str]:
    """Verifica se o grafo definido por adj/used_vertices é uma árvore."""

    violations: List[str] = []

    if not used_vertices:
        violations.append("Solução não contém vértices (lista de arestas vazia).")
        return violations

    # BFS/DFS a partir de um vértice qualquer
    start = next(iter(used_vertices))
    visited: Set[int] = set()
    parent: Dict[int, int] = {}

    queue = deque([start])
    visited.add(start)

    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                queue.append(v)
            else:
                if parent.get(u) != v:
                    pass

    if visited != used_vertices:
        missing = used_vertices - visited
        violations.append(
            f"Árvore não conexa: vértices não alcançáveis a partir de {start}: {sorted(missing)}"
        )

    # Checagem: |E| = |V| - 1
    num_edges = sum(len(vs) for vs in adj.values()) // 2  # cada aresta contada duas vezes
    num_vertices = len(used_vertices)
    if num_edges != num_vertices - 1:
        violations.append(
            f"Não satisfaz |E| = |V| - 1 (tem {num_edges} arestas para {num_vertices} vértices). "
            "Provavelmente não é uma árvore (pode ter ciclo ou múltiplos componentes)."
        )

    return violations


def _check_terminals(instance: Instance, used_vertices: Set[int]) -> List[str]:
    """Verifica se todos os vértices terminais R aparecem na solução."""
    violations: List[str] = []
    missing = set(instance.terminals) - used_vertices
    if missing:
        violations.append(f"Terminais ausentes na solução: {sorted(missing)}")
    return violations


def _shortest_path_tree(adj: Dict[int, List[int]], start: int, target: int) -> List[int]:
    """
    Encontra um caminho (não ponderado) entre start e target no grafo da solução.
    Como a solução deveria ser uma árvore, esse caminho é único se existir.
    """
    if start == target:
        return [start]

    queue = deque([start])
    parent: Dict[int, int] = {start: -1}

    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                queue.append(v)
                if v == target:
                    queue.clear()
                    break

    if target not in parent:
        # não há caminho no grafo da solução (árvore quebrada)
        return []

    # Reconstrói caminho target
    path = []
    cur = target
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _compute_local_tree_vertices_for_cluster(
    adj: Dict[int, List[int]],
    cluster_terminals: List[int],
) -> Set[int]:
    """
    Para um cluster R_k, retorna o conjunto de vértices V_k do menor subgrafo
    que conecta todos os seus terminais, assumindo que a solução é uma árvore.

    Implementação simples:
    - Para cada par de terminais do cluster, pega o caminho entre eles na árvore.
    - V_k é a união de todos os vértices de todos esses caminhos.
    """
    V_k: Set[int] = set()

    if not cluster_terminals:
        return V_k

    # Para evitar O(|R_k|^2) muito grande, poderíamos fixar um "root" e conectar tudo a ele,
    # mas por agora da pra manter simples.
    t_list = cluster_terminals
    for i in range(len(t_list)):
        for j in range(i + 1, len(t_list)):
            a = t_list[i]
            b = t_list[j]
            path = _shortest_path_tree(adj, a, b)
            if not path:
                # Se não há caminho, a solução já está quebrada como árvore
                # (isso será pego nas outras checagens), mas aqui marcamos V_k parcial.
                continue
            V_k.update(path)

    # Caso o cluster tenha só um terminal, a "local tree" é apenas esse vértice.
    if len(t_list) == 1:
        V_k.add(t_list[0])

    return V_k


def _check_cluster_disjointness(instance: Instance, adj: Dict[int, List[int]]) -> List[str]:
    """
    Checa se os local trees dos clusters são disjuntos.

    Para cada cluster R_k:
      - calcula V_k, o conjunto de vértices do menor subgrafo que conecta R_k;
    Depois verifica se V_i ∩ V_j = ∅ para todo i ≠ j.
    """
    violations: List[str] = []
    cluster_vertex_sets: List[Set[int]] = []

    # Construir listas de terminais por cluster a partir de instance.clusters
    for k, Ck in enumerate(instance.clusters):
        if not Ck:
            violations.append(f"Cluster {k} está vazio na Instance (violação de modelo).")
            continue

        Vk = _compute_local_tree_vertices_for_cluster(adj, Ck)
        if not Vk:
            violations.append(
                f"Cluster {k} (terminais {Ck}) não está conectado na solução (local tree vazio)."
            )
        cluster_vertex_sets.append(Vk)

    # Checar interseções
    h = len(cluster_vertex_sets)
    for i in range(h):
        for j in range(i + 1, h):
            inter = cluster_vertex_sets[i] & cluster_vertex_sets[j]
            if inter:
                violations.append(
                    f"Violação de disjunção entre clusters {i} e {j}: "
                    f"local trees compartilham vértices {sorted(inter)}"
                )

    return violations


def verify_solution(instance: Instance, solution: Solution) -> VerificationResult:
    """
    Verifica se a solução é factível para a instância, de acordo com as regras:

      - Árvore (conexa, acíclica)
      - Cobertura de todos os terminais
      - Local trees por cluster disjuntos

    Ainda não estamos recalculando o custo a partir de Instance.edges,
    isso será conectado quando o loader real de arestas estiver pronto.
    """
    violations: List[str] = []

    # Construir grafo da solução
    adj, used_vertices = _build_solution_graph(solution.edges)

    # 1) Checar se índices das arestas estão no range
    for u, v in solution.edges:
        if not (0 <= u < instance.n and 0 <= v < instance.n):
            violations.append(
                f"Aresta ({u}, {v}) fora do range [0, {instance.n - 1}]"
            )

    # 2) Checar se é árvore
    violations.extend(_check_tree(adj, used_vertices))

    # 3) Cobertura de terminais
    violations.extend(_check_terminals(instance, used_vertices))

    # 4) Disjunção dos clusters (local trees)
    violations.extend(_check_cluster_disjointness(instance, adj))

    feasible = len(violations) == 0
    return VerificationResult(feasible=feasible, violations=violations, cost=None)
