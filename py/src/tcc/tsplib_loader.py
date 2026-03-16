from __future__ import annotations

"""
TSPLIB-like loader para as instâncias do dataset do paper.

Suporta:
  - Euclidiano: NODE_COORD_SECTION + (GTSP_SET_SECTION ou CLUSTER_SECTION)
  - Não-euclidiano: EDGE_WEIGHT_SECTION (matriz n×n) + CLUSTER_SECTION
  - Type5: normalmente CLUSTER_SECTION
  - Type1/Type6: às vezes GTSP_SET_SECTION
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import re

from .instance import Instance


def _tsplib_euc_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distância TSPLIB EUC_2D (arredondamento para inteiro)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(int(math.sqrt(dx * dx + dy * dy) + 0.5))


def _find_section_idx(lines: List[str], section_name_upper: str) -> Optional[int]:
    """Acha o índice de uma linha que começa com o nome da seção (case-insensitive)."""
    sec = section_name_upper.upper()
    for i, ln in enumerate(lines):
        up = ln.strip().upper()
        # aceita "CLUSTER_SECTION:" também
        if up.startswith(sec):
            return i
    return None


def _parse_header_value(lines: List[str], key: str) -> Optional[str]:
    """Extrai 'KEY : value' ou 'KEY: value' (case-insensitive)."""
    rx = re.compile(rf"^\s*{re.escape(key)}\s*:?\s*(.*)\s*$", re.IGNORECASE)
    for ln in lines:
        m = rx.match(ln)
        if m:
            val = m.group(1).strip()
            return val if val else None
    return None


def _parse_int_header(lines: List[str], *keys: str) -> Optional[int]:
    for k in keys:
        v = _parse_header_value(lines, k)
        if v is None:
            continue
        m = re.search(r"-?\d+", v)
        if m:
            return int(m.group(0))
    return None


def _parse_coords(lines: List[str], start_idx: int, n: int) -> List[tuple[float, float]]:
    """Lê exatamente n linhas de coordenadas após NODE_COORD_SECTION."""
    coords: List[tuple[float, float]] = [(0.0, 0.0)] * n
    idx = start_idx
    read = 0

    while idx < len(lines) and read < n:
        toks = lines[idx].split()
        idx += 1
        if len(toks) < 3:
            continue
        node_id = int(toks[0]) - 1
        x = float(toks[1])
        y = float(toks[2])
        if not (0 <= node_id < n):
            raise ValueError(f"NODE_COORD_SECTION: id {node_id+1} fora de 1..{n}")
        coords[node_id] = (x, y)
        read += 1

    if read != n:
        raise ValueError(f"NODE_COORD_SECTION: esperava {n} linhas, li {read}")
    return coords


def _parse_full_matrix_numbers(lines: List[str], start_idx: int, end_idx: int, n: int) -> List[float]:
    """Extrai n*n números da matriz EDGE_WEIGHT_SECTION."""
    blob = "\n".join(lines[start_idx:end_idx])
    nums = re.findall(r"-?\d+(?:\.\d+)?", blob)
    if len(nums) != n * n:
        raise ValueError(
            f"EDGE_WEIGHT_SECTION: esperava {n*n} números (matriz {n}x{n}), mas encontrei {len(nums)}"
        )
    return [float(x) for x in nums]


def _parse_clusters_from_section(lines: List[str], start_idx: int) -> List[List[int]]:
    """
    Parseia GTSP_SET_SECTION / CLUSTER_SECTION:
      formato: <cluster_id> v1 v2 ... -1
    """
    tail = "\n".join(lines[start_idx + 1 :])
    tokens = [int(x) for x in re.findall(r"-?\d+", tail)]
    if not tokens:
        raise ValueError("Seção de clusters vazia")

    clusters_by_id: Dict[int, List[int]] = {}
    i = 0
    while i < len(tokens):
        cid = tokens[i]
        i += 1
        vs: List[int] = []
        while i < len(tokens) and tokens[i] != -1:
            vs.append(tokens[i] - 1)  # 0-based
            i += 1
        if i < len(tokens) and tokens[i] == -1:
            i += 1
        if vs:
            clusters_by_id[cid] = vs

    if not clusters_by_id:
        raise ValueError("Não consegui ler nenhum cluster (esperava: 'k v1 v2 ... -1')")

    return [clusters_by_id[cid] for cid in sorted(clusters_by_id.keys())]


def load_tsplib_clusteiner(path: Path) -> Instance:
    """Carrega instância EUC ou NON_EUC em qualquer Type/size do dataset."""
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in raw_lines if ln.strip()]

    name = _parse_header_value(lines, "NAME") or _parse_header_value(lines, "Name") or path.stem
    n = _parse_int_header(lines, "DIMENSION")
    if n is None:
        raise ValueError(f"{path}: DIMENSION não encontrado")

    idx_coord = _find_section_idx(lines, "NODE_COORD_SECTION")
    idx_w = _find_section_idx(lines, "EDGE_WEIGHT_SECTION")
    idx_gtsp = _find_section_idx(lines, "GTSP_SET_SECTION")
    idx_cluster = _find_section_idx(lines, "CLUSTER_SECTION")

    idx_clusters = idx_gtsp if idx_gtsp is not None else idx_cluster
    if idx_clusters is None:
        raise ValueError(f"{path}: não encontrei GTSP_SET_SECTION nem CLUSTER_SECTION")

    clusters = _parse_clusters_from_section(lines, idx_clusters)

    terminals_set = set()
    for ck in clusters:
        terminals_set.update(ck)
    terminals = sorted(terminals_set)

    cluster_of = [-1] * n
    for k, ck in enumerate(clusters):
        for v in ck:
            if not (0 <= v < n):
                raise ValueError(f"{path}: vértice {v+1} em cluster {k} fora de 1..{n}")
            cluster_of[v] = k

    edges: List[Tuple[int, int, float]] = []

    # NON_EUC (matriz de pesos)
    if idx_w is not None:
        if idx_clusters <= idx_w:
            raise ValueError(f"{path}: seção de clusters aparece antes de EDGE_WEIGHT_SECTION (inesperado)")
        nums = _parse_full_matrix_numbers(lines, idx_w + 1, idx_clusters, n)
        for u in range(n):
            base = u * n
            for v in range(u + 1, n):
                edges.append((u, v, float(nums[base + v])))

        inst = Instance(
            name=name,
            n=n,
            m=len(edges),
            edges=edges,
            terminals=terminals,
            clusters=clusters,
            cluster_of=cluster_of,
            is_euclidean=False,
        )
        inst.validate()
        return inst

    # EUC (coordenadas)
    if idx_coord is None:
        raise ValueError(f"{path}: não encontrei EDGE_WEIGHT_SECTION nem NODE_COORD_SECTION")

    coords = _parse_coords(lines, idx_coord + 1, n)
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v, _tsplib_euc_2d(coords[u], coords[v])))

    inst = Instance(
        name=name,
        n=n,
        m=len(edges),
        edges=edges,
        terminals=terminals,
        clusters=clusters,
        cluster_of=cluster_of,
        is_euclidean=True,
    )
    inst.validate()
    return inst
