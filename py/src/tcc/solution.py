from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


TreeEdge = Tuple[int, int]


@dataclass
class Solution:
    """Representa uma solução lida de um arquivo .sol."""

    instance_name: str          # valor da linha INSTANCE ...
    cost: float                 # valor da linha COST ...
    edges: List[TreeEdge]       # lista de arestas (u, v), 0-based


def parse_solution_file(path: Path) -> Solution:
    """
    Lê um arquivo .sol no formato:

        INSTANCE <nome_da_instancia>
        COST <valor_float>

        EDGES
        u0 v0
        u1 v1
        ...

    Retorna um objeto Solution.
    """
    instance_name = ""
    cost = None
    edges: List[TreeEdge] = []

    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    n_lines = len(lines)

    # Procurar INSTANCE
    while i < n_lines and not lines[i].startswith("INSTANCE"):
        i += 1
    if i == n_lines:
        raise ValueError("Arquivo .sol sem linha INSTANCE")
    parts = lines[i].split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Linha INSTANCE mal formada")
    instance_name = parts[1].strip()
    i += 1

    # Procurar COST
    while i < n_lines and not lines[i].startswith("COST"):
        i += 1
    if i == n_lines:
        raise ValueError("Arquivo .sol sem linha COST")
    parts = lines[i].split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Linha COST mal formada")
    cost = float(parts[1].strip())
    i += 1

    # Procurar EDGES
    while i < n_lines and not lines[i].startswith("EDGES"):
        i += 1
    if i == n_lines:
        raise ValueError("Arquivo .sol sem linha EDGES")
    i += 1  # pula linha "EDGES"

    # Linhas restantes: u v
    for j in range(i, n_lines):
        tokens = lines[j].split()
        if len(tokens) != 2:
            raise ValueError(f"Linha de aresta mal formada: {lines[j]!r}")
        u = int(tokens[0])
        v = int(tokens[1])
        edges.append((u, v))

    if cost is None:
        raise ValueError("COST não definido no arquivo .sol")

    return Solution(instance_name=instance_name, cost=cost, edges=edges)


from pathlib import Path

def write_solution_file(path: Path, sol: Solution) -> None:
    """
    Escreve Solution no formato .sol:

        INSTANCE <nome>
        COST <float>

        EDGES
        u v
        ...
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"INSTANCE {sol.instance_name}")
    lines.append(f"COST {sol.cost:.6f}")
    lines.append("")
    lines.append("EDGES")
    for (u, v) in sol.edges:
        lines.append(f"{u} {v}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")