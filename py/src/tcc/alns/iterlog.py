from __future__ import annotations
import csv
import time
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class IterationLogger:
    csv_path: str
    _t0: float = None
    _f: Any = None
    _w: Any = None

    def __post_init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    def open(self) -> None:
        if self._f is not None:
            return
        self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=[
            "iter",          # iteração (0 = solução inicial)
            "eval",          # contador de soluções avaliadas (inclui a inicial)
            "time_s",
            "cost",
            "best_cost",
            "rpd",
            "delta_rpd",
            "accepted",
            "accepted_prob", # prob. de aceitação usada quando o movimento é pior
            "temp",
            "sa_mode",
            "destroy_op",
            "repair_op",
            "destroy_w",     # peso atual do destroy (ALNS adaptativo)
            "repair_w",      # peso atual do repair  (ALNS adaptativo)
            "reward",        # recompensa atribuída aos operadores
            "segment",       # bloco/segmento do ALNS
            "no_improve",    # iterações desde a última melhora do best
            "reset",         # 1 se foi "reset" (R8), 0 caso contrário
            "feasible",
            "num_edges",
        ])
        self._w.writeheader()
        self._f.flush()

    def log(self, row: Dict[str, Any]) -> None:
        if self._f is None:
            self.open()
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
            self._w = None