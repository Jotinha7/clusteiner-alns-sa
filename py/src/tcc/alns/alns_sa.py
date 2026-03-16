from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Optional

from .iterlog import IterationLogger

class _NullLogger:
    """
    Serve para permitir log_path=None sem quebrar o loop.
    """
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    def open(self) -> None:
        pass

    def log(self, row) -> None:
        pass

    def close(self) -> None:
        pass


def rpd_percent(cost: float, bks: float) -> float:
    if bks is None or bks <= 0:
        return 0.0
    return 100.0 * (cost - bks) / bks


def sa_accept(rng: random.Random, curr_cost: float, cand_cost: float, temp: float) -> bool:
    """
    Regra SA:
      - se melhorou: aceita
      - se piorou: aceita com probabilidade exp(-(cand-curr)/T)
    """
    if cand_cost <= curr_cost:
        return True
    if temp <= 1e-12:
        return False

    delta = cand_cost - curr_cost
    p = math.exp(-delta / temp)
    return rng.random() < p


def sa_accept_prob(curr_cost: float, cand_cost: float, temp: float) -> float:
    """A probabilidade usada pelo SA (para log)."""
    if cand_cost <= curr_cost:
        return 1.0
    if temp <= 1e-12:
        return 0.0
    delta = cand_cost - curr_cost
    try:
        return float(math.exp(-delta / temp))
    except OverflowError:
        return 0.0


@dataclass
class AdaptiveConfig:
    enabled: bool = False
    segment_len: int = 200          # a cada quantas iterações atualiza pesos
    reaction: float = 0.20          # rho (quanto "anda" em direção à média)
    w_min: float = 0.05             # evita operador morrer
    # recompensas: novo best > melhorou e aceitou > aceitou piora
    sigma_best: float = 10.0
    sigma_improve: float = 5.0
    sigma_accept: float = 1.0


@dataclass
class SAConfig:
    # Se t0=None e auto_t0=True, o algoritmo estima uma temperatura inicial.
    t0: Optional[float] = None
    alpha: float = 0.995
    mode: str = "geom"               # "geom" | "lam"
    auto_t0: bool = False
    auto_samples: int = 200
    auto_p0: float = 0.80            # prob. desejada de aceitar uma piora "típica" no começo
    # Lam schedule (auto-ajuste do T via taxa de aceitação)
    lam_window: int = 100
    lam_target_accept: float = 0.20
    lam_heat: float = 1.05
    lam_cool: float = 0.95
    # Reaquecimento se estagnar
    reheat_after: int = 1500
    reheat_factor: float = 1.50


class _AdaptiveWeights:
    """Pesos e estatísticas para seleção adaptativa (roulette wheel)."""

    def __init__(self, n: int, *, w0: float = 1.0, w_min: float = 0.05) -> None:
        self.w = [float(w0)] * n
        self.score = [0.0] * n
        self.uses = [0] * n
        self.w_min = float(w_min)

    def pick(self, rng: random.Random) -> int:
        # random.choices lida bem com pesos
        return rng.choices(range(len(self.w)), weights=self.w, k=1)[0]

    def add_reward(self, idx: int, reward: float) -> None:
        self.score[idx] += float(reward)
        self.uses[idx] += 1

    def update(self, reaction: float) -> None:
        rho = float(reaction)
        for i in range(len(self.w)):
            if self.uses[i] > 0:
                avg = self.score[i] / self.uses[i]
                self.w[i] = (1.0 - rho) * self.w[i] + rho * avg
            # evita peso 0
            if self.w[i] < self.w_min:
                self.w[i] = self.w_min
            self.score[i] = 0.0
            self.uses[i] = 0


def _estimate_t0(
    rng: random.Random,
    instance: Any,
    curr_solution: Any,
    curr_cost: float,
    destroy_ops: List[Tuple[str, Callable[[Any, Any, random.Random], Any]]],
    repair_ops: List[Tuple[str, Callable[[Any, Any, random.Random], Any]]],
    cost_fn: Callable[[Any], float],
    feasible_fn: Callable[[Any, Any], bool],
    *,
    samples: int,
    p0: float,
) -> float:
    """Estima T0 automaticamente.

    Ideia (clássica): coletar algumas pioras típicas (Δ>0) e escolher T0 de modo que
    uma piora "mediana" seja aceita com probabilidade p0:

        p0 = exp(-Δ_med / T0)  =>  T0 = -Δ_med / ln(p0)

    Isso tende a estabilizar MUITO o comportamento do SA entre instâncias.
    """
    deltas: List[float] = []
    for _ in range(max(10, samples)):
        dname, destroy = rng.choice(destroy_ops)
        rname, repair = rng.choice(repair_ops)
        ps = destroy(instance, curr_solution, rng)
        cand = repair(instance, ps, rng)
        if not feasible_fn(instance, cand):
            continue
        dc = float(cost_fn(cand) - curr_cost)
        if dc > 1e-12:
            deltas.append(dc)

    if not deltas:
        # escala com o custo
        return max(1e-9, 0.01 * curr_cost)

    deltas.sort()
    med = deltas[len(deltas) // 2]
    p0 = min(max(p0, 1e-4), 0.9999)
    return max(1e-9, -med / math.log(p0))


def run_alns_sa(
    instance: Any,
    instance_id: str,
    build_initial: Callable[[Any], Any],  # retorna Solution
    cost_fn: Callable[[Any], float],
    feasible_fn: Callable[[Any, Any], bool],
    num_edges_fn: Callable[[Any], int],
    destroy_ops: List[Tuple[str, Callable[[Any, Any, random.Random], Any]]],  # (name, fn(inst, sol, rng)->PartialState)
    repair_ops: List[Tuple[str, Callable[[Any, Any, random.Random], Any]]],   # (name, fn(inst, partial, rng)->Solution)
    log_path: str | None,
    bks_cost: Optional[float] = None,
    time_limit_s: float = 2.0,
    max_iters: int = 200,
    max_evals: Optional[int] = None,  # se definido, conta soluções avaliadas (inclui a inicial)
    seed: int = 0,
    # SA
    t0: Optional[float] = None,
    alpha: float = 0.995,
    sa_cfg: Optional[SAConfig] = None,
    # ALNS adaptativo
    adaptive_cfg: Optional[AdaptiveConfig] = None,
    # "Reset" raro (R8) quando estagnar: (name, fn(inst, rng)->Solution)
    reset_op: Optional[Tuple[str, Callable[[Any, random.Random], Any]]] = None,
    reset_min_stagnation: int = 2000,
    reset_prob: float = 0.05,
    reset_cooldown: int = 3000,
) -> Any:

    rng = random.Random(seed)

    if sa_cfg is None:
        sa_cfg = SAConfig(t0=t0, alpha=alpha)
    else:
        # t0/alpha passados direto ainda funcionam (prioridade pro sa_cfg)
        if sa_cfg.t0 is None:
            sa_cfg.t0 = t0
        if sa_cfg.alpha is None:
            sa_cfg.alpha = alpha

    if adaptive_cfg is None:
        adaptive_cfg = AdaptiveConfig(enabled=False)

    do_log = (log_path is not None)

    # logger SEMPRE existe: ou escreve CSV (IterationLogger), ou só mede tempo (_NullLogger)
    logger = IterationLogger(log_path) if do_log else _NullLogger()

    if do_log:
        logger.open()

    # solução inicial
    S = build_initial(instance)
    best = S

    curr_cost = cost_fn(S)
    best_cost = curr_cost

    trace = []
    trace.append(best_cost)

    # se não passar bks, usamos o best_cost inicial como referência (pra ao menos ter um rpd "interno")
    if bks_cost is None:
        bks_cost = best_cost

    # --- SA: escolhe T0 ---
    if sa_cfg.auto_t0 or sa_cfg.t0 is None:
        sa_cfg.t0 = _estimate_t0(
            rng,
            instance,
            S,
            curr_cost,
            destroy_ops,
            repair_ops,
            cost_fn,
            feasible_fn,
            samples=sa_cfg.auto_samples,
            p0=sa_cfg.auto_p0,
        )

    temp = float(sa_cfg.t0)

    feasible0 = feasible_fn(instance, S)
    rpd0 = rpd_percent(curr_cost, bks_cost)

    if do_log:
        logger.log({
            "iter": 0,
            "eval": 1,
            "time_s": logger.elapsed_s(),
            "cost": curr_cost,
            "best_cost": best_cost,
            "rpd": rpd0,
            "delta_rpd": 0.0,
            "accepted": 1,
            "accepted_prob": 1.0,
            "temp": temp,
            "sa_mode": sa_cfg.mode,
            "destroy_op": "none",
            "repair_op": "none",
            "destroy_w": "",
            "repair_w": "",
            "reward": 0.0,
            "segment": 0,
            "no_improve": 0,
            "reset": 0,
            "feasible": int(feasible0),
            "num_edges": num_edges_fn(S),
        })

    prev_rpd = rpd0

    # --- limites ---
    if max_evals is not None:
        # max_evals inclui a solução inicial (eval=1)
        max_iters = max(0, int(max_evals) - 1)

    # --- ALNS adaptativo ---
    dW = _AdaptiveWeights(len(destroy_ops), w_min=adaptive_cfg.w_min)
    rW = _AdaptiveWeights(len(repair_ops), w_min=adaptive_cfg.w_min)
    segment = 0

    # controle de estagnação
    no_improve = 0
    last_reset_it = -10**9
    # lam schedule: janela de aceitação
    lam_acc = 0
    lam_cnt = 0

    it = 0
    while it < max_iters and logger.elapsed_s() < time_limit_s:
        it += 1
        eval_count = it + 1
        segment = (it - 1) // max(1, adaptive_cfg.segment_len) + 1

        # 1) escolha de operadores
        # - Se adaptive_cfg.enabled: roulette wheel por pesos
        # - Caso contrário: uniforme
        if adaptive_cfg.enabled:
            di = dW.pick(rng)
            ri = rW.pick(rng)
            dname, destroy = destroy_ops[di]
            rname, repair = repair_ops[ri]
        else:
            di = ri = -1
            dname, destroy = rng.choice(destroy_ops)
            rname, repair = rng.choice(repair_ops)

        # 1.1) "RESET" raro (R8) quando estagnar
        did_reset = 0
        if reset_op is not None:
            reset_name, reset_fn = reset_op
            if (
                no_improve >= reset_min_stagnation
                and (it - last_reset_it) >= reset_cooldown
                and rng.random() < reset_prob
            ):
                S_cand = reset_fn(instance, rng)
                cand_cost = cost_fn(S_cand)
                cand_feasible = feasible_fn(instance, S_cand)
                dname = "RESET"
                rname = reset_name
                did_reset = 1
            else:
                S_cand = None
                cand_cost = 0.0
                cand_feasible = False
        else:
            S_cand = None
            cand_cost = 0.0
            cand_feasible = False

        # 2) gera candidato (normal) se não foi reset
        if not did_reset:
            partial = destroy(instance, S, rng)
            S_cand = repair(instance, partial, rng)
            cand_cost = cost_fn(S_cand)
            cand_feasible = feasible_fn(instance, S_cand)

        # 3) aceitação SA
        accepted = 0
        accepted_prob = sa_accept_prob(curr_cost, cand_cost, temp)
        curr_before = curr_cost
        if cand_feasible and sa_accept(rng, curr_cost, cand_cost, temp):
            S = S_cand
            curr_cost = cand_cost
            accepted = 1

        # se foi reset, a ideia é "sair do buraco": aceita sempre (se factível)
        if did_reset and cand_feasible:
            S = S_cand
            curr_cost = cand_cost
            accepted = 1
            # reaquecimento pós-reset para explorar
            temp = max(temp, float(sa_cfg.t0))
            last_reset_it = it

        # 4) atualiza best
        improved_best = 0
        if curr_cost < best_cost:
            best = S
            best_cost = curr_cost
            improved_best = 1
            no_improve = 0
        else:
            no_improve += 1

        # 5) métricas e log
        rpd = rpd_percent(curr_cost, bks_cost)
        delta_rpd = rpd - prev_rpd
        prev_rpd = rpd

        # 5.1) reward ALNS
        reward = 0.0
        if cand_feasible:
            if improved_best:
                reward = adaptive_cfg.sigma_best
            elif accepted and cand_cost < curr_before:
                reward = adaptive_cfg.sigma_improve
            elif accepted:
                reward = adaptive_cfg.sigma_accept

        if adaptive_cfg.enabled and di >= 0 and ri >= 0:
            dW.add_reward(di, reward)
            rW.add_reward(ri, reward)

        if do_log:
            logger.log({
                "iter": it,
                "eval": eval_count,
                "time_s": logger.elapsed_s(),
                "cost": curr_cost,
                "best_cost": best_cost,
                "rpd": rpd,
                "delta_rpd": delta_rpd,
                "accepted": accepted,
                "accepted_prob": accepted_prob,
                "temp": temp,
                "sa_mode": sa_cfg.mode,
                "destroy_op": dname,
                "repair_op": rname,
                "destroy_w": ("" if not adaptive_cfg.enabled or di < 0 else dW.w[di]),
                "repair_w": ("" if not adaptive_cfg.enabled or ri < 0 else rW.w[ri]),
                "reward": reward,
                "segment": segment,
                "no_improve": no_improve,
                "reset": did_reset,
                "feasible": int(feasible_fn(instance, S)),
                "num_edges": num_edges_fn(S),
            })

        # 6) SA: atualização de temperatura
        if sa_cfg.mode == "lam":
            lam_cnt += 1
            lam_acc += 1 if accepted else 0
            if lam_cnt >= max(1, sa_cfg.lam_window):
                rate = lam_acc / lam_cnt
                # se aceita demais => esfria; se aceita de menos => aquece
                if rate > sa_cfg.lam_target_accept:
                    temp *= sa_cfg.lam_cool
                else:
                    temp *= sa_cfg.lam_heat
                lam_cnt = 0
                lam_acc = 0
        else:
            temp *= float(sa_cfg.alpha)

        # reaquecimento se ficar muito tempo sem melhorar o best
        if sa_cfg.reheat_after > 0 and no_improve > 0 and (no_improve % sa_cfg.reheat_after == 0):
            temp = min(float(sa_cfg.t0), temp * float(sa_cfg.reheat_factor))

        # 7) ALNS adaptativo: atualização de pesos por segmento
        if adaptive_cfg.enabled and (it % max(1, adaptive_cfg.segment_len) == 0):
            dW.update(adaptive_cfg.reaction)
            rW.update(adaptive_cfg.reaction)

        if it % 50 == 0:
            trace.append(best_cost)

    if it % 50 != 0:
        trace.append(best_cost)

    if do_log:
        logger.close()
    return best, trace