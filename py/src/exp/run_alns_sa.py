from __future__ import annotations

import argparse
import time
import json 
from pathlib import Path
from tcc.solution import write_solution_file

from tcc.alns import (
    run_alns_sa,
    SAConfig,
    AdaptiveConfig,
    destroy_remove_k_global_edges,
    destroy_remove_worst_k_global_edges,
    destroy_remove_global_path_segment,
    destroy_remove_global_steiner_star,
    destroy_disconnect_cluster,
    destroy_break_local_tree,
    repair_r1_dijkstra,
    repair_r1_dijkstra_topL,
    repair_r3_mst_components,
    repair_r4_steiner_hub,
    repair_r5_two_hubs_direct,
    repair_r6_component_bridge_chain,
    repair_r7_global_sph,
    ensure_local_rebuilt,
    build_initial_solution,
    build_reset_solution,
)
from tcc.tsplib_loader import load_tsplib_clusteiner
from tcc.verify import verify_solution
from tcc.solution import Solution

import random

ENABLE_LOG = False  # mude para True se quiser voltar a gerar logs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--time", type=float, default=2.0)
    # Para comparação justa com o paper: eles usam 25_000 avaliações (solutions evaluated).
    # Aqui, cada iteração avalia 1 solução candidata + 1 solução inicial.
    ap.add_argument("--evals", type=int, default=25000, help="Número de soluções avaliadas (inclui a inicial)")
    ap.add_argument("--seed", type=int, default=0)

    # SA
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.995)
    ap.add_argument("--sa_auto_t0", action="store_true", help="Estima T0 automaticamente por amostragem")
    ap.add_argument("--sa_mode", choices=["geom", "lam"], default="geom")

    # D1
    ap.add_argument("--k", type=int, default=2)

    # Top-L
    ap.add_argument("--topL", type=int, default=5, help="Se >0, habilita R1_topL com L=topL")

    # ALNS adaptativo
    ap.add_argument("--adaptive", action="store_true", help="Habilita pesos adaptativos")
    ap.add_argument("--seg", type=int, default=200, help="Tamanho do segmento para atualizar pesos")
    ap.add_argument("--rho", type=float, default=0.20, help="Reação (0.1~0.3). Maior => adapta mais rápido")

    # Reset raro (R8) quando estagnar
    ap.add_argument("--enable_reset", action="store_true", help="Habilita reset raro (R8 em NON_EUC / MST em EUC)")
    ap.add_argument("--reset_stag", type=int, default=3000, help="Estagnação mínima (iters sem melhorar best)")
    ap.add_argument("--reset_prob", type=float, default=0.02, help="Probabilidade de reset ao atingir estagnação")
    ap.add_argument("--reset_cooldown", type=int, default=5000, help="Cooldown mínimo entre resets")

    args = ap.parse_args()

    instance_path = Path(args.instance)
    instance_id = instance_path.stem

    repo_root = Path(__file__).resolve().parents[3]

    if ENABLE_LOG:
        out_dir = repo_root / "experiments" / "results" / "week3_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(out_dir / f"{instance_id}_seed{args.seed}.csv")
    else:
        log_path = None

    inst = load_tsplib_clusteiner(instance_path)

    def build_initial(instance):
        return build_initial_solution(instance, rng=random.Random(args.seed))

    def cost_fn(sol: Solution) -> float:
        return float(sol.cost)

    def feasible_fn(instance, sol: Solution) -> bool:
        return bool(verify_solution(instance, sol).feasible)

    def num_edges_fn(sol: Solution) -> int:
        return len(sol.edges)

    def D1(instance, sol, rng):
        return destroy_remove_k_global_edges(instance, sol, rng, k=args.k)

    def D2(instance, sol, rng):
        return destroy_disconnect_cluster(instance, sol, rng)

    def D3(instance, sol, rng):
        return destroy_break_local_tree(instance, sol, rng)

    def D4(instance, sol, rng):
        return destroy_remove_worst_k_global_edges(instance, sol, rng, k=args.k)

    def D5(instance, sol, rng):
        return destroy_remove_global_path_segment(instance, sol, rng, max_remove=max(2, args.k))

    def D6(instance, sol, rng):
        return destroy_remove_global_steiner_star(instance, sol, rng, max_remove=6)

    destroys = [
        ("D1_rm_k", D1),
        ("D2_disc_cluster", D2),
        ("D3_break_local", D3),
        ("D4_worst_k", D4),
        ("D5_path_seg", D5),
        ("D6_steiner_star", D6),
    ]

    def WRAP(repair_fn):
        """Se o destroy foi D3, reconstrói a local tree antes do repair global."""
        def _wrapped(inst, ps, rng):
            ps2 = ensure_local_rebuilt(inst, ps, rng, max_steiners=15)
            return repair_fn(inst, ps2, rng)
        return _wrapped

    repairs = [
        ("R1_dijkstra", WRAP(repair_r1_dijkstra)),
        ("R3_comp_mst", WRAP(repair_r3_mst_components)),
        ("R7_global_sph", WRAP(repair_r7_global_sph)),
        ("R4_steiner_hub", WRAP(lambda inst, ps, rng: repair_r4_steiner_hub(inst, ps, rng, max_candidates=25))),
        ("R5_two_hubs_direct", WRAP(lambda inst, ps, rng: repair_r5_two_hubs_direct(inst, ps, rng, max_candidates=25))),
        ("R6_bridge_chain", WRAP(lambda inst, ps, rng: repair_r6_component_bridge_chain(inst, ps, rng, max_candidates=25))),
    ]

    if args.topL and args.topL > 0:
        def R1T(instance, partial, rng):
            ps2 = ensure_local_rebuilt(instance, partial, rng, max_steiners=15)
            return repair_r1_dijkstra_topL(instance, ps2, rng, L=args.topL)
        repairs.insert(0, ("R1_topL", R1T))

    sa_cfg = SAConfig(
        t0=args.t0,
        alpha=args.alpha,
        mode=args.sa_mode,
        auto_t0=bool(args.sa_auto_t0),
    )

    adaptive_cfg = AdaptiveConfig(
        enabled=bool(args.adaptive),
        segment_len=int(args.seg),
        reaction=float(args.rho),
    )

    reset_op = None
    if args.enable_reset:
        def _reset(inst, rng):
            return build_reset_solution(inst, rng)
        reset_op = ("R8_reset", _reset)

    start_time = time.perf_counter()

    best, trace = run_alns_sa(
        instance=inst,
        instance_id=instance_id,
        build_initial=build_initial,
        cost_fn=cost_fn,
        feasible_fn=feasible_fn,
        num_edges_fn=num_edges_fn,
        destroy_ops=destroys,
        repair_ops=repairs,
        log_path=log_path,
        bks_cost=None,
        time_limit_s=args.time,
        max_evals=args.evals,
        seed=args.seed,
        sa_cfg=sa_cfg,
        adaptive_cfg=adaptive_cfg,
        reset_op=reset_op,
        reset_min_stagnation=int(args.reset_stag),
        reset_prob=float(args.reset_prob),
        reset_cooldown=int(args.reset_cooldown),
    )

    elapsed_time = time.perf_counter() - start_time

    ok = verify_solution(inst, best).feasible

    output_data = {
        "instance": instance_id,
        "seed": args.seed,
        "best_cost": float(best.cost),
        "feasible": ok,
        "time_s": elapsed_time,
        "trace": trace  # Histórico de custos para plotar o gráfico de convergência
    }

    json_output = json.dumps(output_data)
    
    if ENABLE_LOG:
        print(f"[OK] log={log_path} best_cost={best.cost:.6f} feasible={ok}", flush=True)
    else:
        print(f"[OK] best_cost={best.cost:.6f} feasible={ok}", flush=True)

    # Salvar arquivo .sol dentro de experiments/solutions
    # sol_dir = repo_root / "experiments" / "solutions"
    # sol_path = sol_dir / f"{instance_id}_seed{args.seed}_best.sol"
    # write_solution_file(sol_path, best)
    
   # print(f"---RESULT_JSON---{json_output}---RESULT_JSON---", flush=True)


if __name__ == "__main__":
    main()