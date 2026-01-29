import random
import heapq
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from Graph_Generation.generator import generate_erdos_renyi_weighted, build_G_D, has_path
from Algorithms.Dijkstra import dijkstra_early_stop
from Algorithms.BIDijkstra import bidirectional_dijkstra_edge_alternating
from Algorithms.CHEAP_BID import cheap_bid

# ----------------------------
# Empirical evaluation harness
# ----------------------------

def run_experiment(
    n_values=(200, 500, 1000),
    p_values=(0.01, 0.02, 0.05),
    trials_per_setting=5,
    weight_range=(1.0, 10.0),
    seed=12345,
) -> List[Dict[str, Any]]:
    """
    Runs both algorithms over ER graphs for each (n,p), several trials.
    Picks random (s,t) per trial and retries graph generation until s-t is connected.
    Returns list of dict rows.
    """
    base_rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    wmin, wmax = weight_range

    for n in n_values:
        for p in p_values:
            for trial in range(trials_per_setting):
                # Generate a connected-enough graph for a random s,t
                # (because comparing "infinite distance" is thrilling for nobody)
                attempt = 0
                while True:
                    attempt += 1
                    g_seed = base_rng.randrange(10**9)
                    adj = generate_erdos_renyi_weighted(n, p, wmin, wmax, seed=g_seed)
                    s = base_rng.randrange(n)
                    t = base_rng.randrange(n)
                    while t == s:
                        t = base_rng.randrange(n)
                    if has_path(adj, s, t):
                        break
                    if attempt >= 50:
                        # give up and record as disconnected
                        adj = None
                        break

                if adj is None:
                    rows.append({
                        "n": n, "p": p, "trial": trial, "s": None, "t": None,
                        "status": "disconnected",
                    })
                    continue

                # Run algorithms
                res1 = dijkstra_early_stop(adj, s, t)
                res2 = bidirectional_dijkstra_edge_alternating(adj, s, t)

                # sanity: distances should match (within floating tolerance)
                same = (math.isinf(res1.dist) and math.isinf(res2.dist)) or (
                    abs(res1.dist - res2.dist) <= 1e-9 * max(1.0, abs(res1.dist), abs(res2.dist))
                )

                rows.append({
                    "n": n, "p": p, "trial": trial, "s": s, "t": t,
                    "dist_dijkstra": res1.dist,
                    "dist_bidir": res2.dist,
                    "dist_match": same,
                    "time_dijkstra_sec": res1.time_sec,
                    "time_bidir_sec": res2.time_sec,
                    "pops_dijkstra": res1.pops,
                    "relax_dijkstra": res1.relaxations,
                    "pops_bidir_f": res2.pops_f,
                    "pops_bidir_b": res2.pops_b,
                    "relax_bidir_f": res2.relaxations_f,
                    "relax_bidir_b": res2.relaxations_b,
                    "mu_updates": res2.mu_updates,
                    "pathlen_dijkstra_nodes": len(res1.path),
                    "pathlen_bidir_nodes": len(res2.path),
                    "status": "ok" if same else "mismatch",
                })

    return rows


def summarize(rows: List[Dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") == "mismatch"]
    disc = [r for r in rows if r.get("status") == "disconnected"]

    print(f"Total rows: {len(rows)} | ok: {len(ok)} | mismatches: {len(bad)} | disconnected: {len(disc)}")
    if bad:
        print("Example mismatch row:")
        print(bad[0])

    # Aggregate by (n,p)
    by = {}
    for r in ok:
        key = (r["n"], r["p"])
        by.setdefault(key, []).append(r)

    print("\nAverages (only ok runs):")
    for (n, p), group in sorted(by.items()):
        avg = lambda k: sum(x[k] for x in group) / len(group)
        print(
            f"n={n:4d} p={p:0.3f} | "
            f"time dijk={avg('time_dijkstra_sec'):.6f}s "
            f"time bi={avg('time_bidir_sec'):.6f}s | "
            f"relax dijk={avg('relax_dijkstra'):.1f} "
            f"relax bi={(avg('relax_bidir_f')+avg('relax_bidir_b')):.1f}"
        )

def run_experiment_GD(
    D=(10, 20, 30),
    uniform_weight=1.0,
    trials_per_setting=5,
    other_weight_range=(1.0, 10.0),
    seed=12345,
) -> List[Dict[str, Any]]:
    """
    Compare CHEAP_BID vs bidirectional edge-alternating Dijkstra on G_D graphs.

    Returns a list of row dicts suitable for CSV / pandas.

    Assumes the following are already implemented in scope:
      - build_G_D(D, w_incident=..., other_w_min=..., other_w_max=..., seed=..., return_labels=...)
      - cheap_bid(adj, s, t) -> CheapBidResult (with fields dist, path, used_shortcut, fallback, time_sec)
      - bidirectional_dijkstra_edge_alternating(adj, s, t) -> BiDijkstraResult
    """
    base_rng = random.Random(seed)
    other_min, other_max = other_weight_range

    rows: List[Dict[str, Any]] = []

    for Dval in D:
        for trial in range(trials_per_setting):
            # Use a fresh seed each trial so neighbor order + "other" weights vary
            g_seed = base_rng.randrange(10**9)

            adj, s, t = build_G_D(
                D=Dval,
                w_incident=float(uniform_weight),
                other_w_min=float(other_min),
                other_w_max=float(other_max),
                seed=g_seed,
                return_labels=False,
            )

            # Run CHEAP_BID
            cheap_res = cheap_bid(adj, s, t)

            # Run BID_DIJ_AUTH (always, for fair apples-to-apples timing and stats)
            bidir_res = bidirectional_dijkstra_edge_alternating(adj, s, t)

            # Sanity: CHEAP result should match bidir (it either returns (s,m,t) or bidir result)
            # Use tolerance because float weights exist in "other" edges.
            if math.isinf(cheap_res.dist) and math.isinf(bidir_res.dist):
                dist_match = True
            else:
                dist_match = abs(cheap_res.dist - bidir_res.dist) <= 1e-9 * max(1.0, abs(cheap_res.dist), abs(bidir_res.dist))

            rows.append({
                "D": Dval,
                "trial": trial,
                "uniform_weight": float(uniform_weight),
                "other_w_min": float(other_min),
                "other_w_max": float(other_max),

                # outputs
                "dist_cheap": cheap_res.dist,
                "dist_bidir": bidir_res.dist,
                "dist_match": dist_match,

                # shortcut behavior
                "cheap_used_shortcut": cheap_res.used_shortcut,

                # timing
                "time_cheap_sec": cheap_res.time_sec,
                "time_bidir_sec": bidir_res.time_sec,

                # bidir stats (always available)
                "bidir_pops_f": bidir_res.pops_f,
                "bidir_pops_b": bidir_res.pops_b,
                "bidir_relax_f": bidir_res.relaxations_f,
                "bidir_relax_b": bidir_res.relaxations_b,
                "bidir_mu_updates": bidir_res.mu_updates,

                # cheap fallback stats (only meaningful if it didn't shortcut)
                "cheap_fallback_time_sec": (cheap_res.fallback.time_sec if cheap_res.fallback is not None else 0.0),
                "cheap_fallback_pops_f": (cheap_res.fallback.pops_f if cheap_res.fallback is not None else 0),
                "cheap_fallback_pops_b": (cheap_res.fallback.pops_b if cheap_res.fallback is not None else 0),
                "cheap_fallback_relax_f": (cheap_res.fallback.relaxations_f if cheap_res.fallback is not None else 0),
                "cheap_fallback_relax_b": (cheap_res.fallback.relaxations_b if cheap_res.fallback is not None else 0),
                "cheap_fallback_mu_updates": (cheap_res.fallback.mu_updates if cheap_res.fallback is not None else 0),

                # path lengths (nodes count)
                "pathlen_cheap_nodes": len(cheap_res.path),
                "pathlen_bidir_nodes": len(bidir_res.path),

                "status": "ok" if dist_match else "mismatch",
            })

    return rows


def summarize_GD(rows: List[Dict[str, Any]]) -> None:
    """
    Print a basic summary like before:
      - counts ok/mismatch
      - shortcut rate by D
      - average times by D
      - average bidir work (relaxations) by D
    """
    total = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") == "mismatch"]

    print(f"Total runs: {total} | ok: {len(ok)} | mismatches: {len(bad)}")
    if bad:
        print("Example mismatch row (first):")
        print(bad[0])

    # Group by D
    byD: Dict[int, List[Dict[str, Any]]] = {}
    for r in ok:
        byD.setdefault(r["D"], []).append(r)

    def avg(group, key):
        return sum(x[key] for x in group) / len(group) if group else float("nan")

    print("\nPer-D summary (only ok runs):")
    for Dval in sorted(byD.keys()):
        g = byD[Dval]
        shortcut_rate = sum(1 for x in g if x["cheap_used_shortcut"]) / len(g)

        avg_time_cheap = avg(g, "time_cheap_sec")
        avg_time_bidir = avg(g, "time_bidir_sec")

        avg_relax_bidir = avg(g, "bidir_relax_f") + avg(g, "bidir_relax_b")

        # When CHEAP_BID doesn't shortcut, its fallback is basically bidir plus overhead.
        # This shows whether cheap is "free enough" when it fails.
        nons = [x for x in g if not x["cheap_used_shortcut"]]
        avg_time_cheap_nons = avg(nons, "time_cheap_sec") if nons else float("nan")

        print(
            f"D={Dval:4d} | "
            f"shortcut_rate={shortcut_rate:0.2%} | "
            f"time cheap={avg_time_cheap:.6f}s "
            f"(non-shortcut avg={avg_time_cheap_nons:.6f}s) | "
            f"time bidir={avg_time_bidir:.6f}s | "
            f"bidir relax(avg)={avg_relax_bidir:.1f}"
        )