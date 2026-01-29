import heapq
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from Algorithms.BIDijkstra import bidirectional_dijkstra_edge_alternating, BiDijkstraResult

def reconstruct_path(parent: List[int], s: int, t: int) -> List[int]:
    if s == t:
        return [s]
    if parent[t] == -1:
        return []
    cur = t
    out = []
    while cur != -1:
        out.append(cur)
        if cur == s:
            break
        cur = parent[cur]
    out.reverse()
    return out if out and out[0] == s else []
    
# ----------------------------
# CHEAP_BID implementation
# ----------------------------

'''
@dataclass
class CheapBidResult:
    dist: float
    path: List[int]
    used_shortcut: bool
    # fallback stats (if used_shortcut=False)
    fallback: Optional[BiDijkstraResult]
    time_sec: float
'''

@dataclass
class CheapBidResult:
    dist: float
    path: List[int]
    used_shortcut: bool
    fallback: Optional[object]  # BiDijkstraResult if you want, but keeping generic
    time_sec: float
    # optional: counts to sanity-check scaling
    scanned_s_edges: int
    scanned_v_edges: int
    scanned_t_edges: int


def cheap_bid(adj: List[List[Tuple[int, float]]], s: int, t: int):
    """
    CHEAP_BID that stays local (O(deg(s) + deg(t)) on G_D),
    and falls back to bidirectional_dijkstra_edge_alternating if it can't certify (s,m,t).
    """
    start = time.perf_counter()
    INF = float("inf")

    m: Optional[int] = None
    t_pointer: Optional[int] = None

    b_s = INF
    ell_s = INF

    b_t = INF
    ell_t = INF

    # store the candidate weights for the (s,m,t) path
    w_sm = INF
    w_mt = INF

    scanned_s_edges = 0
    scanned_v_edges = 0
    scanned_t_edges = 0

    # Scan full neighborhood of s
    for v, w_sv in adj[s]:
        scanned_s_edges += 1

        # lower bound leaving s
        if w_sv < b_s:
            b_s = w_sv

        # abort cheap check if s adjacent to t
        if v == t:
            m = None
            t_pointer = None
            break

        # Scan neighbors of v, but stop once |N_v| > 2
        Nv = set()
        w_vt = None

        for u, w_vu in adj[v]:
            scanned_v_edges += 1

            Nv.add(u)
            if u == t:
                t_pointer = t
                ell_s = w_sv          # weight(s,v)
                ell_t = w_vu          # weight(v,t)
                w_vt = w_vu

            if len(Nv) > 2:
                break

        # Detect m with neighborhood exactly {s,t}
        # For m, this loop will see exactly two neighbors and never break early.
        if Nv == {s, t} and w_vt is not None:
            m = v
            w_sm = w_sv
            w_mt = w_vt

    # If we ever saw a neighbor-of-s connected to t, scan t's neighborhood for b_t
    if t_pointer is not None:
        for v, w_tv in adj[t]:
            scanned_t_edges += 1
            if w_tv < b_t:
                b_t = w_tv

    # Certificate check
    if m is not None and t_pointer is not None and (b_s + b_t == ell_s + ell_t):
        end = time.perf_counter()
        return CheapBidResult(
            dist=w_sm + w_mt,
            path=[s, m, t],
            used_shortcut=True,
            fallback=None,
            time_sec=end - start,
            scanned_s_edges=scanned_s_edges,
            scanned_v_edges=scanned_v_edges,
            scanned_t_edges=scanned_t_edges,
        )

    # fallback
    fallback = bidirectional_dijkstra_edge_alternating(adj, s, t)
    end = time.perf_counter()
    return CheapBidResult(
        dist=fallback.dist,
        path=fallback.path,
        used_shortcut=False,
        fallback=fallback,
        time_sec=end - start,
        scanned_s_edges=scanned_s_edges,
        scanned_v_edges=scanned_v_edges,
        scanned_t_edges=scanned_t_edges,
    )