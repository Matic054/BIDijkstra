import random
import heapq
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# ----------------------------
# Dijkstra (single-source) with early stop
# ----------------------------

@dataclass
class DijkstraResult:
    dist: float
    path: List[int]
    pops: int
    relaxations: int
    time_sec: float


def dijkstra_early_stop(adj: List[List[Tuple[int, float]]], s: int, t: int) -> DijkstraResult:
    """
    Standard Dijkstra, but terminates as soon as it closes a vertex u with
    dist[u] >= dist[t] (current best estimate to t).
    """
    n = len(adj)
    INF = float("inf")
    dist = [INF] * n
    parent = [-1] * n
    closed = [False] * n

    dist[s] = 0.0
    pq = [(0.0, s)]
    pops = 0
    relaxations = 0

    start = time.perf_counter()

    while pq:
        du, u = heapq.heappop(pq)
        pops += 1
        if closed[u]:
            continue
        closed[u] = True

        # Early stop condition:
        # when we close u and dist[u] >= dist[t].
        if dist[t] < INF and dist[u] >= dist[t]:
            break

        for v, w in adj[u]:
            relaxations += 1
            if closed[v]:
                continue
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    end = time.perf_counter()

    path = reconstruct_path(parent, s, t) if dist[t] < INF else []
    return DijkstraResult(dist=dist[t], path=path, pops=pops, relaxations=relaxations, time_sec=end - start)


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