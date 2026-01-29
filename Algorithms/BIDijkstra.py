import random
import heapq
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# ----------------------------
# Bidirectional Dijkstra with *edge-relaxation alternation*
# ----------------------------

@dataclass
class BiDijkstraResult:
    dist: float
    path: List[int]
    pops_f: int
    pops_b: int
    relaxations_f: int
    relaxations_b: int
    time_sec: float
    mu_updates: int


class EdgeRelaxingDijkstraStepper:
    """
    Dijkstra execution stepped by ONE edge relaxation at a time.
    Uses randomized tie-breaking among equal-distance candidates to avoid
    the 'smallest node id always wins' artifact.
    """

    def __init__(self, adj: List[List[Tuple[int, float]]], source: int, seed: Optional[int] = None):
        self.adj = adj
        self.n = len(adj)
        self.INF = float("inf")

        self.dist = [self.INF] * self.n
        self.parent = [-1] * self.n
        self.closed = [False] * self.n

        self.dist[source] = 0.0

        # Random tie-breaker RNG
        self.rng = random.Random(seed)

        # Heap entries: (dist, tie, node)
        self.pq: List[Tuple[float, float, int]] = [(0.0, self.rng.random(), source)]

        # current expansion state
        self.active_u: Optional[int] = None
        self.active_du: float = self.INF
        self.active_i: int = 0

        self.last_closed: Optional[int] = None

        # stats
        self.pops = 0
        self.relaxations = 0
        self.exhausted = False

    def _acquire_next_active(self) -> bool:
        while self.pq:
            du, _tie, u = heapq.heappop(self.pq)
            self.pops += 1
            if self.closed[u]:
                continue
            self.closed[u] = True
            self.last_closed = u
            self.active_u = u
            self.active_du = du
            self.active_i = 0
            return True

        self.exhausted = True
        self.active_u = None
        return False

    def step_one_edge_relaxation(self) -> Optional[Tuple[int, int, float]]:
        if self.exhausted:
            return None

        if self.active_u is None:
            if not self._acquire_next_active():
                return None

        u = self.active_u
        du = self.active_du

        while u is not None and self.active_i >= len(self.adj[u]):
            self.active_u = None
            if not self._acquire_next_active():
                return None
            u = self.active_u
            du = self.active_du

        assert u is not None
        v, w = self.adj[u][self.active_i]
        self.active_i += 1

        self.relaxations += 1

        if not self.closed[v]:
            nd = du + w
            if nd < self.dist[v]:
                self.dist[v] = nd
                self.parent[v] = u
                heapq.heappush(self.pq, (nd, self.rng.random(), v))

        return (u, v, w)


def bidirectional_dijkstra_edge_alternating(adj: List[List[Tuple[int, float]]], s: int, t: int) -> BiDijkstraResult:
    """
    Bidirectional Dijkstra that:
      (1) alternates ONE edge relaxation forward, then ONE backward, etc.
      (2) stops when dist_f[last_closed_f] + dist_b[last_closed_b] >= mu
          where mu is best s-t path length found so far (updated on cross edges).
    Reconstruction uses the stored "middle edge" (u,v) discovered by cross condition.
    """
    start = time.perf_counter()

    # This only makes sense if graph is undircted
    f = EdgeRelaxingDijkstraStepper(adj, s)
    b = EdgeRelaxingDijkstraStepper(adj, t)

    mu = float("inf")
    mu_updates = 0
    e_mid: Optional[Tuple[int, int]] = None  # (u, v) oriented as in forward scan

    # These are the "u_s" and "u_t" in your pseudocode: most recently closed nodes
    u_s = s
    u_t = t

    # Alternate: forward edge relaxation, then backward, repeating
    turn_forward = True

    # Run until termination condition fires or one side exhausts.
    while True:
        if f.exhausted or b.exhausted:
            break

        if turn_forward:
            # A single edge relaxation in forward
            edge = f.step_one_edge_relaxation()
            if edge is None:
                break

            # update u_s if a new node was closed during that step
            if f.last_closed is not None:
                u_s = f.last_closed

            # stopping condition checked when u_s/u_t exist (they do)
            # (your condition: d(s,u_s) + d(u_t,t) >= mu terminates whole algorithm)
            if f.dist[u_s] + b.dist[u_t] >= mu:
                break

            u, v, w = edge

            # cross check: if v is closed in backward, update mu with path via edge u-v
            # note: even if v got relaxed but isn't closed in forward, we only use the
            # exact form from your pseudocode: d_f[u] + w + d_b[v]
            if b.closed[v]:
                cand = f.dist[u] + w + b.dist[v]
                if cand < mu:
                    mu = cand
                    mu_updates += 1
                    e_mid = (u, v)

        else:
            # A single edge relaxation in backward
            edge = b.step_one_edge_relaxation()
            if edge is None:
                break

            if b.last_closed is not None:
                u_t = b.last_closed

            if f.dist[u_s] + b.dist[u_t] >= mu:
                break

            u, v, w = edge

            # Symmetric cross update, but now the "middle edge" for the final path
            # should still be stored as (u_fwd, v_fwd). In backward scan, the edge
            # is (u_back, v_back). Since the graph is undirected here, we can treat
            # it the same way, but we must orient it to match:
            # forward part ends at v_back and backward part ends at u_back,
            # so the mid edge could be (v_back, u_back).
            if f.closed[v]:
                cand = b.dist[u] + w + f.dist[v]
                if cand < mu:
                    mu = cand
                    mu_updates += 1
                    e_mid = (v, u)  # orient as forward-endpoint then backward-endpoint

        turn_forward = not turn_forward

    end = time.perf_counter()

    if mu == float("inf") or e_mid is None:
        return BiDijkstraResult(
            dist=float("inf"),
            path=[],
            pops_f=f.pops,
            pops_b=b.pops,
            relaxations_f=f.relaxations,
            relaxations_b=b.relaxations,
            time_sec=end - start,
            mu_updates=mu_updates,
        )

    u, v = e_mid
    # forward path: s -> ... -> u using forward parents
    path_s_u = reconstruct_path(f.parent, s, u)
    # backward path: t -> ... -> v using backward parents (but that's from t outward),
    # we want v -> ... -> t, so reconstruct v -> t using b.parent which points "toward t"
    # because b ran from t as its source.
    path_t_v = reconstruct_path(b.parent, t, v)  # gives t->...->v
    path_v_t = list(reversed(path_t_v))          # v->...->t

    if not path_s_u or not path_v_t:
        # If reconstruction fails for some reason (it shouldn't, but humans love surprises)
        final_path = []
    else:
        # concatenate: s..u + [v]..t with middle edge (u,v)
        final_path = path_s_u + [v] + path_v_t[1:]  # avoid duplicating v

    return BiDijkstraResult(
        dist=mu,
        path=final_path,
        pops_f=f.pops,
        pops_b=b.pops,
        relaxations_f=f.relaxations,
        relaxations_b=b.relaxations,
        time_sec=end - start,
        mu_updates=mu_updates,
    )

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