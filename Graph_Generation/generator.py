import random
import heapq
import math
from typing import List, Tuple, Optional, Dict, Any

def generate_erdos_renyi_weighted(
    n: int,
    p: float,
    w_min: float = 1.0,
    w_max: float = 10.0,
    seed: Optional[int] = None,
) -> List[List[Tuple[int, float]]]:
    """
    Undirected ER(n,p) graph with positive weights on edges.
    Returns adjacency list: adj[u] = [(v, w), ...]
    """
    rng = random.Random(seed)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                w = rng.uniform(w_min, w_max)
                adj[u].append((v, w))
                adj[v].append((u, w))
    return adj

def build_G_D(
    D: int,
    w_incident: float = 1.0,          # uniform weight for ALL edges incident to s OR t
    other_w_min: float = 1.0,
    other_w_max: float = 10.0,
    seed: Optional[int] = None,
    return_labels: bool = False,
) -> Tuple[List[List[Tuple[int, float]]], int, int] | Tuple[List[List[Tuple[int, float]]], int, int, Dict[int, str]]:
    """
    Construct undirected graph family G_D:

      Nodes: s, t, m, s1..sD, t1..tD  => total 3 + 2D nodes
      Edges:
        - (s,m), (m,t)
        - clique on {s, s1..sD}
        - clique on {t, t1..tD}
      Weights:
        - every edge incident to s OR t has weight w_incident (uniform, positive)
        - all other edges have arbitrary positive weights (random uniform in [other_w_min, other_w_max])

    Returns:
      adj, s, t (and optionally labels if return_labels=True)

    Stable indexing:
      0: s
      1: t
      2: m
      3..(3+D-1): s1..sD
      (3+D)..(3+2D-1): t1..tD
    """
    if D < 1:
        raise ValueError("D must be >= 1")
    if w_incident <= 0:
        raise ValueError("w_incident must be positive")
    if other_w_min <= 0 or other_w_max <= 0 or other_w_min > other_w_max:
        raise ValueError("other_w_min/other_w_max must be positive and other_w_min <= other_w_max")

    rng = random.Random(seed)

    n = 3 + 2 * D
    s = 0
    t = 1
    m = 2
    s_nodes = list(range(3, 3 + D))               # s1..sD
    t_nodes = list(range(3 + D, 3 + 2 * D))       # t1..tD

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    def add_undirected_edge(u: int, v: int, w: float) -> None:
        # undirected: store both directions
        adj[u].append((v, w))
        adj[v].append((u, w))

    def edge_weight(u: int, v: int) -> float:
        # Uniform weight if incident to s OR t
        if u == s or v == s or u == t or v == t:
            return w_incident
        # Arbitrary positive weight otherwise
        return rng.uniform(other_w_min, other_w_max)

    # Path edges
    add_undirected_edge(s, m, edge_weight(s, m))  # incident to s => w_incident
    add_undirected_edge(m, t, edge_weight(m, t))  # incident to t => w_incident

    # Clique on {s} ∪ {s1..sD}
    left = [s] + s_nodes
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            u, v = left[i], left[j]
            add_undirected_edge(u, v, edge_weight(u, v))  # edges (s,si) get w_incident automatically

    # Clique on {t} ∪ {t1..tD}
    right = [t] + t_nodes
    for i in range(len(right)):
        for j in range(i + 1, len(right)):
            u, v = right[i], right[j]
            add_undirected_edge(u, v, edge_weight(u, v))  # edges (t,ti) get w_incident automatically

    # Randomize adjacency list order
    #for u in range(n):
    #    rng.shuffle(adj[u])

    if not return_labels:
        return adj, s, t

    labels: Dict[int, str] = {s: "s", t: "t", m: "m"}
    for i, node in enumerate(s_nodes, start=1):
        labels[node] = f"s{i}"
    for i, node in enumerate(t_nodes, start=1):
        labels[node] = f"t{i}"
    return adj, s, t, labels

def has_path(adj: List[List[Tuple[int, float]]], s: int, t: int) -> bool:
    """Unweighted reachability check via BFS."""
    n = len(adj)
    seen = [False] * n
    stack = [s]
    seen[s] = True
    while stack:
        u = stack.pop()
        if u == t:
            return True
        for v, _ in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return False