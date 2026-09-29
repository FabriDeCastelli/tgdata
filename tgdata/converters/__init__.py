from __future__ import annotations

Edge = tuple[int, int, float]


def drop_duplicate_edges(edges: list[Edge]) -> list[Edge]:
    """For static topologies; time-varying ones keep repeats, which are genuine multi-edges."""
    weights: dict[tuple[int, int], float] = {}
    for src, dst, w in edges:
        if weights.setdefault((src, dst), w) != w:
            raise ValueError(f"edge {(src, dst)} repeats with weights {weights[src, dst]} and {w}")
    return [(src, dst, w) for (src, dst), w in weights.items()]
