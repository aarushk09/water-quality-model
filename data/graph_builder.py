"""Build PyG-style graph tensors from station metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def self_loop_edge_index(num_nodes: int = 1) -> np.ndarray:
    """Degenerate N=1 graph: single self-loop."""
    return np.array([[0], [0]], dtype=np.int64)


def build_edge_index_from_edges(
    site_ids: List[str],
    edges: List[Tuple[str, str]],
    travel_steps: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build directed edge_index from (upstream, downstream) pairs.

    Returns edge_index shape [2, E] and edge_attr [E, 1] (travel lag in steps).
    """
    site_to_idx = {s: i for i, s in enumerate(site_ids)}
    src, dst, attrs = [], [], []
    for i, (up, down) in enumerate(edges):
        if up in site_to_idx and down in site_to_idx:
            src.append(site_to_idx[up])
            dst.append(site_to_idx[down])
            lag = 1.0
            if travel_steps is not None and i < len(travel_steps):
                lag = float(max(1, travel_steps[i]))
            attrs.append(lag)
    if not src:
        return self_loop_edge_index(len(site_ids)), None
    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.array(attrs, dtype=np.float32).reshape(-1, 1)
    return edge_index, edge_attr


def k_nn_edges(
    lats: np.ndarray,
    lons: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Directed edges from higher latitude (upstream heuristic) to lower."""
    n = len(lats)
    if n <= 1:
        return self_loop_edge_index(n)
    coords = np.stack([lats, lons], axis=1)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    src, dst = [], []
    for i in range(n):
        neighbors = np.argsort(dists[i])[:k]
        for j in neighbors:
            # Heuristic: northern site -> southern site (downstream)
            if lats[i] >= lats[j]:
                src.append(i)
                dst.append(j)
            else:
                src.append(j)
                dst.append(i)
    if not src:
        return self_loop_edge_index(n)
    return np.array([src, dst], dtype=np.int64)


def load_graph_from_files(
    stations_json: Path,
    edges_csv: Optional[Path] = None,
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """Load stations and edges from NWIS fetch outputs."""
    with open(stations_json) as f:
        stations = json.load(f)
    site_ids = [s["site_no"] for s in stations]

    if edges_csv and Path(edges_csv).exists():
        edges_df = pd.read_csv(edges_csv, dtype={"upstream": str, "downstream": str})
        edges_df["upstream"] = edges_df["upstream"].str.zfill(8)
        edges_df["downstream"] = edges_df["downstream"].str.zfill(8)
        pairs = list(zip(edges_df["upstream"], edges_df["downstream"]))
        travel_steps = None
        if "travel_steps" in edges_df.columns:
            travel_steps = [int(v) for v in edges_df["travel_steps"]]
        elif "travel_minutes" in edges_df.columns:
            travel_steps = [max(1, int(v) // 15) for v in edges_df["travel_minutes"]]
        edge_index, edge_attr = build_edge_index_from_edges(
            site_ids, pairs, travel_steps=travel_steps
        )
    elif len(site_ids) == 1:
        edge_index = self_loop_edge_index(1)
        edge_attr = None
    else:
        lats = np.array([s.get("lat", 0.0) for s in stations])
        lons = np.array([s.get("lon", 0.0) for s in stations])
        edge_index = k_nn_edges(lats, lons, k=min(3, len(site_ids) - 1))
        edge_attr = None

    return site_ids, edge_index, edge_attr


def stations_metadata_dict(site_no: str, lat: float = 0.0, lon: float = 0.0) -> Dict:
    return {"site_no": site_no, "lat": lat, "lon": lon}
