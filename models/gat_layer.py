"""Graph attention layer for spatial mixing across monitoring stations."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def _batched_edge_index(
    edge_index: torch.Tensor, num_nodes: int, batch_size: int
) -> torch.Tensor:
    """Repeat a fixed graph topology for `batch_size` disjoint graphs."""
    offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
    ei = edge_index.unsqueeze(2) + offsets.view(1, 1, -1)
    return ei.reshape(2, -1)


class SpatialGAT(nn.Module):
    """
    Apply GAT per time step: [B, N, F] -> [B, N, hidden].

    Supports travel-time lag on upstream→downstream edges via edge_lags.
    For N=1 only a linear projection is used.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        force_pyg: Optional[bool] = None,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        self.heads = heads
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_dim = hidden_channels
        self.edge_dim = edge_dim

        use_pyg = (force_pyg if force_pyg is not None else HAS_PYG) and num_nodes > 1

        if num_nodes == 1:
            self.node_proj = nn.Linear(in_channels, hidden_channels)
            self.mode = "n1"
        elif use_pyg:
            self.conv = GATConv(
                in_channels,
                hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                concat=True,
                edge_dim=edge_dim,
            )
            self.mode = "pyg"
        else:
            self.lin = nn.Linear(in_channels, hidden_channels)
            self.edge_gate = nn.Linear(in_channels + edge_dim, hidden_channels)
            self.mode = "fallback"

    def _forward_n1(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, F] -> [B, 1, H]."""
        h = self.node_proj(x.squeeze(1))
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h.unsqueeze(1)

    def _forward_batched_pyg(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, f = x.shape
        x_flat = x.reshape(b * n, f)
        ei = _batched_edge_index(edge_index, n, b)
        ea = None
        if edge_attr is not None:
            ea = edge_attr.repeat(b, 1)
        out = self.conv(x_flat, ei, edge_attr=ea)
        return out.view(b, n, -1)

    def _forward_lag_fallback(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Message passing without PyG. Cross-site lags are precomputed in features.
        """
        b, n, _ = x.shape
        h_self = self.lin(x)
        h_agg = torch.zeros_like(h_self)
        deg = torch.zeros(n, device=x.device, dtype=h_self.dtype)

        src, dst = edge_index[0], edge_index[1]
        for e in range(edge_index.size(1)):
            s = int(src[e].item())
            d = int(dst[e].item())
            neighbor = x[:, s, :]
            edge_feat = edge_attr[e] if edge_attr is not None else torch.zeros(
                self.edge_dim, device=x.device, dtype=x.dtype
            )
            msg_in = torch.cat(
                [neighbor, edge_feat.view(1, -1).expand(b, -1)], dim=-1
            )
            msg = F.relu(self.edge_gate(msg_in))
            h_agg[:, d, :] = h_agg[:, d, :] + msg
            deg[d] = deg[d] + 1

        deg = deg.clamp(min=1).view(1, n, 1)
        h_out = F.relu((h_self + h_agg / deg) / 2)
        return F.dropout(h_out, p=self.dropout, training=self.training)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
        edge_attr: Optional[torch.Tensor] = None,
        x_history: Optional[torch.Tensor] = None,
        time_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, n, _ = x.shape
        if n == 1 or self.mode == "n1":
            return self._forward_n1(x), None
        if self.mode == "pyg":
            return self._forward_batched_pyg(x, edge_index, edge_attr), None
        return self._forward_lag_fallback(x, edge_index, edge_attr), None


def apply_gat_over_time(
    gat: SpatialGAT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    return_attention: bool = False,
    edge_attr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply lag-aware GAT at each timestep.

    x: [B, N, T, F] -> [B, N, T, H]
    """
    b, n, t, _ = x.shape
    if n == 1:
        x_bt = x.permute(0, 2, 1, 3).reshape(b * t, 1, -1)
        h_bt, _ = gat(x_bt, edge_index, return_attention)
        h = h_bt.view(b, t, 1, -1).permute(0, 2, 1, 3)
        return h, None

    x_bt = x.permute(0, 2, 1, 3).reshape(b * t, n, -1)
    h_bt, _ = gat(x_bt, edge_index, return_attention, edge_attr=edge_attr)
    h = h_bt.view(b, t, n, -1).permute(0, 2, 1, 3)
    return h, None
