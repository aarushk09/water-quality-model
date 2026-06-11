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
                add_self_loops=False,  # We manage topology externally; avoid OOB scatter on batched edge_index
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


class LagAwareGAT(nn.Module):
    """
    Graph Attention with learned travel-time lags for dam-causality modeling.

    Key idea: when computing the message from upstream node j -> downstream node i,
    shift j's feature history back by tau_{ij} timesteps (the travel-time lag).
    tau_{ij} is initialized from physical estimates and fine-tuned jointly.

    Novel scientific contribution: plot learned tau_{ij} vs. published travel-time
    studies to validate causal delay capture.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        num_edges: int,
        heads: int = 4,
        dropout: float = 0.1,
        initial_lags: Optional[torch.Tensor] = None,
        max_lag: int = 32,
        interpolate_lags: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        self.heads = heads
        self.dropout_rate = dropout
        self.max_lag = max_lag
        self.interpolate_lags = interpolate_lags
        self.out_dim = hidden_channels

        # Learned lag parameters in log space to enforce >= 0
        if initial_lags is not None:
            init_lags = initial_lags.float().clamp(0, max_lag)
        else:
            init_lags = torch.zeros(num_edges)
        self.log_lags = nn.Parameter(torch.log1p(init_lags))

        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.msg_net = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.attn_q = nn.Linear(hidden_channels, heads, bias=False)
        self.attn_k = nn.Linear(hidden_channels, heads, bias=False)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

    @property
    def lags(self) -> torch.Tensor:
        return torch.expm1(self.log_lags).clamp(0, self.max_lag)

    def _get_lagged_features(self, x_history: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
        """Linear interpolation for differentiable lag shift."""
        b, n, t, f = x_history.shape
        lag_f = lag.clamp(0, t - 1)
        lo = lag_f.long().clamp(0, t - 2)
        hi = (lo + 1).clamp(0, t - 1)
        frac = lag_f - lo.float()
        idx_lo = t - 1 - lo
        idx_hi = t - 1 - hi
        h_lo = x_history[:, :, idx_lo, :]
        h_hi = x_history[:, :, idx_hi, :]
        return h_lo * (1 - frac) + h_hi * frac

    def forward(
        self,
        x_history: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x_history: [B, N, T, F] -> [B, N, H] at current (last) timestep."""
        b, n, t, f = x_history.shape
        lags = self.lags
        x_now = x_history[:, :, -1, :]
        h_self = F.relu(self.node_proj(x_now))
        h_agg = torch.zeros_like(h_self)
        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        for e in range(edge_index.size(1)):
            s = int(src_nodes[e].item())
            d = int(dst_nodes[e].item())
            if self.interpolate_lags:
                h_src_lagged = self._get_lagged_features(x_history, lags[e])[:, s, :]
            else:
                shift = int(lags[e].round().item())
                idx = max(0, t - 1 - shift)
                h_src_lagged = x_history[:, s, idx, :]
            h_dst = h_self[:, d, :]
            msg = self.msg_net(torch.cat([h_src_lagged, h_dst], dim=-1))
            q = self.attn_q(h_dst)
            k = self.attn_k(F.relu(self.node_proj(h_src_lagged)))
            attn = torch.sigmoid((q * k).sum(-1, keepdim=True) / (self.heads ** 0.5))
            h_agg[:, d, :] = h_agg[:, d, :] + attn * msg

        h_out = self.norm(h_self + F.dropout(h_agg, p=self.dropout_rate, training=self.training))
        return self.out_proj(h_out), None

    @torch.no_grad()
    def get_learned_lags(self) -> dict:
        lags = self.lags.cpu().numpy()
        return {
            "lags_timesteps": lags,
            "lags_minutes": lags * 15,
            "lags_hours": lags * 15 / 60,
        }


def apply_lag_gat(
    gat: LagAwareGAT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    return_attention: bool = False,
    edge_attr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply LagAwareGAT across the full temporal sequence.
    x: [B, N, T, F] -> [B, N, T, H]
    """
    b, n, t, f = x.shape
    outputs = []
    for ti in range(t):
        x_hist = x[:, :, :ti + 1, :]
        h, attn = gat(x_hist, edge_index, return_attention, edge_attr)
        outputs.append(h.unsqueeze(2))
    result = torch.cat(outputs, dim=2)
    return result, attn
