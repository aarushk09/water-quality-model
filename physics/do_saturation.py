"""
Dissolved oxygen saturation concentration DO_sat(T) for freshwater.

Benson & Krause (1984) freshwater formulation.
"""

from __future__ import annotations

import torch


def do_saturation_benson_krause(temperature_c: torch.Tensor) -> torch.Tensor:
    t = temperature_c
    tk = t + 273.15
    ln_do = (
        -139.34411
        + (1.575701e5) / tk
        - (6.642308e7) / (tk**2)
        + (1.243800e10) / (tk**3)
        - (8.621949e11) / (tk**4)
    )
    return torch.exp(ln_do)


do_saturation = do_saturation_benson_krause


def reaeration_residual(
    do: torch.Tensor,
    temperature_c: torch.Tensor,
    dt_hours: float = 0.25,
    k_reaer: float = 0.5,
) -> torch.Tensor:
    """
    Simplified reaeration: dDO/dt ≈ k * (DO_sat - DO).
    Returns mean squared residual over interior timesteps.
    """
    do_sat = do_saturation(temperature_c).clamp(min=0.5)
    ddo = (do[..., 1:] - do[..., :-1]) / dt_hours
    target_rate = k_reaer * (do_sat[..., 1:] - do[..., 1:])
    return (ddo - target_rate).pow(2).mean()
