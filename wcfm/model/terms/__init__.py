"""Terms: what the objective asks of a feature bundle. `hybrid` is `DinoTerm` with
`score_injected=True`."""

from .base import Term, TermOutput
from .charge import ChargeTerm
from .dino import DinoTerm
from .gather import gather_at_coords, match_and_gather
from .heads import DINOProjectionHead
from .losses import (
    DinoLossOutput,
    PixelDINOLoss,
    charge_loss,
    occupancy_loss,
    two_stage_mean,
)
from .occupancy import OccupancyTerm

__all__ = [
    "ChargeTerm",
    "DINOProjectionHead",
    "DinoLossOutput",
    "DinoTerm",
    "OccupancyTerm",
    "PixelDINOLoss",
    "Term",
    "TermOutput",
    "charge_loss",
    "gather_at_coords",
    "occupancy_loss",
    "match_and_gather",
    "two_stage_mean",
]
