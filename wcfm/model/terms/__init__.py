"""Terms: what the objective asks of a feature bundle, or of the masked tokens of a point cloud.
`hybrid` is `DinoTerm` with `score_injected=True`."""

from .base import GroupTerm, Term, TermBase, TermOutput
from .chamfer import ChamferTerm
from .charge import ChargeTerm
from .dino import DinoTerm
from .distill import DistillTerm
from .energy import EnergyTerm
from .gather import gather_at_coords, match_and_gather
from .heads import DINOProjectionHead
from .losses import (
    DinoLossOutput,
    PixelDINOLoss,
    charge_loss,
    distill_loss,
    occupancy_loss,
    two_stage_mean,
)
from .occupancy import OccupancyTerm

__all__ = [
    "ChamferTerm",
    "ChargeTerm",
    "DINOProjectionHead",
    "DinoLossOutput",
    "DinoTerm",
    "DistillTerm",
    "EnergyTerm",
    "GroupTerm",
    "OccupancyTerm",
    "PixelDINOLoss",
    "Term",
    "TermBase",
    "TermOutput",
    "charge_loss",
    "distill_loss",
    "gather_at_coords",
    "occupancy_loss",
    "match_and_gather",
    "two_stage_mean",
]
