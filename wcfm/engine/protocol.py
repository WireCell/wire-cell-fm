"""The ONLY thing the framework knows about a model.

The training infrastructure does not know what the backbone does. It knows how to run a
training step, reduce diagnostics, and checkpoint. Everything domain-specific -- views,
masking, teachers, heads, losses -- lives under `wcfm.model`, and
`tests/test_import_graph.py` fails the build if a framework package imports it.

The contract is the `TrainingModule` protocol below, six methods a model must implement:

- `training_step(batch, ctx)` runs the forward and returns the loss on `StepOutput`.
  The engine takes the backward.
- `param_groups()` is the whole input to the optimizer, so a model declares a frozen
  backbone or a discriminative rate there.
- `observables()` returns detached tensors under names the model chooses. The metrics
  layer treats them as opaque: `"student/dec_full"` is a string the model picked.
- `on_step_end(ctx)` runs after the optimizer step -- teacher EMA, per-term updates,
  model-owned schedules.
- `state_dict()` and `load_state_dict(sd)` are plain `nn.Module` semantics. What they
  write is one opaque blob in the checkpoint.

Three more hooks are optional: `provenance()` (`ReportsProvenance` below),
`grad_taxonomy` and `last_term_gradients`. A model that defines none of them
gets the engine's default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass
class StepOutput:
    """What a training step reports back.

    - `scalars` are already-detached floats and `n_samples` is what this micro-step saw.
      Both go into the metrics.
    - `loss` is the differentiable total, which the engine backwards under Fabric, inside
      the OOM guard, with the accumulation gate applied. `None` means there is nothing to
      differentiate this step, and no backward is taken.
    """

    scalars: dict[str, float]
    n_samples: int
    loss: Tensor | None = None


@dataclass
class StepContext:
    """Capabilities and position handed to the module for one step.

    - `step` and `epoch` place the step in the run.
    - `device` is where the batch already is.
    - `all_reduce` lets a module reduce its own diagnostics without importing distributed.
    - `is_last_microstep` is true on the micro-step that closes an accumulation window,
      which is the one the optimizer steps on.
    """

    step: int
    epoch: int
    device: torch.device

    # `module` is the actual model, wrapped by DDP.
    #
    #  NOTE: Call this, not `self`, for every forward whose gradients matter.
    #
    #        DDP arms its reducer inside `DistributedDataParallel.forward`
    #        (`prepare_for_backward`): the per-parameter autograd hooks exist
    #        from construction but return early until it has run.
    #        A module computing through its own submodules therefore bypasses
    #        the wrapper and nothing is reduced --> same numbers on one GPU, N
    #        unsynchronised copies on N. This would be a SILENT error.
    #
    #        Arming happens once per call and the engine takes one backward per micro-step,
    #        so everything that backward will touch must come out of ONE call to this handle.
    module: Callable[..., Any]
    all_reduce: Callable[..., Tensor]
    is_last_microstep: bool = True
    # Free-form scratch the engine does not read; kept so modules never need a global.
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrainingModule(Protocol):
    """The contract. A `torch.nn.Module` that implements these is a model the engine can run."""

    def training_step(self, batch: Any, ctx: StepContext) -> StepOutput:
        """Run the forward and return the loss on `StepOutput`; the engine backwards it."""
        ...

    def param_groups(self) -> list[dict]:
        """Parameter groups in `torch.optim` form. The optimizer is built from these."""
        ...

    def observables(self) -> dict[str, Tensor]:
        """Detached tensors under names of the module's choosing, for the metrics layer."""
        ...

    def on_step_end(self, ctx: StepContext) -> None:
        """Teacher EMA, per-term updates, model-owned schedules. Runs after the optimizer step."""
        ...

    def state_dict(self) -> dict:
        """Everything needed to resume this module. One opaque blob to the framework."""
        ...

    def load_state_dict(self, sd: dict) -> None:
        """Restore what `state_dict` wrote.

        The engine reads the checkpoint with `map_location="cpu"`, so a tensor the module
        materialises during this call lands on the CPU while the module itself is already
        on the device. A module that creates a buffer here must place it on its own device
        (`next(self.parameters()).device`); one whose buffers are registered at
        construction has nothing to do. The engine does not repair it, because rebinding a
        tensor after `fabric.setup()` breaks DDP's bucket views.
        """
        ...


# --------------------------------------------------------------------- optional hooks
#
# Discovered by `getattr`. A model that does not define one gets the
# engine's default, which is what keeps the contract above at six methods.

@runtime_checkable
class ReportsProvenance(Protocol):
    """Say something about yourself that belongs in `run_metadata.json`.

    The engine records the repo's `git` block and parameter counts on its own.
    A model reports its backbone variant, its tap count, and anything else of its own.
    The return value is written verbatim under the file's `module` key.
    It is an opaque dict because `wcfm.config` must not name a model type.
    """

    def provenance(self) -> dict: ...
