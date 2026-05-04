import torch.nn as nn

from .linear import MiCALinear


def apply_mica(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
) -> nn.Module:
    """
    Apply MiCA to all nn.Linear layers whose immediate name is in target_modules.

    Steps:
      1. Freeze every parameter in the model.
      2. Replace each target nn.Linear with a MiCALinear (adds a trainable A).

    Returns the model (modified in-place).
    """
    for param in model.parameters():
        param.requires_grad_(False)

    _replace_recursive(model, set(target_modules), rank, alpha)
    return model


def _replace_recursive(
    module: nn.Module,
    targets: set[str],
    rank: int,
    alpha: float,
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, MiCALinear(child, rank=rank, alpha=alpha))
        else:
            _replace_recursive(child, targets, rank, alpha)
