import math
import warnings
from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW

EXCLUDE_FROM_WEIGHT_DECAY = ["nondecay", "bias", "norm", "embedding", "beta"]


def register_optimiser_recursive(module, optimizer):
    """
    Recursively walks the module tree and registers the optimiser
        with any layer that supports .register_optimiser()
    """
    # Check if the module itself has the method
    if hasattr(module, "register_optimiser") and callable(module.register_optimiser):
        module.register_optimiser(optimizer)

    # Recursively check children
    for child in module.children():
        register_optimiser_recursive(child, optimizer)


def get_optimiser(
    model,
    d_base_model=None,
    optimiser=AdamW,
    lr=1e-3,
    weight_decay=1e-2,
    eps=1e-8,
    exclude_keywords=EXCLUDE_FROM_WEIGHT_DECAY,
):
    """
    Set up an optimiser for a transformer model, excluding appropriate submodules
        from weight decay and scaling up weight decay for tensors that have scaled
        from `d_base_model`
    """

    weight_decay_exclude_params = []
    weight_decay_exclude_names = []

    for keyword in exclude_keywords:
        weight_decay_exclude_params += [
            p for name, p in model.named_parameters() if keyword in name.lower()
        ]
        weight_decay_exclude_names += [
            name for name, _ in model.named_parameters() if keyword in name.lower()
        ]

    weight_decay_exclude_params = set(weight_decay_exclude_params)

    if len(weight_decay_exclude_params) > 0:
        warnings.warn(
            "Excluded the following parameters from weight decay based on "
            f"exclude keywords: {set(weight_decay_exclude_names)}",
            stacklevel=2,
        )

    weight_decay_include = [
        p for p in model.parameters() if p not in weight_decay_exclude_params
    ]

    weight_decay_levels = defaultdict(list)
    weight_decay_levels[0.0] = [
        p for p in model.parameters() if p in weight_decay_exclude_params
    ]

    for p in weight_decay_include:
        if len(p.size()) == 2:
            width = max(p.size())
            if d_base_model is not None:
                scaling_factor = math.sqrt(width) / math.sqrt(d_base_model)
            else:
                scaling_factor = 1
            adjusted_weight_decay = scaling_factor * weight_decay
            weight_decay_levels[adjusted_weight_decay].append(p)
        else:
            weight_decay_levels[0.0].append(p)

    default_weight_decay = max(
        weight_decay_levels.keys(), key=lambda x: len(weight_decay_levels[x])
    )

    parameter_groups = []

    for k, v in weight_decay_levels.items():
        if k != default_weight_decay:
            parameter_groups.append({"params": v, "weight_decay": k})

    parameter_groups.append({"params": weight_decay_levels[default_weight_decay]})

    return optimiser(
        parameter_groups,
        weight_decay=default_weight_decay,
        lr=lr,
        eps=eps,
    )
