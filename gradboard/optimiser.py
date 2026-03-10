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


def get_adjusted_learning_rate(parameter_name, base_learning_rate) -> float:
    coefficients = {
        "embedding": 1.0,
        "attn.q_proj.weight": 0.8,
        "attn.k_proj.weight": 0.8,
        "attn.v_proj.weight": 0.4,
        "attn.out_proj.weight": 0.4,
        "ff.linear_in.weight": 0.6,
        "ff.linear_out.weight": 0.6,
    }
    for k, v in coefficients.items():
        if k in parameter_name:
            return base_learning_rate * v

    return base_learning_rate


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

    levels = defaultdict(list)

    for name, p in model.named_parameters():

        adjusted_learning_rate = get_adjusted_learning_rate(name, lr)

        if p not in weight_decay_exclude_params:
            if len(p.size()) == 2:
                width = max(p.size())
                if d_base_model is not None:
                    scaling_factor = math.sqrt(width) / math.sqrt(d_base_model)
                else:
                    scaling_factor = 1
                adjusted_weight_decay = scaling_factor * weight_decay
                levels[(adjusted_learning_rate, adjusted_weight_decay)].append(p)
            else:
                levels[(adjusted_learning_rate, 0.0)].append(p)
        else:
            levels[(adjusted_learning_rate, 0.0)].append(p)

    parameter_groups = []

    for k, v in levels.items():
        group_learning_rate, group_weight_decay = k
        parameter_groups.append(
            {"params": v, "weight_decay": group_weight_decay, "lr": group_learning_rate}
        )

    return optimiser(
        parameter_groups,
        weight_decay=weight_decay,
        lr=lr,
        eps=eps,
    )
