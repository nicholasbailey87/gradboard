import math
import warnings
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
    optimiser=AdamW,
    lr=1e-3,
    weight_decay=1e-2,
    exclude_keywords=EXCLUDE_FROM_WEIGHT_DECAY,
):
    """
    Defaults are from one of the presets from the accompanying repo to Hassani
        et al. (2023) "Escaping the Big Data Paradigm with Compact Transformers",
        https://github.com/SHI-Labs/Compact-Transformers/blob/main/configs/
        pretrained/cct_7-3x1_cifar100_1500epochs.yml
    """
    weight_decay_exclude_params = []
    weight_decay_exclude_names = []

    for keyword in exclude_keywords:
        weight_decay_exclude_params += [
            p for name, p in model.named_parameters() if keyword in name.lower()
        ]
        weight_decay_exclude_names += [
            name for name, p in model.named_parameters() if keyword in name.lower()
        ]

    weight_decay_exclude_params = set(weight_decay_exclude_params)

    if len(weight_decay_exclude_params) > 0:
        warnings.warn(
            "Excluded the following parameters from weight decay based on "
            f"exclude keywords: {set(weight_decay_exclude_names)}",
            stacklevel=2,
        )

    weight_decay_include = set(model.parameters()) - weight_decay_exclude_params

    return optimiser(
        [
            {"params": list(weight_decay_include)},
            {"params": list(weight_decay_exclude_params), "weight_decay": 0.0},
        ],
        weight_decay=weight_decay,
        lr=lr,
    )
