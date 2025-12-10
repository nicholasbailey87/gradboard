import math
import warnings
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW

EXCLUDE_FROM_WEIGHT_DECAY = ["nondecay", "bias", "norm", "embedding", "beta"]


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
    weight_decay_exclude = []

    for keyword in exclude_keywords:
        weight_decay_exclude += [
            p for name, p in model.named_parameters() if keyword in name.lower()
        ]

    weight_decay_exclude = set(weight_decay_exclude)

    if len(weight_decay_exclude) > 0:
        warnings.warn(
            "Excluded the following parameters from weight decay based on "
            "exclude keywords: {weight_decay_exclude}",
            stacklevel=2,
        )

    weight_decay_include = set(model.parameters()) - weight_decay_exclude

    return optimiser(
        [
            {"params": list(weight_decay_include)},
            {"params": list(weight_decay_exclude), "weight_decay": 0.0},
        ],
        weight_decay=weight_decay,
        lr=lr,
    )
