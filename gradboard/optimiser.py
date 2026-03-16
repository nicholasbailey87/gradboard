import math
import warnings
from collections import defaultdict

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
    base_model_embedding_size,
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

        if (len(p.size()) == 2) and (p not in weight_decay_exclude_params):
            _, in_features = p.size()
            weight_decay_coefficient = math.sqrt(in_features) / math.sqrt(
                base_model_embedding_size
            )
        else:
            weight_decay_coefficient = 0.0

        if "ff.linear_in" in name:
            learning_rate_coefficient = 1 / math.sqrt(model.transformer_ff_ratio)
            weight_decay_coefficient *= math.sqrt(model.transformer_ff_ratio)
        else:
            learning_rate_coefficient = 1.0

        levels[
            (lr * learning_rate_coefficient, weight_decay * weight_decay_coefficient)
        ].append(p)

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
