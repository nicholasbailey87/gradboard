import math
import warnings
from collections import defaultdict, OrderedDict

from torch.optim import AdamW

EXCLUDE_FROM_WEIGHT_DECAY = ["nondecay", "bias", "norm", "embedding", "beta"]

SHARPNESS_DISPARITY_COEFFICIENTS = OrderedDict(
    [
        ("initial_ff", 1.43),
        ("embedding", 1.43),
        ("attn.q_proj.weight", 1.14),
        ("attn.k_proj.weight", 1.14),
        ("attn.v_proj.weight", 0.57),
        ("attn.out_proj.weight", 0.57),
        ("ff.linear_in.weight", 0.86),
        ("ff.linear_out.weight", 0.86),
    ]
)


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
    weight_decay_exclusions=EXCLUDE_FROM_WEIGHT_DECAY,
    blockwise_lr=False,
    blockwise_lr_coefficients=None,
):
    """
    Set up an optimiser for a transformer model, excluding appropriate submodules
        from weight decay and scaling up weight decay for tensors that have scaled
        from `d_base_model`
    """

    weight_decay_exclude_names = set()
    levels = defaultdict(list)

    for name, p in model.named_parameters():

        wd_exclude = any(keyword in name.lower() for keyword in weight_decay_exclusions)

        if not p.requires_grad:
            continue
        elif wd_exclude:
            weight_decay_exclude_names.add(name)
            weight_decay_coefficient = 0.0
        elif len(p.size()) == 2:
            _, in_features = p.size()
            weight_decay_coefficient = math.sqrt(in_features) / math.sqrt(
                base_model_embedding_size
            )
        elif len(p.size()) > 2:
            weight_decay_coefficient = 1.0
        else:
            weight_decay_coefficient = 0.0

        lr_coefficient = 1.0

        if blockwise_lr:
            if blockwise_lr_coefficients is None:
                blockwise_lr_coefficients = SHARPNESS_DISPARITY_COEFFICIENTS
            for pattern, coeff in blockwise_lr_coefficients.items():
                if pattern in name:
                    lr_coefficient = coeff
                    break

        levels[(lr * lr_coefficient, weight_decay * weight_decay_coefficient)].append(p)

    parameter_groups = []

    for k, v in levels.items():
        group_learning_rate, group_weight_decay = k
        parameter_groups.append(
            {"params": v, "weight_decay": group_weight_decay, "lr": group_learning_rate}
        )

    if weight_decay_exclude_names:
        warnings.warn(
            "Excluded the following parameters from weight decay based on "
            f"exclude keywords: {set(weight_decay_exclude_names)}",
            stacklevel=2,
        )

    return optimiser(
        parameter_groups,
        weight_decay=weight_decay,
        lr=lr,
        eps=eps,
    )
