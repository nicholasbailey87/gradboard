import math
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW


class AdamS(Optimizer):
    r"""
    Implements Adam with stable weight decay (AdamS) as proposed in
        "On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them:
        A Gradient-Norm Perspective" (https://openreview.net/pdf?id=vnGcubtzR1).

    This implementation was from the git repo
        http://github.com/zeke-xie/stable-weight-decay-regularization/
            blob/master/swd_optim/adams.py (MIT license ca. July 2025)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-4)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        weight_decay *= lr
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        exp_avg_sq_hat_sum = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_size += p.numel()

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamS does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                beta1, beta2 = group["betas"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum()

        # Calculate the sqrt of the mean of all elements in exp_avg_sq_hat
        exp_avg_mean_sqrt = math.sqrt(exp_avg_sq_hat_sum / param_size)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Perform stable weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(
                        1 - group["weight_decay"] * group["lr"] / exp_avg_mean_sqrt
                    )

                beta1, beta2 = group["betas"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                denom = exp_avg_sq_hat.sqrt().add(group["eps"])

                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        # Make sure internal tensors are still leaf tensors
        # state['exp_avg'] = state['exp_avg'].detach()
        # state['exp_avg_sq'] = state['exp_avg_sq'].detach()

        return loss


def get_optimiser(model, optimiser=AdamW, lr=1e-3, weight_decay=1e-2):
    """
    Defaults are from one of the presets from the accompanying repo to Hassani
        et al. (2023) "Escaping the Big Data Paradigm with Compact Transformers",
        https://github.com/SHI-Labs/Compact-Transformers/blob/main/configs/
        pretrained/cct_7-3x1_cifar100_1500epochs.yml
    """
    # TODO: print a warning when params are excluded from weight decay IFF wd is set
    weight_decay_exclude = []
    for keyword in ["nondecay", "bias", "norm", "embedding", "beta"]:
        weight_decay_exclude += [
            p for name, p in model.named_parameters() if keyword in name.lower()
        ]
    weight_decay_exclude = set(weight_decay_exclude)
    weight_decay_include = set(model.parameters()) - weight_decay_exclude
    return optimiser(
        [
            {"params": list(weight_decay_include)},
            {"params": list(weight_decay_exclude), "weight_decay": 0.0},
        ],
        weight_decay=weight_decay,
        lr=lr,
    )
