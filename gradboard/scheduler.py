"""
Based on Smith (2017) https://arxiv.org/abs/1506.01186
"""

from typing import Optional
import copy
import math

from torch.amp import GradScaler

from .cycles import Cycle


class PASS:
    """
    A self-configuring learning rate scheduler
    """

    def __init__(
        self,
        learning_rate_schedule: Cycle,
        model,
        optimiser,
        scaler: Optional[GradScaler] = None,
        range_test: bool = False,
        cool_point_multiplier: float = 1 / 60,
    ):
        """
        If not using range test, we assume the optimiser has the learning rates
            set as desired.
        """

        if type(model).__name__ == "OptimizedModule":
            raise ValueError(
                "PASS received a compiled model. You must initialize PASS with the "
                "raw model *before* calling torch.compile().\n"
                "Correct order:\n"
                "  scheduler = PASS(..., model, ...)\n"
                "  model = torch.compile(model)"
            )

        self.model = model
        self.optimiser = optimiser
        self.scaler = scaler

        self.learning_rate_schedule = learning_rate_schedule

        self.range_test = range_test

        self.original_param_groups = copy.deepcopy(optimiser.param_groups)

        self.cool_point_multiplier = cool_point_multiplier

        self.step_count = 0

        self.range_test_results = []

        if self.range_test:
            self.start_range_test()  # sets LR to 1E-7

    @property
    def lr(self):
        """
        Return first lr from self.optimiser.param_groups
            (this is used in learning rate range tests, in which case we can
            assume they are all the same!)
        """
        for group in self.optimiser.param_groups:
            return group["lr"]

    @property
    def in_range_test(self):
        if not self.range_test:
            return False
        elif (len(self.range_test_results) == 0) or (
            not math.isnan(self.range_test_results[-1][1])
        ):
            return True
        else:
            return False

    @property
    def trained(self):
        if not self.range_test:
            return True
        elif math.isnan(self.range_test_results[-1][1]):
            return True
        else:
            return False

    @property
    def finished(self):
        return self.step_count >= len(self.learning_rate_schedule) - 1

    def clean_model_state_dict(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_state_dict[k[10:]] = v  # Strip the "_orig_mod." prefix (length 10)
            else:
                new_state_dict[k] = v
        return new_state_dict

    def _current_states(self):
        saved_states = {
            "model": copy.deepcopy(
                self.clean_model_state_dict(self.model.state_dict())
            ),
            "optimiser": copy.deepcopy(self.optimiser.state_dict()),
        }
        if self.scaler is not None:
            saved_states["scaler"] = copy.deepcopy(self.scaler.state_dict())
        return saved_states

    def save_states(self):
        self.saved_states = self._current_states()

    def load_states(self, states: dict):
        self.model.load_state_dict(self.clean_model_state_dict(states["model"]))
        self.optimiser.load_state_dict(states["optimiser"])
        if self.scaler is not None:
            self.scaler.load_state_dict(states["scaler"])

    def recover_states(self):
        self.load_states(self.saved_states)

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "range_test": self.range_test,
            "range_test_results": self.range_test_results,
            "original_param_groups": self.original_param_groups,
            "cool_point_multiplier": self.cool_point_multiplier,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict):
        self.load_states(state_dict)
        self.range_test = state_dict["range_test"]
        self.range_test_results = state_dict["range_test_results"]
        self.original_param_groups = state_dict["original_param_groups"]
        self.cool_point_multiplier = state_dict["cool_point_multiplier"]
        self.step_count = state_dict["step_count"]

    @property
    def _schedule_multiplier(self):
        return self.learning_rate_schedule(
            min(self.step_count, self.learning_rate_schedule.total_steps)
        )

    def set_all_lr(self, lr):
        for group in self.optimiser.param_groups:
            group["lr"] = lr

    def start_range_test(self):
        self.range_test = True
        self.range_test_results = []
        self.save_states()
        self.set_all_lr(1e-7)

    def scale_all_lr(self, scaling_factor):
        self.set_all_lr(self.lr * scaling_factor)

    def end_range_test(self):
        self.recover_states()
        self.update_learning_rates()

    def _smoothed_range_test(self, range_test_results):
        range_test_results = sorted(range_test_results, key=lambda x: x[0])
        learning_rates = [t[0] for t in range_test_results]
        losses = [t[1] for t in range_test_results]
        losses = losses[:-1] + [10 * max(losses)]
        return list(zip(learning_rates, losses, strict=True))

    def _plot_range_test(self, range_test_results):
        """
        Returns a tuple with x values (learning rates) and y values (losses)
            which can then be passed to e.g. pyplot. We recommend presenting
            the plot with a logarithmic x axis.
        """
        range_test_results = sorted(range_test_results, key=lambda x: x[0])
        learning_rates = [t[0] for t in range_test_results]
        losses = [t[1] for t in range_test_results]
        return learning_rates, losses

    def _apply_range_test_result(self):
        """
        ...
        """
        range_test_results = self._smoothed_range_test(self.range_test_results)
        minimum = min(range_test_results, key=lambda x: x[1])
        points_left_of_min = [r for r in range_test_results if r[0] < minimum[0]]
        max_left_of_min = max(points_left_of_min, key=lambda x: x[1])
        difference = max_left_of_min[1] - minimum[1]
        max_lr = None
        for p in sorted(points_left_of_min, key=lambda x: x[0]):
            if (max_lr is None) and (p[1] < minimum[1] + 0.2 * difference):
                max_lr = p[0]
            else:
                continue
        self.set_all_lr(max_lr)
        self.original_param_groups = copy.deepcopy(self.optimiser.param_groups)
        print("High LR", max_lr)

    def update_learning_rates(self):
        if not self.finished:
            for original, current in zip(
                self.original_param_groups, self.optimiser.param_groups, strict=True
            ):
                base_lr = original["lr"]
                min_lr = base_lr * self.cool_point_multiplier
                current_lr = min_lr + (base_lr - min_lr) * self._schedule_multiplier
                current["lr"] = current_lr

    def _append_to_range_test(self, loss_item: float):

        lr = self.lr

        self.range_test_results.append((lr, loss_item))

        if math.isnan(loss_item) or (lr >= 1.0):
            self._apply_range_test_result()
            self.end_range_test()
        else:
            # Continue range test, step up learning rate
            self.scale_all_lr(1.05)

    def step(self, loss_item: Optional[float] = None):
        """
        This function manages the process of
            * Doing an initial range test
            * Training for one microcycle using the learning rates from the
                  initial range test ("burn in")
            * Doing a second range test to set the learning rate schedule for
                  the rest of training
            * Updating learning rates during training according to the macrocycle
        """
        if self.in_range_test:  # True at init unless self.range_test = False
            if not isinstance(loss_item, float):
                raise ValueError(
                    "When using range test functionality, "
                    "`step()` expects a loss item."
                )
            self._append_to_range_test(loss_item)
        elif self.trained and not self.finished:
            self.step_count += 1
            self.update_learning_rates()
        else:
            pass
