"""
Utilities for generating a range of learning rate schedules.
"""

import math
from typing import Optional, List, Union, Callable


def ascent(step: int, total_steps: int) -> float:
    """
    Get a sequence of numbers evenly spaced between 0 and 1 so that the first
        number is 0 and the last is 1 and there are `total_steps` numbers in
        the sequence.
    """
    return round(step / (total_steps - 0.999), 8)


def triangle(step: int, total_steps: int) -> float:
    """
    Get a triangular sequence of numbers between 0 and 1, going up in half of
        `total_steps` and coming down in the other half, peaking at ~1.
    """
    half = int(math.ceil(total_steps / 2))
    if step < half:
        return 2 * ascent(step, total_steps)
    else:
        return 2 - 2 * ascent(step, total_steps)


def cosine(step: int, total_steps: int) -> float:
    """
    Get a sequence of numbers between 0 and 1 in the shape of a cosine wave with
        wavelength `total_steps`.
    """
    assert total_steps != 0
    angle = (step / (total_steps - 0.999)) * (2 * math.pi)
    return round((math.cos(angle) + 1) / 2, 8)


def quarter_circle(step: int, total_steps: int) -> float:
    """
    Get a sequence of numbers between 0 and 1 in the shape of a semi-circle with
        diameter `total_steps'.
    """
    assert total_steps != 0
    x = step / (total_steps - 0.999)
    # x^2 + y^2 = r^2 = 1
    # Therefore y^2 = 1 - x^2
    # Therefore y^2 = (1 + x)(1 - x)
    y_squared = max(1 - x, 0) * (1 + x)
    try:
        return math.sqrt(y_squared)
    except ValueError:
        print("total_steps: ", total_steps)
        print("step: ", total_steps)
        raise ValueError("Still broken!") from None


def half_cosine(step: int, total_steps: int) -> float:
    """
    Get a sequence of numbers between 0 and 1 in the shape of the descending
        half of a cosine wave with wavelength 2*`total_steps`.
    """
    return cosine(step, (total_steps * 2) - 1)


def cycloid(step: int, total_steps: int) -> float:
    """
    Get a sequence of numbers between 0 and 1 in the shape of a cycloid with
        circle diameter 1.0 and `total_steps/(2*math.pi)` steps per cycle.
    """
    x = step * (math.pi / (total_steps - 1))

    def fx(t):
        return 0.5 * (t - math.sin(t)) - x

    def fx_prime(t):
        return 0.5 - 0.5 * math.cos(t)

    def fy_prime(t):
        return 0.5 - 0.5 * -math.sin(t)

    angle_estimate = 0.5 * x

    # XXX: 200 iterations is too many! Use a more efficient root finding algorithm
    for _ in range(200):
        if abs(fx_prime(angle_estimate)) > 0.1:
            update = fx(angle_estimate) / fx_prime(angle_estimate)
        else:
            update = fx(angle_estimate) / fy_prime(angle_estimate)
        angle_estimate = angle_estimate - update

    return 0.5 * (1 - math.cos(angle_estimate))


def half_cycloid(step: int, total_steps: int) -> float:
    return cycloid(total_steps + step, 2 * total_steps)


class Cycle:
    def __init__(
        self,
        generating_function: Union[str, Callable],
        training_examples,
        epochs,
        batch_size,
        t_0: Optional[int] = None,
        t_mult: float = 1.0,
        t_scale: float = 1.0,
        low=0.0,
        high=1.0,
        reflect=False,
    ):
        self.training_examples = training_examples
        self.epochs = epochs
        self.batch_size = batch_size
        self.total_steps = int(
            epochs * (math.floor(training_examples / batch_size) + 1)
        )

        self.t_0 = (
            t_0 * (training_examples / batch_size)
            if t_0 is not None
            else self.total_steps
        )
        self.t_mult = t_mult
        self.t_scale = t_scale

        self.low = low
        self.high = high

        self.reflect = reflect

        if not callable(generating_function):
            raise NotImplementedError(
                "`generating_function` must be a callable object or one of "
                '"ascent", "triangle", "cosine", "half_cosine" or "half_cycloid"'
            )
        else:
            self._generating_function = generating_function

    def _get_window(self, step):
        windows = self._windows()
        cumulative = [
            sum([w[0] for w in windows][: i + 1]) for i in range(len(windows))
        ]
        position = None
        local_step = None
        for i, c in enumerate(cumulative):
            if c > step:
                position = i
                local_step = step if i == 0 else step - cumulative[i - 1]
                break
        window_width, window_height = windows[position]
        return window_width, local_step, window_height

    def _generate(self, step) -> list:
        total_steps, step, scale = self._get_window(step)
        y = self._generating_function(step, total_steps)
        y = y * scale
        y = 1 - y if self.reflect else y
        return y * (self.high - self.low) + self.low

    def __call__(self, n):
        return self._generate(n)

    def __len__(self):
        return self.total_steps

    def _windows(self):
        assert self.t_mult > 0

        # Get tile widths
        widths = [self.t_0]
        while True:
            next_item = widths[-1] * self.t_mult
            if sum(widths) + next_item <= self.total_steps:
                widths.append(next_item)
            else:
                break
        for i in range(1, len(widths)):
            widths[i] = int(widths[i] * (self.total_steps / sum(widths)))
        widths[-1] += self.total_steps - sum(widths)

        # Get tile heights
        heights = [1.0 * self.t_scale**i for i in range(len(widths))]

        return list(zip(widths, heights, strict=True))

    @property
    def stats(self) -> float:
        """
        Returns the area (as a percentage of the area of a curve where the learning
            rate is constant max_lr), percentage ascent steps and percentage descent
            steps of a learning rate schedule.
        """
        total_area = 0
        max_area = 0
        ascent_steps = 0
        descent_steps = 0
        avg_up_gradient = 0
        avg_down_gradient = 0
        total_gradient = 0
        previous_lr = None
        for s in range(self.total_steps):
            height = self(s)
            total_area += height
            max_area += 1
            if previous_lr is None:
                pass
            elif previous_lr > height:
                descent_steps += 1
                avg_down_gradient += height - previous_lr
                total_gradient += height - previous_lr
            elif previous_lr < height:
                ascent_steps += 1
                avg_up_gradient += height - previous_lr
                total_gradient += height - previous_lr
            else:
                total_gradient += height
            previous_lr = height
        return {
            "area": total_area / max_area,
            "pc_ascent": round(ascent_steps / self.total_steps, 3),
            "pc_descent": round(descent_steps / self.total_steps, 3),
            "avg_up_gradient": round(avg_up_gradient, 3),
            "avg_down_gradient": round(avg_down_gradient, 3),
            "avg_gradient": round(-(self.high - self.low) / self.total_steps, 3),
        }


class CycleProduct(Cycle):
    def __init__(self, cycles: List[Cycle], reflect=False, normalise: bool = False):
        """
        Args:
            normalise: if true, the square root of the product is returned (i.e.
                the geometric mean of the two cycles that were multiplied together)
        """
        main_training_examples = cycles[0].training_examples
        main_batch_size = cycles[0].batch_size

        assert all(c.training_examples == main_training_examples for c in cycles)
        assert all(c.batch_size == main_batch_size for c in cycles)

        self.cycles = cycles
        self.reflect = reflect
        self.normalise = normalise

        def generating_function(step: int, total_steps: int) -> float:
            output = self.cycles[0](step)
            for c in self.cycles[1:]:
                output *= c(step % c.total_steps)
            if self.normalise:
                output = math.sqrt(output)
            return output

        super().__init__(
            generating_function=generating_function,
            training_examples=self.cycles[0].training_examples,
            epochs=self.cycles[0].epochs,
            batch_size=self.cycles[0].batch_size,
            reflect=reflect,
        )


class CycleSequence:
    def __init__(self, cycles: List[Cycle]):
        self.total_steps = sum([c.total_steps for c in cycles])
        self.cycles = cycles

    def _generate(self, step):
        cycle, step = self._get_cycle_and_step(step)
        return self.cycles[cycle](step)

    def _get_cycle_and_step(self, step):
        cycle_lengths = [c.total_steps for c in self.cycles]
        cumulative = [sum(cycle_lengths[: i + 1]) for i in range(len(cycle_lengths))]
        cycle = None
        local_step = None
        for i, c in enumerate(cumulative):
            if c > step:
                cycle = i
                local_step = step if i == 0 else step - cumulative[i - 1]
                break
        return cycle, local_step

    def __call__(self, step):
        return self._generate(step)

    def __len__(self):
        return self.total_steps

    @property
    def stats(self) -> float:
        """
        Returns the area (as a percentage of the area of a curve where the learning
            rate is constant max_lr), percentage ascent steps and percentage descent
            steps of a learning rate schedule.
        """
        cycle_ratios = [c.total_steps for c in self.cycles]
        cycle_stats = {k: v * cycle_ratios[0] for k, v in self.cycles[0].stats.items()}
        for i, cycle in enumerate(self.cycles):
            if i == 0:
                continue  # We already did the first one, above
            for k, v in cycle.stats.items():
                cycle_stats[k] += cycle_ratios[i] * v

        return {k: v / self.total_steps for k, v in cycle_stats.items()}
