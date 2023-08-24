from typing import Callable
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from mimicus.darkemu_utils import Bias


class Extrapolator(ABC):
    def __init__(
        self, hmf: "HaloMassFunction", bias: Callable = Bias(), min_n: float = 1.0e-6,
    ):
        self.hmf = hmf
        self.bias = bias
        self.min_n = min_n

    def get_bias(self, n):
        if n >= self.min_n:
            return self.bias(n)
        return self.bias(self.min_n)

    def fill_zeroes_with_last(
        self, n, n_non_linear, y_non_linear,
    ):
        y = jnp.zeros((len(n),) + y_non_linear.shape[1:])
        non_linear_mask = (n[:, 0] >= self.min_n) & (n[:, 1] >= self.min_n)
        y = y.at[non_linear_mask].set(y_non_linear)
        prev = jnp.arange(len(y))
        prev = prev.at[~non_linear_mask].set(0.0)
        prev = np.maximum.accumulate(prev)
        return y[prev]

    @abstractmethod
    def get_bias_correction(self, n):
        pass

    def __call__(
        self, n, n_non_linear, y_non_linear,
    ):
        y = self.fill_zeroes_with_last(
            n=n, n_non_linear=n_non_linear, y_non_linear=y_non_linear
        )

        bias_correction = self.get_bias_correction(n).reshape(
            (-1,) + (1,) * (y.ndim - 1)
        )
        return bias_correction * y


class XiExtrapolator(Extrapolator):
    def get_bias_correction(self, n):
        return jnp.array(
            [
                self.bias(n_left)
                * self.bias(n_right)
                / self.get_bias(n_left)
                / self.get_bias(n_right)
                for n_left, n_right in n
            ]
        )


class VExtrapolator(Extrapolator):
    def get_bias_correction(self, n):
        return jnp.array(
            [
                (self.bias(n_left) + self.bias(n_right))
                / (self.get_bias(n_left) + self.get_bias(n_right))
                for n_left, n_right in n
            ]
        )
