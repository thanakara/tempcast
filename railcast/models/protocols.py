from typing import Protocol, runtime_checkable  # noqa: F401

import tensorflow as tf


# @runtime_checkable
class Forecaster(Protocol):
    """Contract every railcast model must follow."""

    @property
    def keras_model(self) -> tf.keras.Model: ...

    def build_optimizer(self) -> tf.keras.optimizers.Optimizer: ...

    def build_callbacks(self, extra: list | None) -> list: ...
