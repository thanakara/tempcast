from abc import ABC, abstractmethod
from functools import cached_property

import tensorflow as tf

from omegaconf import DictConfig


class BaseForecaster(ABC):
    def __init__(self, arch: DictConfig, training: DictConfig, series: DictConfig):
        self.arch = arch
        self.train_cfg = training
        self.series = series
        self.n_features = len(series.features) if self.series.is_mulvar else 1
        self.inputs = tf.keras.layers.Input(shape=[None, self.n_features])

    @abstractmethod
    def _build_keras_model(self) -> tf.keras.Model:
        """Build and return keras.Model using FunctionalAPI."""
        raise NotImplementedError

    @cached_property
    def keras_model(self) -> tf.keras.Model:
        return self._build_keras_model()

    def build_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
        }
        cls = optimizers.get(self.train_cfg.optimizer)
        if cls is None:
            raise ValueError(f"Unknown optimizer: {self.train_cfg.optimizer}")
        return cls(learning_rate=self.train_cfg.lr)

    def build_callbacks(
        self, extra: list[tf.keras.callbacks.Callback] | None
    ) -> list[tf.keras.callbacks.Callback]:
        if extra is None:
            extra = []

        callbacks = []

        if self.train_cfg.early_stopping:
            es = self.train_cfg.early_stopping
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=es.monitor,
                    patience=es.patience,
                    min_delta=es.min_delta,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        return callbacks + extra
