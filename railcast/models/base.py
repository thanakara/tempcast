from abc import ABC, abstractmethod

import tensorflow as tf

from omegaconf import DictConfig


class BaseForecaster(ABC):
    def __init__(self, arch: DictConfig, training: DictConfig, series: DictConfig):
        self.arch = arch
        self.train_cfg = training
        self.series = series

    @property
    @abstractmethod
    def keras_model(self) -> tf.keras.Model:
        pass

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

    def build_callbacks(self, extra: list | None) -> list:
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
