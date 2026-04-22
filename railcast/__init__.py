import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging

import hydra

from omegaconf import DictConfig

from railcast.core import CONFIG_PATH
from railcast.utils import count_batches, create_tfrecord_path
from railcast.trainer import Trainer
from railcast.protobuf import load_tfrecord

config_path = os.path.dirname(CONFIG_PATH)
log = logging.getLogger(__name__)


@hydra.main(
    config_path=config_path,
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    prefix = "mulvar" if cfg.series.is_mulvar else "univar"
    train_path, valid_path, test_path = [
        create_tfrecord_path(prefix, split) for split in ("train", "valid", "test")
    ]

    train_steps = count_batches(train_path, cfg)
    valid_steps = count_batches(valid_path, cfg)
    test_steps = count_batches(test_path, cfg)

    train_ds = load_tfrecord(train_path, cfg, shuffle=True, repeat=True)
    valid_ds = load_tfrecord(valid_path, cfg, repeat=True)
    test_ds = load_tfrecord(test_path, cfg, repeat=True)

    trainer = Trainer(cfg)
    trainer.fit_and_evaluate(
        train_ds,
        valid_ds,
        test_ds,
        train_steps,
        valid_steps,
        test_steps,
    )

    return trainer.model.keras_model
