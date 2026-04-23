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
from railcast.models.base import TrainerProtocol

config_path = os.path.dirname(CONFIG_PATH)
log = logging.getLogger(__name__)


@hydra.main(
    config_path=config_path,
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    splits = ("train", "valid", "test")
    prefix = "mulvar" if cfg.series.is_mulvar else "univar"
    paths = [create_tfrecord_path(prefix, split) for split in splits]
    steps = [count_batches(path, cfg) for path in paths]

    train_ds = load_tfrecord(paths[0], cfg, shuffle=True, repeat=True)
    valid_ds = load_tfrecord(paths[1], cfg, repeat=True)
    test_ds = load_tfrecord(paths[2], cfg, repeat=True)

    trainer = Trainer(cfg)
    assert isinstance(trainer, TrainerProtocol)

    _ = trainer.fit_and_evaluate(
        train_ds,
        valid_ds,
        test_ds,
        *steps,
    )

    return trainer.model.keras_model
