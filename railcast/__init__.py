import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import logging

import hydra
import pandas as pd

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

import wandb

from railcast.core import CONFIG_PATH
from railcast.utils import count_batches, create_tfrecord_path
from railcast.trainer import Trainer
from railcast.protobuf import load_tfrecord

load_dotenv(override=True)
config_path = os.path.dirname(CONFIG_PATH)
log = logging.getLogger(__name__)


@hydra.main(
    config_path=config_path,
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    if cfg.wandb.mode != "disabled":
        wandb.login(key=cfg.wandb.api_key)
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_object(cfg),
            name=cfg.model.name,
        )

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
    history, results = trainer.fit_and_evaluate(
        train_ds,
        valid_ds,
        test_ds,
        train_steps,
        valid_steps,
        test_steps,
    )

    log.info("History tail:\n%s", pd.DataFrame(history.history).tail())
    log.info("Test results:\n%s", json.dumps(results, indent=3))

    if cfg.wandb.mode != "disabled":
        wandb.finish()

    return trainer.model.keras_model
