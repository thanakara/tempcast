import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig

import wandb

from railcast.core import CONFIG_PATH
from railcast.protobuf import load_tfrecord

_ = load_dotenv(override=True)
config_path = os.path.dirname(CONFIG_PATH)
log = logging.getLogger(__name__)


@hydra.main(
    config_path=config_path,
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    log.debug("__on_job_start__")
    log.warning(f"wandb.mode: {cfg.wandb.mode}")
    if cfg.wandb.mode != "online":
        raise KeyError(f"mode set to {cfg.wandb.mode}")

    wandb.login(key=cfg.wandb.api_key)

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        job_type="inspect",
    ) as run:
        artifact = run.use_artifact("cta-tfrecords-univar:latest")
        artifact_dir = artifact.download()
        train_path = os.path.join(artifact_dir, "rail_train.tfrecord")
        log.info("Loading: %s", train_path)

        train_ds = load_tfrecord(train_path, cfg, shuffle=True)

        for X_batch, y_batch in train_ds.take(5):
            log.info(X_batch.shape)
            log.info(y_batch.shape)
    log.debug("__on_job_end__")
