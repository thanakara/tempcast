import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

from pathlib import Path

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig

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
    log.info("on_job_start")
    path = Path(cfg.series.tfrecord_dir) / "mulvar_rail_test.tfrecord"
    test_ds = load_tfrecord(str(path), cfg)

    for X_batch, y_batch in test_ds.take(1):
        log.debug("inputs-shape:%s", X_batch.shape)
        log.debug("targets-shape:%s", y_batch.shape)
    log.info("on_job_end")
