import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

import hydra

from omegaconf import DictConfig

from railcast.core import CONFIG_PATH
from railcast.process import to_timeseries_dataset

config_path = os.path.dirname(CONFIG_PATH)
log = logging.getLogger(__name__)


@hydra.main(
    config_path=config_path,
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    *_, test_ds = to_timeseries_dataset(cfg)
    log.info(test_ds.__class__.__name__)

    for sequence in test_ds.take(1):
        log.info("seq_shape: %s", sequence.shape)
