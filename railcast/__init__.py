import logging

from pathlib import Path

import hydra

from omegaconf import OmegaConf, DictConfig

config_path = Path("conf").as_posix()
log = logging.getLogger(__name__)


@hydra.main(
    config_path=config_path,
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    log.debug("__on_job_start__")
    log.info("\n%s", OmegaConf.to_yaml(cfg))
    log.debug("__on_job_end__")
