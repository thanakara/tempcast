import uuid

from typing import Literal
from pathlib import Path

import pandas as pd
import tensorflow as tf

from omegaconf import DictConfig

from railcast.core import DATASET_URL, DATASETS_DIR
from railcast.protobuf import load_tfrecord


def fetch_ridership_dataset() -> Path:  # TODO: pull: from-registry
    """
    Fetches the dataset using keras.utils. Use this once.
    """
    filepath = tf.keras.utils.get_file(
        "ridership.tgz",
        DATASET_URL,
        cache_dir=".",
        extract=True,
    )

    if "_extracted" in filepath:
        ridership_path = Path(filepath) / "ridership"
    else:
        ridership_path = Path(filepath).with_name("ridership")

    return ridership_path


def create_tfrecord_path(
    prefix: Literal["univar", "mulvar"],
    set_: Literal["train", "valid", "test"],
) -> str:
    tfrecords_dir = Path(DATASETS_DIR) / "tfrecords"
    tfrecords_dir.mkdir(parents=True, exist_ok=True)
    return str(tfrecords_dir.joinpath(f"{prefix}_rail_{set_}.tfrecord"))


def make_timeseries_from_array(
    series: pd.Series | pd.DataFrame, seq_length: int, steps_ahead: int
) -> tf.data.Dataset:
    return tf.keras.utils.timeseries_dataset_from_array(
        series.to_numpy(),
        targets=None,
        sequence_length=seq_length + steps_ahead,
        batch_size=None,
        shuffle=False,
    )


def count_batches(path: str, cfg: DictConfig) -> int:
    return sum(1 for _ in load_tfrecord(path, cfg))


def generate_job_id(cfg: DictConfig) -> str:
    short_id = uuid.uuid4().hex[:8]
    mode = "mulvar" if cfg.series.is_mulvar else "univar"
    return f"{cfg.model.arch.name}_{mode}__{short_id}"
