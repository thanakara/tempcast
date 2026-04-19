from pathlib import Path

import pandas as pd
import tensorflow as tf

from railcast.core import DATASET_URL


def fetch_ridership_dataset() -> Path:  # TODO: pull: from-registry
    """
    Fetches the dataset using keras.utils. Use this once.
    """
    filepath = tf.keras.utils.get_file(
        "ridership.tgz",
        DATASET_URL,
        cache_dir="railcast/",
        extract=True,
    )

    if "_extracted" in filepath:
        ridership_path = Path(filepath) / "ridership"
    else:
        ridership_path = Path(filepath).with_name("ridership")

    return ridership_path


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
