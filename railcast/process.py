from functools import partial

import pandas as pd
import tensorflow as tf

from omegaconf import DictConfig

from railcast.core import DATAPATH
from railcast.utils import make_timeseries_from_array

_SeriesT = list[pd.Series | pd.DataFrame]


def _load_series(cfg: DictConfig) -> pd.Series | pd.DataFrame:
    """
    Load pd.DataFrame for analysis.
    Returns scaled univar or multivar series.
    """
    df = pd.read_csv(DATAPATH, parse_dates=["service_date"])
    df.columns = ["date", "day_type", "bus", "rail", "total"]
    df = df.sort_values("date").set_index("date")
    df.drop("total", axis=1, inplace=True)
    df = df.drop_duplicates()

    if cfg.series.is_mulvar:
        df_num = df[["bus", "rail"]] / 1e6
        df_cat = pd.get_dummies(df, columns=["day_type"], dtype=int)

        return pd.concat([df_num, df_cat.iloc[:, 2:]], axis=1)

    return df["rail"] / 1e6


def _split_series(cfg: DictConfig) -> _SeriesT:
    series_full = _load_series(cfg)
    seq_length = cfg.series.seq_length
    train_size = int(len(series_full) * 0.7)
    valid_size = int(len(series_full) * 0.15)

    series_train = series_full[:train_size]
    series_valid = series_full[train_size - seq_length : train_size + valid_size]
    series_test = series_full[train_size + valid_size - seq_length :]

    return series_train, series_valid, series_test


def to_timeseries_dataset(cfg: DictConfig) -> list[tf.data.Dataset]:
    tf.keras.utils.set_random_seed(cfg.seed)

    series_train, series_valid, series_test = _split_series(cfg)
    partial_to_timeseries = partial(
        make_timeseries_from_array,
        seq_length=cfg.series.seq_length,
        steps_ahead=cfg.series.steps_ahead,
    )
    return [
        partial_to_timeseries(series=series)
        for series in (series_train, series_valid, series_test)
    ]
