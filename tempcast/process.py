from typing import TypedDict
from functools import partial

import pandas as pd
import tensorflow as tf

from omegaconf import DictConfig

from tempcast.core import DATAPATH
from tempcast.utils import make_timeseries_from_array

type _SeriesT = list[pd.Series | pd.DataFrame]


class _Stats(TypedDict):
    mean: float
    std: float


def _normalize(df: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame | pd.Series, _Stats]:
    stats: _Stats = {"mean": df.mean(), "std": df.std()}
    normalized = (df - stats["mean"]) / stats["std"]
    return normalized, stats


def denormalize(
    df: pd.DataFrame | pd.Series, stats: _Stats
) -> pd.DataFrame | pd.Series:
    return df * stats["std"] + stats["mean"]


def get_temp_stats(stats: _Stats, is_mulvar: bool) -> _Stats:
    if is_mulvar:
        return {"mean": stats["mean"]["temp"], "std": stats["std"]["temp"]}
    return stats


def _load_series(cfg: DictConfig) -> tuple[pd.Series | pd.DataFrame, _Stats]:
    df = pd.read_csv(DATAPATH, index_col="datetime", parse_dates=True)

    if cfg.series.is_mulvar:
        series, stats = _normalize(df)
    else:
        series, stats = _normalize(df["temp"])
    return series, stats


def split_series(cfg: DictConfig) -> tuple[_SeriesT, _Stats]:
    series_full, stats = _load_series(cfg)
    seq_length = cfg.series.seq_length
    train_size = int(len(series_full) * 0.7)
    valid_size = int(len(series_full) * 0.15)

    series_train = series_full[:train_size]
    series_valid = series_full[train_size - seq_length : train_size + valid_size]
    series_test = series_full[train_size + valid_size - seq_length :]

    return [series_train, series_valid, series_test], stats


def to_timeseries_dataset(cfg: DictConfig) -> list[tf.data.Dataset]:
    tf.keras.utils.set_random_seed(cfg.seed)

    (series_train, series_valid, series_test), _ = split_series(cfg)
    partial_to_timeseries = partial(
        make_timeseries_from_array,
        seq_length=cfg.series.seq_length,
        steps_ahead=cfg.series.steps_ahead,
    )
    return [
        partial_to_timeseries(series=series)
        for series in (series_train, series_valid, series_test)
    ]
