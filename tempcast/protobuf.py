from functools import partial

import numpy as np
import tensorflow as tf

from omegaconf import DictConfig


def _float_feature(value: float | np.floating) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def make_sequence_example_for_univar(
    window: tf.Tensor,
) -> tf.train.SequenceExample:
    timesteps = window.numpy().tolist()

    return tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                "length": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(timesteps)])
                )
            }
        ),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                "window": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in timesteps]
                )
            }
        ),
    )


def _parse_sequence_example_univar(serialized_record: tf.Tensor) -> tf.Tensor:
    context_spec = {"length": tf.io.FixedLenFeature([], tf.int64)}
    sequence_spec = {"window": tf.io.FixedLenSequenceFeature([], tf.float32)}

    _, sequence = tf.io.parse_single_sequence_example(
        serialized_record,
        context_features=context_spec,
        sequence_features=sequence_spec,
    )

    return sequence["window"]


def _split_window_univar(
    window: tf.Tensor, cfg: DictConfig
) -> tuple[tf.Tensor, tf.Tensor]:
    return window[: -cfg.series.steps_ahead], window[-cfg.series.steps_ahead :]


def make_sequence_example_for_mulvar(
    window: tf.Tensor,
) -> tf.train.SequenceExample:
    w = window.numpy()

    return tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                "length": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(w)])
                ),
                "n_features": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[w.shape[-1]])
                ),
            }
        ),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                "temp": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 0]]
                ),
                "humidity": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 1]]
                ),
                "precip": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 2]]
                ),
                "windspeed": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 3]]
                ),
                "cloudcover": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 4]]
                ),
                "solarradiation": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 5]]
                ),
            }
        ),
    )


def _parse_sequence_example_mulvar(serialized_record: tf.Tensor) -> tf.Tensor:
    context_spec = {
        "length": tf.io.FixedLenFeature([], tf.int64),
        "n_features": tf.io.FixedLenFeature([], tf.int64),
    }
    sequence_spec = {
        "temp": tf.io.FixedLenSequenceFeature([], tf.float32),
        "humidity": tf.io.FixedLenSequenceFeature([], tf.float32),
        "precip": tf.io.FixedLenSequenceFeature([], tf.float32),
        "windspeed": tf.io.FixedLenSequenceFeature([], tf.float32),
        "cloudcover": tf.io.FixedLenSequenceFeature([], tf.float32),
        "solarradiation": tf.io.FixedLenSequenceFeature([], tf.float32),
    }

    _, sequences = tf.io.parse_single_sequence_example(
        serialized_record,
        context_features=context_spec,
        sequence_features=sequence_spec,
    )

    window = tf.stack(
        [
            sequences["temp"],
            sequences["humidity"],
            sequences["precip"],
            sequences["windspeed"],
            sequences["cloudcover"],
            sequences["solarradiation"],
        ],
        axis=-1,
    )
    return window


def _split_window_mulvar(
    window: tf.Tensor, cfg: DictConfig
) -> tuple[tf.Tensor, tf.Tensor]:
    X = window[: -cfg.series.steps_ahead]
    y = window[-cfg.series.steps_ahead :, cfg.series.target_idx]  # temp

    return X, y


def write_tfrecord(dataset: tf.Tensor, path: str, cfg: DictConfig) -> None:
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(path, options) as writer:
        for window in dataset:
            if cfg.series.is_mulvar:
                seq_example = make_sequence_example_for_mulvar(window)
            else:
                seq_example = make_sequence_example_for_univar(window)
            writer.write(seq_example.SerializeToString())


def load_tfrecord(
    path: str,
    cfg: DictConfig,
    shuffle: bool = False,
    repeat: bool = False,
) -> tf.data.TFRecordDataset:
    ds = tf.data.TFRecordDataset(path, compression_type="GZIP")
    split_mulvar = partial(_split_window_mulvar, cfg=cfg)
    split_univar = partial(_split_window_univar, cfg=cfg)

    if cfg.series.is_mulvar:
        ds = ds.map(_parse_sequence_example_mulvar).map(split_mulvar)
    else:
        ds = ds.map(_parse_sequence_example_univar).map(split_univar)

    if shuffle:
        ds = ds.shuffle(buffer_size=500)

    if repeat:
        ds = ds.repeat()
    return ds.batch(cfg.model.training.batch_size)
