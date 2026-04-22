from functools import partial

import tensorflow as tf

from omegaconf import DictConfig


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def make_sequence_example_for_univar(
    window: tf.data.Dataset,
) -> tf.train.SequenceExample:
    timesteps = window.numpy().tolist()  # list-of: 70_floats

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


def _parse_sequence_example_univar(serialized_record):  # TODO: type-hint
    context_spec = {"length": tf.io.FixedLenFeature([], tf.int64)}
    sequence_spec = {"window": tf.io.FixedLenSequenceFeature([], tf.float32)}

    _, sequence = tf.io.parse_single_sequence_example(
        serialized_record,
        context_features=context_spec,
        sequence_features=sequence_spec,
    )

    return sequence["window"]


def _split_window_univar(window: tf.data.Dataset, cfg: DictConfig):
    return window[: -cfg.series.steps_ahead], window[-cfg.series.steps_ahead :]


def make_sequence_example_for_mulvar(
    window: tf.data.Dataset,
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
                "bus": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 0]]
                ),
                "rail": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 1]]
                ),
                "day_type_A": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 2]]
                ),
                "day_type_U": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 3]]
                ),
                "day_type_W": tf.train.FeatureList(
                    feature=[_float_feature(v) for v in w[:, 4]]
                ),
            }
        ),
    )


def _parse_sequence_example_mulvar(serialized_record):
    context_spec = {
        "length": tf.io.FixedLenFeature([], tf.int64),
        "n_features": tf.io.FixedLenFeature([], tf.int64),
    }
    sequence_spec = {
        "bus": tf.io.FixedLenSequenceFeature([], tf.float32),
        "rail": tf.io.FixedLenSequenceFeature([], tf.float32),
        "day_type_A": tf.io.FixedLenSequenceFeature([], tf.float32),
        "day_type_U": tf.io.FixedLenSequenceFeature([], tf.float32),
        "day_type_W": tf.io.FixedLenSequenceFeature([], tf.float32),
    }

    _, sequences = tf.io.parse_single_sequence_example(
        serialized_record,
        context_features=context_spec,
        sequence_features=sequence_spec,
    )

    window = tf.stack(
        [
            sequences["bus"],
            sequences["rail"],
            sequences["day_type_A"],
            sequences["day_type_U"],
            sequences["day_type_W"],
        ],
        axis=-1,
    )
    return window


def _split_window_mulvar(window: tf.data.Dataset, cfg: DictConfig):
    X = window[: -cfg.series.steps_ahead]
    y = window[-cfg.series.steps_ahead :, 1]  # rail

    return X, y


def write_tfrecord(dataset: tf.data.Dataset, path: str, cfg: DictConfig) -> None:
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
) -> tf.data.Dataset:
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
