import os
import uuid

from typing import Literal
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import requests
import tensorflow as tf

from dotenv import load_dotenv
from omegaconf import DictConfig

from tempcast.core import DATAPATH, DATASETS_DIR
from tempcast.protobuf import load_tfrecord

load_dotenv(override=True)
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
CSV_PATH = Path(DATAPATH)


def fetch_day(target_date: date) -> pd.DataFrame:
    r = requests.get(
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Athens,GR/{target_date}/{target_date}",
        params={
            "unitGroup": "metric",
            "include": "hours",
            "elements": "datetime,temp,humidity,windspeed,solarradiation,precip,cloudcover",  # noqa: E501
            "contentType": "json",
            "key": API_KEY,
        },
    )
    if r.status_code != 200:  # noqa: PLR2004
        raise Exception(f"API error {r.status_code}: {r.text[:200]}")

    rows = []
    for day in r.json()["days"]:
        for hour in day["hours"]:
            hour["datetime"] = f"{day['datetime']}T{hour['datetime']}"
            rows.append(hour)

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%dT%H:%M:%S")
    return df.set_index("datetime")


def update_dataset() -> pd.DataFrame:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, index_col="datetime", parse_dates=True)
        last_date = df.index.max().date()
    else:
        df = pd.DataFrame()
        last_date = date.today() - timedelta(days=31)  # cold start

    yesterday = date.today() - timedelta(days=1)

    missing = []
    cursor = last_date + timedelta(days=1)
    while cursor <= yesterday:
        missing.append(cursor)
        cursor += timedelta(days=1)

    if not missing:
        print("Already up to date.")
        return df

    print(f"Fetching {len(missing)} missing day(s): {missing[0]} → {missing[-1]}")

    new_dfs = []
    for d in missing:
        try:
            new_dfs.append(fetch_day(d))
            print(f"  ✓ {d}")
        except Exception as e:
            print(f"  ✗ {d}: {e}")
            continue

    if new_dfs:
        df = pd.concat([df] + new_dfs)
        df = df[~df.index.duplicated(keep="last")]  # safety dedup
        df.sort_index(inplace=True)
        df.to_csv(CSV_PATH)
        print(f"Saved → {len(df)} total rows ({df.index.min()} → {df.index.max()})")
    else:
        print("Warning: no new data fetched. Dataset unchanged.")

    return df


def create_tfrecord_path(
    prefix: Literal["univar", "mulvar"],
    set_: Literal["train", "valid", "test"],
) -> str:
    tfrecords_dir = Path(DATASETS_DIR) / "tfrecords"
    tfrecords_dir.mkdir(parents=True, exist_ok=True)
    return str(tfrecords_dir.joinpath(f"{prefix}_temp_{set_}.tfrecord"))


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


def generate_job_id(model: str, is_mulvar: bool) -> str:
    """Use this once @on_job_start; then set the state."""
    short_id = uuid.uuid4().hex[:8]
    mode = "mulvar" if is_mulvar else "univar"
    return f"{model}_{mode}__{short_id}"


def reconstruct_job_id(cfg: DictConfig, run_id: str) -> str:
    mode = "mulvar" if cfg.series.is_mulvar else "univar"
    return f"{cfg.model.arch.name}_{mode}__{run_id}"


def save_run_id(run_id: str, checkpoint_dir: Path) -> None:
    path = checkpoint_dir / "wandb_run_id"
    path.write_text(run_id)


def load_run_id(checkpoint_id: Path) -> str | None:
    path = checkpoint_id / "wandb_run_id"
    if path.exists():
        return path.read_text().strip()
    return None


def get_checkpoint_dir(cfg: DictConfig, job_id: str) -> Path:
    return Path(cfg.checkpoint_dir) / job_id
