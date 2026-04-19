from pathlib import Path

import click

from hydra import compose, initialize_config_dir
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

import wandb

from railcast.core import CONFIG_PATH
from railcast.utils import create_tfrecord_path
from railcast.process import to_timeseries_dataset
from railcast.protobuf import write_tfrecord

_ = load_dotenv(override=True)

config_dir = str(Path(CONFIG_PATH).parent.absolute())
config_name = str(Path(CONFIG_PATH).name)


def get_cfg(overrides: list[str] = None) -> DictConfig:
    """Load hydra config programmatically from within click."""
    if overrides is None:
        overrides = []
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


@click.group
def records():
    """Railcast core data-records operations."""
    pass


@records.command()
@click.option(
    "--mulvar",
    type=click.BOOL,
    is_flag=True,
    default=False,
    help="Use multivariate series.",
)
@click.option(
    "--upload",
    type=click.BOOL,
    is_flag=True,
    default=False,
    help="Upload to W&B registry after writing.",
)
@click.option(
    "--wandb-mode",
    type=click.Choice(["online", "offline", "disabled"]),
    default=None,
    help="Override W&B mode from config.",
)
def write(mulvar: bool, upload: bool, wandb_mode: str | None) -> None:
    """
    Write TFRecords for univar or mulvar series.
    Optionally uploads to Weights & Biases registry.
    """

    overrides = []
    prefix = "univar"
    if mulvar:
        overrides.append("series.is_mulvar=true")
        prefix = "mulvar"
    if wandb_mode:
        overrides.append(f"++wandb.mode={wandb_mode}")  # inject: hydra-compose

    cfg = get_cfg(overrides)
    click.echo(f"\nMode: {'mulvar' if cfg.series.is_mulvar else 'univar'}")

    click.echo("Writing records. Please hold...")
    train_ds, valid_ds, test_ds = to_timeseries_dataset(cfg)
    train_path = create_tfrecord_path(prefix, "train")
    valid_path = create_tfrecord_path(prefix, "valid")
    test_path = create_tfrecord_path(prefix, "test")

    paths = {
        "rail_train": train_path,
        "rail_valid": valid_path,
        "rail_test": test_path,
    }

    write_tfrecord(train_ds, train_path, cfg)
    write_tfrecord(valid_ds, valid_path, cfg)
    write_tfrecord(test_ds, test_path, cfg)

    if not upload:
        upload = click.confirm("\nUpload to registry?", default=False)

    if upload:
        _upload_artifact(cfg, paths)
    else:
        click.echo("Upload skipped.")


def _upload_artifact(cfg: DictConfig, paths: dict[str, str]) -> None:
    """Shared upload logic across `make` and `upload`"""

    mode = cfg.wandb.mode

    if mode == "disabled":
        click.echo("\nW&B mode is 'disabled' in config. Skipping upload.")
        click.echo(
            "Set: --wandb-mode=online/offline to direct-upload/local-run & sync after."
        )
        click.echo("Example: records write --mulvar --upload --wandb-mode=offline")

    click.echo("\nUploading to W&B registry...")

    wandb.login(key=cfg.wandb.api_key)

    series_mode = "mulvar" if cfg.series.is_mulvar else "univar"

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        job_type="upload-dataset",
        mode=mode,
    ) as run:
        artifact = wandb.Artifact(
            name=f"cta-tfrecords-{series_mode}",
            type="dataset",
            description=f"CTA-rideship {series_mode} tfrecords",
            metadata=OmegaConf.to_object(cfg.series),
        )

        for split, path in paths.items():
            artifact.add_file(path, name=f"{split}.tfrecord")
            click.echo(f"       added: {path}")

        run.log_artifact(artifact)

    click.echo(f"Upload complete: cta-tfrecords-{series_mode}:latest")
