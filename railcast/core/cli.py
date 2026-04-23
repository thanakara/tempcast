from pathlib import Path
from datetime import datetime

import click

from rich import box
from hydra import compose, initialize_config_dir
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from rich.console import Console
from rich.progress import Progress, TextColumn, SpinnerColumn

import wandb

from railcast.core import CONFIG_PATH
from railcast.utils import create_tfrecord_path
from railcast.process import to_timeseries_dataset
from railcast.protobuf import write_tfrecord

_ = load_dotenv(override=True)

config_dir = str(Path(CONFIG_PATH).parent.absolute())
config_name = str(Path(CONFIG_PATH).name)

theme = Theme(
    {
        "info": "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "dim": "dim white",
    }
)

console = Console(theme=theme)


def get_cfg(overrides: list[str] = None) -> DictConfig:
    """Load hydra config programmatically from within click."""
    if overrides is None:
        overrides = []
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


@click.group
def assets():
    """Railcast core data-records operations."""
    pass


@assets.command(name="make-tfrecords")
@click.option("--mulvar", is_flag=True, default=False, help="Use multivariate series.")
@click.option(
    "--upload",
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
def make_tfrecords(mulvar: bool, upload: bool, wandb_mode: str | None) -> None:
    """
    Write TFRecords for univar or mulvar series.
    Optionally uploads to W&B registry.
    """

    overrides = []
    prefix = "univar"
    if mulvar:
        overrides.append("series.is_mulvar=true")
        prefix = "mulvar"
    if wandb_mode:
        overrides.append(f"++wandb.mode={wandb_mode}")

    cfg = get_cfg(overrides)
    series_mode = "mulvar" if cfg.series.is_mulvar else "univar"

    console.print(
        Panel(
            f"[info]Series mode:[/info] {series_mode}\n"
            f"[info]seq_length:[/info]  {cfg.series.seq_length}\n"
            f"[info]steps_ahead:[/info] {cfg.series.steps_ahead}",
            title="[bold]make-tfrecords[/bold]",
            border_style="cyan",
        )
    )

    train_path = create_tfrecord_path(prefix, "train")
    valid_path = create_tfrecord_path(prefix, "valid")
    test_path = create_tfrecord_path(prefix, "test")

    paths = {
        "rail_train": train_path,
        "rail_valid": valid_path,
        "rail_test": test_path,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Preparing datasets...", total=None)

        train_ds, valid_ds, test_ds = to_timeseries_dataset(cfg)

        for ds, path, name in [
            (train_ds, train_path, "train"),
            (valid_ds, valid_path, "valid"),
            (test_ds, test_path, "test"),
        ]:
            progress.update(task, description=f"Writing [cyan]{name}[/cyan] → {path}")
            write_tfrecord(ds, path, cfg)

    console.print("[success]✓[/success] TFRecords written successfully.")
    for split, path in paths.items():
        console.print(f"  [dim]→[/dim] [white]{split}[/white]: [dim]{path}[/dim]")

    if not upload:
        upload = click.confirm("\nUpload to registry?", default=False)

    if upload:
        _upload_artifact(cfg, paths)
    else:
        console.print("\n[dim]Upload skipped.[/dim]")


def _upload_artifact(cfg: DictConfig, paths: dict[str, str]) -> None:
    """Shared upload logic."""

    mode = cfg.wandb.mode
    series_mode = "mulvar" if cfg.series.is_mulvar else "univar"

    if mode == "disabled":
        console.print(
            Panel(
                "[warning]W&B mode is 'disabled'.[/warning] Set --wandb-mode=online or offline.\n"  # noqa: E501
                "[dim]Example: assets make-tfrecords --upload --wandb-mode=online[/dim]",  # noqa: E501
                border_style="yellow",
            )
        )
        return

    console.print(
        f"\n[info]Uploading to W&B registry[/info] [dim](mode={mode})[/dim]..."
    )

    wandb.login(key=cfg.wandb.api_key)

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        job_type="upload-dataset",
        mode=mode,
    ) as run:
        artifact = wandb.Artifact(
            name=f"cta-tfrecords-{series_mode}",
            type="dataset",
            description=f"CTA-ridership {series_mode} tfrecords",
            metadata=OmegaConf.to_object(cfg.series),
        )

        for split, path in paths.items():
            artifact.add_file(path, name=f"{split}.tfrecord")
            console.print(f"  [dim]→[/dim] added: [white]{path}[/white]")

        run.log_artifact(artifact)

    console.print(
        f"[success]✓[/success] Upload complete: "
        f"[bold]cta-tfrecords-{series_mode}:latest[/bold]"
    )


console = Console()


@assets.command(name="list-cps")
@click.option("--model", default=None, help="Filter by model name")
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["univar", "mulvar"]),
    help="Filter by series mode.",
)
@click.option("--date", default=None, help="Filter by date")
def list_checkpoints(model: str | None, mode: str | None, date: str | None) -> None:  # noqa: C901
    """List available model checkpoints and metadata."""

    overrides = []
    cfg = get_cfg(overrides)

    base = Path(cfg.checkpoint_dir)
    if not base.exists():
        click.echo(f"No checkpoints directory found at: {base.absolute()}")
        return

    runs = sorted(base.iterdir())
    if not runs:
        click.echo("No checkpoints found.")
        return

    results = []
    for run_dir in runs:
        if not run_dir.is_dir():
            continue

        # parse-name: {model_name}_{mode}__{short_id}
        name = run_dir.name
        try:
            left, short_id = name.split("__")
            parts = left.rsplit("_", 1)
            model_name = parts[0]
            series_mode = parts[1]
        except (ValueError, IndexError):
            model_name = name
            series_mode = "unknown"
            short_id = "unknown"

        weights = run_dir / "model.weights.h5"
        wandb_id_file = run_dir / "wandb_run_id"

        modified = (
            datetime.fromtimestamp(weights.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if weights.exists()
            else None
        )

        # apply-filters
        if model and model not in model_name:
            continue
        if mode and mode != series_mode:
            continue
        if date and (not modified or not modified.startswith(date)):
            continue

        results.append(
            {
                "name": name,
                "model": model_name,
                "mode": series_mode,
                "short_id": short_id,
                "wandb_id": wandb_id_file.read_text().strip()
                if wandb_id_file.exists()
                else None,
                "modified": modified,
                "size_kb": round(weights.stat().st_size / 1024, 1)
                if weights.exists()
                else None,
                "weights": weights.exists(),
            }
        )

    if not results:
        click.echo("No checkpoints match the filters.")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="[bold]Model Checkpoints[/bold]",
        title_style="bold white",
    )

    table.add_column("Name", style="white", no_wrap=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Mode", style="magenta", no_wrap=True)
    table.add_column("Short ID", style="yellow", no_wrap=True)
    table.add_column("W&B ID", style="yellow", no_wrap=True)
    table.add_column("Modified", style="dim", no_wrap=True)
    table.add_column("Size (KB)", style="green", justify="right")
    table.add_column("Weights", style="bold", justify="center")

    for e in results:
        table.add_row(
            e["name"],
            e["model"],
            e["mode"],
            e["short_id"],
            e["wandb_id"] or "[dim]N/A[/dim]",
            e["modified"] or "[dim]N/A[/dim]",
            str(e["size_kb"]) if e["size_kb"] else "[dim]N/A[/dim]",
            "[green]✓[/green]" if e["weights"] else "[red]✗[/red]",
        )

    console.print()
    console.print(table)
    console.print(f"[dim]Total: — {len(results)} run(s) — ")
