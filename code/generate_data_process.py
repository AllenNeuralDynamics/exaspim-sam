#!/usr/bin/env python3
"""Generate AIND DataProcess metadata for the whole brain masking run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aind_data_schema.core.processing import Code, DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName


REPO_URL = "https://github.com/AllenNeuralDynamics/exaspim-sam"
REPO_BRANCH = "main"
RUN_SCRIPT = "code/run"
DEFAULT_EXPERIMENTER = "Cameron Arshadi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate /results/data_process.json for whole brain masking."
    )
    parser.add_argument("--s3-zarr-path", required=True)
    parser.add_argument("--res-488", required=True, type=int)
    parser.add_argument("--res-561", required=True, type=int)
    parser.add_argument("--inference-channel", required=True, choices=["488", "561"])
    parser.add_argument("--start-date-time", required=True)
    parser.add_argument("--end-date-time", required=True)
    parser.add_argument("--run-status", required=True, choices=["success", "failed"])
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--output-json", default="/results/data_process.json")
    parser.add_argument(
        "--metadata-yml",
        default=str(Path(__file__).resolve().parents[1] / "metadata" / "metadata.yml"),
    )
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_remote_main_sha(repo_url: str = REPO_URL, branch: str = REPO_BRANCH) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repo_url, f"refs/heads/{branch}"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"No remote SHA found for {repo_url} refs/heads/{branch}")
    return output.split()[0]


def load_metadata_yaml(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}

    try:
        import yaml

        with metadata_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return load_metadata_yaml_fallback(metadata_path)


def load_metadata_yaml_fallback(metadata_path: Path) -> dict[str, Any]:
    """Parse the simple Code Ocean metadata.yml author shape without PyYAML."""
    authors: list[dict[str, str]] = []
    current_author: dict[str, str] | None = None

    with metadata_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("- name:"):
                name = line.split(":", 1)[1].strip()
                current_author = {"name": name}
                authors.append(current_author)
            elif current_author is not None and line.startswith("name:"):
                current_author["name"] = line.split(":", 1)[1].strip()

    return {"authors": authors} if authors else {}


def get_experimenters(metadata_path: Path) -> list[str]:
    metadata = load_metadata_yaml(metadata_path)
    experimenters: list[str] = []

    for author in metadata.get("authors", []):
        if isinstance(author, dict) and author.get("name"):
            experimenters.append(str(author["name"]))
        elif isinstance(author, str):
            experimenters.append(author)

    return experimenters or [DEFAULT_EXPERIMENTER]


def build_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "s3_zarr_path": args.s3_zarr_path,
        "res_488": args.res_488,
        "res_561": args.res_561,
        "inference_channel": args.inference_channel,
        "raw_save_dir": "/results/raw",
        "mask_save_dir": "/results/mask",
        "pred_save_dir": "/results/pred",
        "postprocessed_save_dir": "/results/postprocessed",
        "preprocess": {
            "channels": ["488", "561"],
            "gaussian_sigma": 2,
            "threshold_method": "fixed",
            "fixed_threshold": 1,
        },
        "inference": {
            "medsam_checkpoint": "/data/medsam_best.pth",
            "medsam_model_type": "vit_b",
            "device": "cuda:0",
            "batch_size": 4,
            "use_amp": True,
            "use_torch_compile": True,
            "bbox_prompt_type": "whole",
        },
        "postprocess": {
            "input_path": "/results/pred",
            "output_dir": "/results/postprocessed",
            "raw_base_dir": "/results/raw",
            "kernel_size": 3,
        },
    }


def write_data_process(args: argparse.Namespace) -> Path:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    data_process = DataProcess(
        process_type=ProcessName.OTHER,
        stage=ProcessStage.PROCESSING,
        name="Whole brain masking",
        code=Code(
            url=REPO_URL,
            name="exaspim-sam",
            version=get_remote_main_sha(),
            run_script=RUN_SCRIPT,
            language="Python",
            parameters=build_parameters(args),
        ),
        experimenters=get_experimenters(Path(args.metadata_yml)),
        start_date_time=parse_datetime(args.start_date_time),
        end_date_time=parse_datetime(args.end_date_time),
        output_path="/results/postprocessed",
        notes=(
            "Whole brain masking pipeline using preprocessing, MedSAM inference, "
            f"and mask postprocessing. Run status: {args.run_status}; "
            f"exit code: {args.exit_code}."
        ),
    )

    output_json.write_text(data_process.model_dump_json(indent=3), encoding="utf-8")
    return output_json


def main() -> int:
    args = parse_args()
    try:
        output_json = write_data_process(args)
    except Exception as exc:
        print(f"ERROR: Failed to generate data_process.json: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote DataProcess metadata to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
