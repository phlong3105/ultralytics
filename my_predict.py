#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements the prediction scripts for YOLOv8."""

from __future__ import annotations

import socket

import click
import mon

import ultralytics.utils
from ultralytics import YOLO

console = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir = _current_file.parents[0]

ultralytics.utils.DATASETS_DIR = mon.DATA_DIR


# region Predict


def predict(args: dict):
    model = YOLO(args["model"])
    _project = args.pop("project")
    _name = args.pop("name")
    project = f"{_project}/{_name}"
    sources = args.pop("source")
    sources = [sources] if not isinstance(sources, list) else sources
    for source in sources:
        path = mon.Path(source)
        name = path.parent.name if path.name == "images" else path.name
        _ = model(source=source, project=f"{project}", name=name, **args)


# endregion


# region Main


@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root", type=str, default=None, help="Project root.")
@click.option("--config", type=str, default=None, help="Model config.")
@click.option("--weights", type=str, default=None, help="Weights paths.")
@click.option("--model", type=str, default=None, help="Model name.")
@click.option("--data", type=str, default=None, help="Source data directory.")
@click.option("--fullname", type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir", type=str, default=None, help="Optional saving directory.")
@click.option("--device", type=str, default=None, help="Running devices.")
@click.option("--imgsz", type=int, default=None, help="Image sizes.")
@click.option("--conf", type=float, default=None, help="Confidence threshold.")
@click.option("--iou", type=float, default=None, help="IoU threshold.")
@click.option("--max-det", type=int, default=None, help="Max detections per image.")
@click.option("--resize", is_flag=True)
@click.option("--augment", is_flag=True)
@click.option("--agnostic-nms", is_flag=True)
@click.option("--benchmark", is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose", is_flag=True)
def main(
    root: str,
    config: str,
    weights: str,
    model: str,
    data: str,
    fullname: str,
    save_dir: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    resize: bool,
    augment: bool,
    agnostic_nms: bool,
    benchmark: bool,
    save_image: bool,
    verbose: bool,
) -> str:
    hostname = socket.gethostname().lower()

    # Get config args
    config = mon.parse_config_file(project_root=_current_dir.parent / "config", config=config)
    args = mon.load_config(config)

    # Parse arguments
    root = root or args.get("root")
    weights = weights or args.get("weights")
    data = data or args.get("data")
    fullname = fullname or args.get("name")
    device = device or args.get("device")
    imgsz = imgsz or args.get("imgsz")
    conf = conf or args.get("conf")
    iou = iou or args.get("iou")
    max_det = max_det or args.get("max_det")
    augment = augment or args.get("augment")
    agnostic_nms = agnostic_nms or args.get("agnostic_nms")
    verbose = verbose or args.get("verbose")

    # Prioritize input args --> config file args
    root = mon.Path(root)
    weights = mon.to_list(weights)
    weights = weights[0]
    save_dir = save_dir or root / "run" / "train" / model
    save_dir = mon.Path(save_dir)

    # Update arguments
    args["mode"] = "predict"
    args["model"] = weights
    args["project"] = str(save_dir.parent)
    args["name"] = str(save_dir.name)
    args["imgsz"] = imgsz
    args["conf"] = conf
    args["iou"] = iou
    args["max_det"] = max_det
    args["augment"] = augment
    args["agnostic_nms"] = agnostic_nms
    args["batch"] = 1
    args["device"] = device
    args["verbose"] = verbose
    args["source"] = data

    predict(args=args)
    return str(save_dir)


if __name__ == "__main__":
    main()

# endregion
