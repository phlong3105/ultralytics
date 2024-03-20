#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the training scripts for YOLOv8."""

from __future__ import annotations

import socket

import click

import ultralytics.utils
from mon import core, DATA_DIR
from ultralytics import YOLO

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]

ultralytics.utils.DATASETS_DIR = DATA_DIR


# region Train

def train(args: dict):
    model = YOLO(args["model"])
    _     = model.train(**args)

# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",     type=str,   default=None, help="Project root.")
@click.option("--config",   type=str,   default=None, help="Model config.")
@click.option("--weights",  type=str,   default=None, help="Weights paths.")
@click.option("--model",    type=str,   default=None, help="Model name.")
@click.option("--fullname", type=str,   default=None, help="Save results to root/run/train/fullname.")
@click.option("--save-dir", type=str,   default=None, help="Optional saving directory.")
@click.option("--device",   type=str,   default=None, help="Running devices.")
@click.option("--epochs",   type=int,   default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",    type=int,   default=None, help="Stop training once this number of steps is reached.")
@click.option("--conf",     type=float, default=None, help="Confidence threshold.")
@click.option("--iou",      type=float, default=None, help="IoU threshold.")
@click.option("--max-det",  type=int,   default=None, help="Max detections per image.")
@click.option("--exist-ok", is_flag=True)
@click.option("--verbose",  is_flag=True)
def main(
    root    : str,
    config  : str,
    weights : str,
    model   : str,
    fullname: str,
    save_dir: str,
    device  : str,
    epochs  : int,
    steps   : int,
    conf    : float,
    iou     : float,
    max_det : int,
    exist_ok: bool,
    verbose : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = core.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root     or args["root"]
    weights  = weights  or args["model"]
    data     = args["data"]
    fullname = fullname or args["name"]
    device   = device   or args["device"]
    epochs   = epochs   or args["epochs"]
    conf     = conf     or args["conf"]
    iou      = iou      or args["iou"]
    max_det  = max_det  or args["max_det"]
    exist_ok = exist_ok or args["exist_ok"]
    verbose  = verbose  or args["verbose"]
    
    # Parse arguments
    root     = core.Path(root)
    weights  = core.to_list(weights)
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data     = core.Path(data)
    data     = data  if data.exists() else _current_dir / "data" / data.name
    data     = str(data.config_file())
    save_dir = save_dir or root / "run" / "train" / fullname
    save_dir = core.Path(save_dir)
    
    # Update arguments
    args["mode"]     = "train"
    args["model"]    = weights
    args["data"]     = data
    args["project"]  = str(save_dir.parent)
    args["name"]     = str(save_dir.name)
    args["epochs"]   = epochs
    args["conf"]     = conf
    args["iou"]      = iou
    args["max_det"]  = max_det
    args["device"]   = device
    args["exist_ok"] = exist_ok
    args["verbose"]  = verbose
    
    if not exist_ok:
        core.delete_dir(paths=core.Path(save_dir))
    
    train(args=args)
    return str(save_dir)
    

if __name__ == "__main__":
    main()

# endregion
