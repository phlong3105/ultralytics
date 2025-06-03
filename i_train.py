#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Ultralytics".

References:
    - https://github.com/ultralytics/ultralytics
"""

import mon
from ultralytics import YOLO
from ultralytics import settings

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    torchrun     = args["torchrun"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    seed         = args["seed"]
    batch_size   = args["batch_size"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_result  = args["save_result"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    save_nearby  = args["save_nearby"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]

    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")

    # Device
    device = mon.set_device(device)

    # Seed
    mon.set_random_seed(seed)

    # Data I/O
    settings.update({"datasets_dir": str(root)})

    # Trainer
    opts         = args["options"]
    opts["mode"] = "train"

    weights    = weights[0] if isinstance(weights, list | tuple) else weights
    weights    = mon.parse_weights_file(root, weights)    if weights else None
    pretrained = opts["pretrained"]
    pretrained = mon.parse_weights_file(root, pretrained) if isinstance(pretrained, str) else None
    if weights and weights.is_weights_file(exist=True):
        opts["model"]      = weights
    elif pretrained and pretrained.is_weights_file(exist=True):
        opts["model"]      = pretrained
        opts["pretrained"] = True
    else:
        opts["pretrained"] = True

    if not mon.Path(opts["data"]).is_yaml_file():
        opts["data"] = str(current_dir / "ultralytics" / "cfg" / "datasets" / f"{opts["data"]}")

    opts["epochs"]   = epochs
    opts["device"]   = device
    opts["project"]  = str(save_dir.parent)
    opts["name"]     = str(save_dir.name)
    opts["exist_ok"] = exist_ok
    opts["verbose"]  = verbose
    opts["seed"]     = seed
    opts["save_txt"] = save_result

    # Training
    model = YOLO(opts["model"])
    model.info()
    _ = model.train(**opts)


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
