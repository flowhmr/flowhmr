import os
import argparse

from flowhmr.core.framework.loaders import read_yaml, load_module


def trim_ymlname(ymlname):
    return os.path.basename(ymlname).split(".")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2M Training")
    parser.add_argument("--model", type=str, required=True, help="Model config YAML")
    parser.add_argument("--data", type=str, required=True, help="Data config YAML")
    parser.add_argument("--train", type=str, required=True, help="Train config YAML")
    parser.add_argument("--postfix", type=str, default=None, help="Experiment name postfix")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from state dir (e.g. output/.../state_epoch24) or checkpoint file (e.g. output/.../epoch24.ckpt)")
    args = parser.parse_args()

    cfg = {}
    cfg.update(read_yaml(args.model))
    cfg.update(read_yaml(args.data))
    cfg.update(read_yaml(args.train))

    if args.name is not None:
        expname = args.name
    else:
        expname = f"output/v2m_train/{trim_ymlname(args.model)}_{trim_ymlname(args.data)}_{trim_ymlname(args.train)}"
    if args.postfix is not None:
        expname += f"_{args.postfix}"
    cfg["exp"] = expname

    trainer_cls_path = cfg.get("trainer", "flowhmr/trainer/v2m_trainer.V2MTrainer")
    TrainerCls = load_module(trainer_cls_path)
    trainer = TrainerCls(cfg)
    trainer.fit(resume=args.resume)