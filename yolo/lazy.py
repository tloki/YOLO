import sys
from pathlib import Path
from typing import cast

import hydra

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config, TrainConfig, InferenceConfig
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.solver import ModelTester, ModelTrainer, ModelValidator
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.deploy_utils import FastModelLoader
from yolo.utils.logging_utils import ProgressLogger
from yolo.utils.model_utils import get_device


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    progress = ProgressLogger(cfg, exp_name=cfg.name)
    device, use_ddp = get_device(cfg.device)
    dataloader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task, use_ddp)
    if getattr(cfg.task, "fast_inference", False):
        model = FastModelLoader(cfg).load_model(device)
    else:
        model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
        model = model.to(device)

    converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, device)

    if cfg.task.task == "train":
        solver = ModelTrainer(cast(cfg, TrainConfig), model, converter, progress, device, use_ddp)
    elif cfg.task.task == "validation":
        solver = ModelValidator(cfg.task, cfg.dataset, model, converter, progress, device)
    elif cfg.task.task == "inference":
        solver = ModelTester(cast(cfg, InferenceConfig), model, converter, progress, device)
    else:
        raise ValueError("unsupported task " + str(cfg.task.task))

    progress.start()
    solver.solve(dataloader)


if __name__ == "__main__":
    main()
