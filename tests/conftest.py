import sys
from pathlib import Path
from typing import List, cast

import pytest
import torch
from hydra import compose, initialize
from torch import device

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo import Anc2Box, Config, Vec2Box, create_converter, create_model
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import StreamDataLoader, YoloDataLoader
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils.logging_utils import ProgressLogger, set_seed


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "requires_cuda: mark test to run only if CUDA is available")


def get_cfg(overrides: List=[]) -> Config:
    config_path = "../yolo/config"
    with initialize(config_path=config_path, version_base=None):
        cfg: Config = compose(config_name="config", overrides=overrides)
        set_seed(cfg.lucky_number)
        return cfg


@pytest.fixture(scope="session")
def train_cfg() -> Config:
    return get_cfg(overrides=["task=train", "dataset=mock"])


@pytest.fixture(scope="session")
def validation_cfg() -> Config:
    return get_cfg(overrides=["task=validation", "dataset=mock"])


@pytest.fixture(scope="session")
def inference_cfg() -> Config:
    return get_cfg(overrides=["task=inference", "task.data.source='../demo/images/inference/image.png'"])


@pytest.fixture(scope="session")
def inference_v7_cfg() -> Config:
    return get_cfg(overrides=["task=inference", "model=v7"])


@pytest.fixture(scope="session")
def device() -> device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def train_progress_logger(train_cfg: Config) -> ProgressLogger:
    progress_logger = ProgressLogger(train_cfg, exp_name=train_cfg.name)
    return progress_logger


@pytest.fixture(scope="session")
def validation_progress_logger(validation_cfg: Config) -> ProgressLogger:
    progress_logger = ProgressLogger(validation_cfg, exp_name=validation_cfg.name)
    return progress_logger


@pytest.fixture(scope="session")
def model(train_cfg: Config, device: torch.device) -> YOLO:
    model = create_model(train_cfg.model)
    return model.to(device)


@pytest.fixture(scope="session")
def model_v7(inference_v7_cfg: Config, device) -> YOLO:
    model = create_model(inference_v7_cfg.model)
    return model.to(device)


@pytest.fixture(scope="session")
def vec2box(train_cfg: Config, model: YOLO, device: torch.device) -> Vec2Box:
    vec2box = create_converter(train_cfg.model.name, model, train_cfg.model.anchor, train_cfg.image_size, device)
    return cast(Vec2Box, vec2box)


@pytest.fixture(scope="session")
def anc2box(inference_v7_cfg: Config, model: YOLO, device: torch.device) -> Anc2Box:
    anc2box = create_converter(
        inference_v7_cfg.model.name, model, inference_v7_cfg.model.anchor, inference_v7_cfg.image_size, device
    )
    return cast(Anc2Box, anc2box)


@pytest.fixture(scope="session")
def train_dataloader(train_cfg: Config) -> YoloDataLoader:
    prepare_dataset(train_cfg.dataset, task="train")
    return YoloDataLoader(train_cfg.task.data, train_cfg.dataset, train_cfg.task.task)


@pytest.fixture(scope="session")
def validation_dataloader(validation_cfg: Config) -> YoloDataLoader:
    prepare_dataset(validation_cfg.dataset, task="val")
    return YoloDataLoader(validation_cfg.task.data, validation_cfg.dataset, validation_cfg.task.task)


@pytest.fixture(scope="session")
def file_stream_data_loader(inference_cfg: Config) -> StreamDataLoader:
    return StreamDataLoader(inference_cfg.task.data)


@pytest.fixture(scope="session")
def file_stream_data_loader_v7(inference_v7_cfg: Config) -> StreamDataLoader:
    return StreamDataLoader(inference_v7_cfg.task.data)


@pytest.fixture(scope="session")
def directory_stream_data_loader(inference_cfg: Config) -> StreamDataLoader:
    inference_cfg.task.data.source = "tests/data/images/train"
    return StreamDataLoader(inference_cfg.task.data)
