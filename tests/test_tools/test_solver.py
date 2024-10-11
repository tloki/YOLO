import sys
from pathlib import Path

import pytest
import torch
from torch import allclose, tensor

from yolo import ProgressLogger

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config, TrainConfig
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import StreamDataLoader, YoloDataLoader
from yolo.tools.solver import ModelTester, ModelTrainer, ModelValidator
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box


def test_model_validator_initialization(model_validator: ModelValidator) -> None:
    assert isinstance(model_validator.model, YOLO)
    assert hasattr(model_validator, "solve")


@pytest.fixture
def model_validator(validation_cfg: Config, model: YOLO, vec2box: Vec2Box, validation_progress_logger: ProgressLogger, device: torch.device) -> ModelValidator:
    validator = ModelValidator(
        validation_cfg.task, validation_cfg.dataset, model, vec2box, validation_progress_logger, device
    )
    return validator


def test_model_validator_solve_mock_dataset(model_validator: ModelValidator, validation_dataloader: YoloDataLoader) -> None:
    mAPs = model_validator.solve(validation_dataloader)
    except_mAPs = {"mAP.5": tensor(0.6969), "mAP.5:.95": tensor(0.4195)}
    assert mAPs is not None
    assert allclose(mAPs["mAP.5"], except_mAPs["mAP.5"], rtol=0.1)
    print(mAPs)
    assert allclose(mAPs["mAP.5:.95"], except_mAPs["mAP.5:.95"], rtol=0.1)


@pytest.fixture
def model_tester(inference_cfg: Config, model: YOLO, vec2box: Vec2Box, validation_progress_logger: ProgressLogger, device: torch.device) -> ModelTester:
    tester = ModelTester(inference_cfg, model, vec2box, validation_progress_logger, device)
    return tester


@pytest.fixture
def model_v7_tester(inference_v7_cfg: Config, model_v7: YOLO, anc2box: Anc2Box, validation_progress_logger: ProgressLogger, device: torch.device) -> ModelTester:
    tester = ModelTester(inference_v7_cfg, model_v7, anc2box, validation_progress_logger, device)
    return tester


def test_model_tester_initialization(model_tester: ModelTester) -> None:
    assert isinstance(model_tester.model, YOLO)
    assert hasattr(model_tester, "solve")


def test_model_tester_solve_single_image(model_tester: ModelTester, file_stream_data_loader: StreamDataLoader) -> None:
    model_tester.solve(file_stream_data_loader)


def test_model_v7_tester_solve_single_image(model_v7_tester: ModelTester, file_stream_data_loader_v7: StreamDataLoader) -> None:
    model_v7_tester.solve(file_stream_data_loader_v7)


@pytest.fixture
def model_trainer(train_cfg: Config, model: YOLO, vec2box: Vec2Box, train_progress_logger: ProgressLogger, device: torch.device) -> ModelTrainer:
    assert isinstance(train_cfg.task, TrainConfig)
    train_cfg.task.epoch = 2
    trainer = ModelTrainer(train_cfg, model, vec2box, train_progress_logger, device, use_ddp=False)
    return trainer


def test_model_trainer_initialization(model_trainer: ModelTrainer) -> None:

    assert isinstance(model_trainer.model, YOLO)
    assert hasattr(model_trainer, "solve")
    assert model_trainer.optimizer is not None
    assert model_trainer.scheduler is not None
    assert model_trainer.loss_fn is not None


# def test_model_trainer_solve_mock_dataset(model_trainer: ModelTrainer, train_dataloader: YoloDataLoader):
#     model_trainer.solve(train_dataloader)
