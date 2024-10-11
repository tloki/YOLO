import sys
from pathlib import Path
from typing import Tuple, List, cast

import pytest
import torch
from hydra import compose, initialize
from torch import Tensor

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model, YOLO
from yolo.tools.loss_functions import DualLoss, create_loss_function
from yolo.utils.bounding_box_utils import Vec2Box


@pytest.fixture
def cfg() -> Config:
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg = compose(config_name="config", overrides=["task=train"])
    return cast(Config, cfg)


@pytest.fixture
def model(cfg: Config) -> YOLO:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model, weight_path=None)
    return model.to(device)


@pytest.fixture
def vec2box(cfg: Config, model: YOLO) -> Vec2Box:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Vec2Box(model, cfg.model.anchor, cfg.image_size, device)


@pytest.fixture
def loss_function(cfg: Config, vec2box: Vec2Box) -> DualLoss:
    return create_loss_function(cfg, vec2box)


@pytest.fixture
def data() -> Tuple[List[Tensor], Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = torch.zeros(1, 20, 5, device=device)
    predicts = [torch.zeros(1, 8400, *cn, device=device) for cn in [(80,), (4, 16), (4,)]]
    return predicts, targets


def test_yolo_loss(loss_function: DualLoss, data: Tuple[List[Tensor], Tensor]) -> None:
    predicts, targets = data
    loss, loss_dict = loss_function(predicts, predicts, targets)
    assert torch.isnan(loss)
    assert torch.isnan(loss_dict["BoxLoss"])
    assert torch.isnan(loss_dict["DFLoss"])
    assert torch.isinf(loss_dict["BCELoss"])
