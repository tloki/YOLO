import sys
from pathlib import Path

from PIL import Image
from torch import tensor

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import YOLO
from yolo.tools.drawer import draw_bboxes, draw_model


def test_draw_model_by_config(train_cfg: Config) -> None:
    """Test the drawing of a model based on a configuration."""
    draw_model(model_cfg=train_cfg.model)


def test_draw_model_by_model(model: YOLO) -> None:
    """Test the drawing of a YOLO model."""
    draw_model(model=model)


def test_draw_bboxes() -> None:
    """Test drawing bounding boxes on an image."""
    predictions = tensor([[0, 60, 60, 160, 160, 0.5], [0, 40, 40, 120, 120, 0.5]])
    pil_image = Image.open("tests/data/images/train/000000050725.jpg")
    draw_bboxes(pil_image, [predictions])
