import sys
from pathlib import Path

# Add the root of layoutparser (i.e., /src/) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from layoutparser.models.base_layoutmodel import BaseLayoutModel as LayoutModel
except ImportError:
    from ..base_layoutmodel import BaseLayoutModel as LayoutModel
from ...elements import TextBlock
import importlib

# Prevent local detectron2 folder from shadowing real package
sys.path = [p for p in sys.path if "src/layoutparser/models/detectron2" not in p]

# Dynamically load detectron2 components
get_cfg = importlib.import_module("detectron2.config").get_cfg
DefaultPredictor = importlib.import_module("detectron2.engine").DefaultPredictor
model_zoo = importlib.import_module("detectron2.model_zoo")
Boxes = importlib.import_module("detectron2.structures").Boxes
import numpy as np

class Detectron2LayoutModel(LayoutModel):
    def __init__(
        self,
        config_path,
        label_map,
        extra_config=[],
        enforce_cpu=False,
        model_path=None,
    ):
        super().__init__()

        cfg = get_cfg()
        if config_path.startswith("lp://"):
            config_path = model_zoo.get_config_file(config_path[5:])
        cfg.merge_from_file(config_path)

        for setting in extra_config:
            cfg.merge_from_list(setting)

        if enforce_cpu:
            cfg.MODEL.DEVICE = "cpu"

        if model_path is not None:
            cfg.MODEL.WEIGHTS = model_path
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.label_map = label_map

    def detect(self, image):
        outputs = self.predictor(np.array(image))
        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.tolist()
        labels = instances.pred_classes.tolist()

        layout = []
        for box, score, label in zip(boxes, scores, labels):
            if label in self.label_map:
                layout.append(
                    TextBlock(
                        list(box),
                        type=self.label_map[label],
                        score=score,
                    )
                )

        return layout