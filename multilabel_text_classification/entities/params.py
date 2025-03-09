from dataclasses import dataclass, field
from pathlib import Path

import marshmallow_dataclass
import yaml

from multilabel_text_classification.entities.data_params import DataParams
from multilabel_text_classification.entities.paths import Paths
from multilabel_text_classification.entities.train_params import TrainParams


@dataclass()
class PipelineParams:
    paths: Paths = field(default_factory=Paths)
    train_params: TrainParams = field(default_factory=TrainParams)
    data_params: DataParams = field(default_factory=DataParams)
    random_state: int = field(default=42)


PipelineParamsSchema = marshmallow_dataclass.class_schema(PipelineParams)


def read_pipeline_params(path: Path) -> PipelineParams:
    with open(path, "r") as input_stream:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
