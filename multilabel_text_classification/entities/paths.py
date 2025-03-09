from dataclasses import dataclass


@dataclass()
class Paths:
    data_path: str
    raw_data_path: str
    processed_data_path: str
    single_cls_model_path: str
    multi_cls_model_path: str
    metrics_path: str
