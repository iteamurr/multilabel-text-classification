from dataclasses import dataclass


@dataclass()
class DataParams:
    train_file: str
    test_file: str
    val_file: str
    single_cls_metrics_file: str
    multi_cls_metrics_file: str
