from dataclasses import dataclass, field

import marshmallow
import marshmallow.validate


@dataclass()
class TrainParams:
    batch_size: int = field(
        default=16,
        metadata={"validate": marshmallow.validate.Range(min=1)},
    )
    train_split: float = field(
        default=0.8,
        metadata={"validate": marshmallow.validate.Range(min=0.0, max=1.0)},
    )
    learning_rate: float = field(
        default=0.001,
        metadata={
            "validate": marshmallow.validate.Range(
                min=0.0,
            )
        },
    )
    num_epochs: int = field(
        default=10,
        metadata={"validate": marshmallow.validate.Range(min=1)},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"validate": marshmallow.validate.Range(min=1)},
    )
    pretrained_model: str = field(
        default="bert-base-uncased",
        metadata={
            "validate": marshmallow.validate.Length(min=1),
            "required": True,
        },
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={
            "validate": marshmallow.validate.Range(
                min=0.0,
                max=1.0,
            )
        },
    )
