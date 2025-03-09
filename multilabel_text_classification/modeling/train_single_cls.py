import torch
import transformers
import typer
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import entities.params
import modeling.train


app = typer.Typer()

writer = SummaryWriter(log_dir="runs")


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


class SingleCLSClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(SingleCLSClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
            nn.Tanh(),
            nn.Linear(base_model.config.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


def create_dataloader(texts, labels, tokenizer, config: entities.params.PipelineParams, shuffle):
    logger.info(
        f"Creating {'train' if shuffle else 'validation'} dataloader with batch size {config.train_params.batch_size}"
    )
    dataset = CustomDataset(
        texts,
        labels,
        tokenizer,
        max_length=config.train_params.max_seq_length,
    )
    return DataLoader(dataset, batch_size=config.train_params.batch_size, shuffle=shuffle)


@app.command()
def main(config_path: str) -> None:
    config = entities.params.read_pipeline_params(config_path)
    modeling.train.set_seed(config.random_state)

    logger.info("Loading training and validation data...")
    train_data, val_data = map(
        lambda file: modeling.train.load_data(file, config),
        [config.data_params.train_file, config.data_params.val_file],
    )

    logger.info("Extracting labels...")
    train_texts, train_labels = modeling.train.extract_texts_labels(train_data)
    val_texts, val_labels = modeling.train.extract_texts_labels(val_data)

    logger.info("Initializing tokenizer and model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.train_params.pretrained_model
    )
    base_model = transformers.AutoModel.from_pretrained(config.train_params.pretrained_model)
    model = SingleCLSClassifier(base_model, train_labels.shape[1])

    logger.info("Setting up optimizer and scheduler...")
    optimizer = transformers.AdamW(model.parameters(), lr=config.train_params.learning_rate)
    total_steps = (
        len(train_texts) // config.train_params.batch_size
    ) * config.train_params.num_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    train_dataloader = create_dataloader(train_texts, train_labels, tokenizer, config, True)
    val_dataloader = create_dataloader(val_texts, val_labels, tokenizer, config, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    logger.info("Starting training and evaluation...")
    metrics_history = modeling.train.train_and_evaluate(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device,
        config,
        writer,
    )

    logger.info("Saving model and metrics...")
    modeling.train.save_model_and_metrics(model, tokenizer, metrics_history, config, True)
    logger.success("Training complete")


if __name__ == "__main__":
    app()
