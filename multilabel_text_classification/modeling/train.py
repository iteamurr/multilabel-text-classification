from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
from loguru import logger
import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm

import entities.params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(labels, preds):
    metrics = {
        "precision_micro": sklearn.metrics.precision_score(
            labels, preds, average="micro", zero_division=0
        ),
        "recall_micro": sklearn.metrics.recall_score(
            labels, preds, average="micro", zero_division=0
        ),
        "f1_micro": sklearn.metrics.f1_score(labels, preds, average="micro", zero_division=0),
        "precision_macro": sklearn.metrics.precision_score(
            labels, preds, average="macro", zero_division=0
        ),
        "recall_macro": sklearn.metrics.recall_score(
            labels, preds, average="macro", zero_division=0
        ),
        "f1_macro": sklearn.metrics.f1_score(labels, preds, average="macro", zero_division=0),
    }
    metrics.update(
        {
            f"roc_auc_class_{i}": sklearn.metrics.roc_auc_score(labels[:, i], preds[:, i])
            for i in range(labels.shape[1])
        }
    )
    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, device, loss_metric, writer):
    model.train()
    all_preds = []
    all_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_metric.update(loss.item())

        preds = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if step % 10 == 0:
            writer.add_scalar("Loss/Train", loss.item(), step)

    avg_loss = loss_metric.compute().item()
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            losses.append(loss.item())

            preds = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    config: entities.params.PipelineParams,
    writer,
):
    loss_metric = torchmetrics.MeanMetric().to(device)
    metrics_history = []

    for epoch in range(config.train_params.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.train_params.num_epochs}")
        train_loss, train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, loss_metric, writer=writer
        )
        val_loss, val_metrics = evaluate(model, val_dataloader, device)

        logger.info(
            f"Train Loss: {train_loss:.4f}\nTrain Metrics: {train_metrics}\nVal Loss: {val_loss:.4f}\nVal Metrics: {val_metrics}"
        )

        writer.add_scalar("Loss/Validation", val_loss, epoch)
        for metric, value in {**train_metrics, **val_metrics}.items():
            writer.add_scalar(
                f"Metrics/{'Train' if metric in train_metrics else 'Validation'}/{metric}",
                value,
                epoch,
            )

        metrics_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

    return metrics_history


def save_model_and_metrics(
    model,
    tokenizer,
    metrics_history,
    config: entities.params.PipelineParams,
    multi=False
):
    metrics_df = pd.DataFrame(metrics_history).set_index("epoch")
    if multi:
        metrics_df.to_csv(Path(config.paths.metrics_path) / config.data_params.single_cls_metrics_file)
        logger.info(
            f"Metrics history saved to {Path(config.paths.metrics_path) / config.data_params.single_cls_metrics_file}"
        )
        model_dir = Path(config.paths.single_cls_model_path)
    else:
        metrics_df.to_csv(Path(config.paths.metrics_path) / config.data_params.multi_cls_metrics_file)
        logger.info(
            f"Metrics history saved to {Path(config.paths.metrics_path) / config.data_params.multi_cls_metrics_file}"
        )
        model_dir = Path(config.paths.multi_cls_model_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    model.base_model.save_pretrained(model_dir)
    torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pth")

    logger.success(f"Model and tokenizer saved in: {model_dir}")


def load_data(file, config: entities.params.PipelineParams, target_column="comment_text"):
    data = pd.read_csv(Path(config.paths.processed_data_path) / file)
    return data.dropna(subset=[target_column])


def extract_texts_labels(data, target_column="comment_text"):
    return (
        data[target_column].tolist(),
        data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values,
    )
