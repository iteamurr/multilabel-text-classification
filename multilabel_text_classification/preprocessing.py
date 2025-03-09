import re
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

import entities.params


app = typer.Typer()


def preprocess_text(text: str, remove_digits=True) -> str:
    pattern = r"[^a-zA-Z\s]" if remove_digits else r"[^a-zA-Z0-9\s]"
    return re.sub(r"\s+", " ", re.sub(pattern, "", text).lower()).strip()


@app.command()
def main(config_path: str):
    config = entities.params.read_pipeline_params(config_path)

    logger.info(f"Loading dataset from {config.paths.raw_data_path}...")
    df = pd.read_csv(Path(config.paths.raw_data_path) / config.data_params.train_file)
    logger.debug(f"Dataset shape: {df.shape}")
    logger.success("Dataset loaded successfully.")

    df.dropna(inplace=True)

    logger.info("Transforming dataset...")
    df["comment_text"] = df["comment_text"].apply(preprocess_text)

    train_data, val_data = train_test_split(
        df,
        test_size=1 - config.train_params.train_split,
        random_state=config.random_state,
    )

    processed_data_dir = Path(config.paths.processed_data_path)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(processed_data_dir / config.data_params.train_file, index=False)
    val_data.to_csv(processed_data_dir / config.data_params.val_file, index=False)

    logger.success(
        f"Processing dataset complete. Processed dataset saved to {config.paths.processed_data_path}."
    )


if __name__ == "__main__":
    app()
