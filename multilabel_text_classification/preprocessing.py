import re
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split

import entities.params


app = typer.Typer()


@app.command()
def main(config_path: str):
    config = entities.params.read_pipeline_params(config_path)

    logger.info(f"Loading dataset from {config.paths.raw_data_path}...")
    df = pd.read_csv(Path(config.paths.raw_data_path) / config.data_params.train_file)
    logger.debug(f"Dataset shape: {df.shape}")
    logger.success("Dataset loaded successfully.")

    logger.info("Transforming dataset...")
    df.dropna(inplace=True)

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
