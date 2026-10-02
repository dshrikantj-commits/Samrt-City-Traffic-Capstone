import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/Metro_Interstate_Traffic_Volume.csv")
FEAT_PATH = Path("data/ft_Metro_Interstate_Traffic_Volume.csv")
logger = logging.getLogger(__name__)


def setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(
        "Feature engineering start: %s rows, %s columns", df.shape[0], df.shape[1]
    )
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df = df.fillna('No Holiday')
           
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    logger.debug("traffic quartiles q1=%.1f q2=%.1f q3=%.1f", q1, q2, q3)

    def traffic_vol(tc):
        if tc <= q1:
            return "Low"
        if tc <= q2:
            return "Medium"
        if tc <= q3:
            return "High"
        return "Peak"

    labels = []
    for tc in df["traffic_volume"]:
        labels.append(traffic_vol(tc))
    df["traffic_category"] = labels

    encoded_df = pd.get_dummies(df,
    columns=['weather_main'],dtype= int,
    drop_first=False)
    df = pd.concat([df, encoded_df], axis=1)

    logger.info(
        "Feature engineering end: %s rows, %s columns", df.shape[0], df.shape[1]
    )
    return df


def main():
    setup_logging()
    try:
        df = pd.read_csv(CLEAN_PATH)
        logger.info("Loaded cleaned data from %s", CLEAN_PATH)
        df = add_features(df)
        df.to_csv(FEAT_PATH, index=False)
        logger.info("Saved features to %s", FEAT_PATH)
    except FileNotFoundError:
        logger.error("Cleaned file missing. Run pipeline.py first.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Feature engineering failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()