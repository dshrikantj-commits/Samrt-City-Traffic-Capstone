import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FEAT_PATH = Path("data/ft_Metro_Interstate_Traffic_Volume.csv")
FIG_DIR = Path("figures")
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


def main():
    setup_logging()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(FEAT_PATH)
        logger.info("Loaded features: %s rows", len(df))

        g = df.groupby("hour")["traffic_volume"].mean()
        p1 = FIG_DIR / "avg_trafffic_by_hour.png"
        plt.figure(figsize=(8, 4))
        plt.bar(g.index, g.values, color="teal")
        plt.xlabel("hour")
        plt.ylabel("traffic_volume")
        plt.title("Traffic Volume By Hour")
        plt.tight_layout()
        plt.savefig(p1)
        plt.close()
        logger.info("Saved figure: %s", p1)

        g2 = df.groupby("is_weekend")["traffic_volume"].mean()
        p2 = FIG_DIR / "weekday_vs_weekend.png"
        plt.figure(figsize=(5, 4))
        plt.bar(["Weekday", "Weekend"], [g2.get(0, 0), g2.get(1, 0)], color=["coral", "seagreen"])
        plt.ylabel("traffic_volume")
        plt.title("Weekday vs weekend traffic volume")
        plt.tight_layout()
        plt.savefig(p2)
        plt.close()
        logger.info("Saved figure: %s", p2)

        g3 = df.groupby("holiday")["traffic_volume"].mean().sort_values()
        p3 = FIG_DIR / "traffic_holiday.png"
        plt.figure(figsize=(7, 4))
        plt.barh(g3.index, g3.values, color="slateblue")
        plt.xlabel("holiday")
        plt.title("Traffic Volume my Holiday")
        plt.tight_layout()
        plt.savefig(p3)
        plt.close()
        logger.info("Saved figure: %s", p3)

        logger.info("All figures created")
    except FileNotFoundError:
        logger.error("Features file missing. Run feature_engineering.py first.", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Visualisation step failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()