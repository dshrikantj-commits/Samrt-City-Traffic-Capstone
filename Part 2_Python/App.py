import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

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


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(FEAT_PATH)
        df["date_time"] = pd.to_datetime(df["date_time"])
        return df
    except FileNotFoundError:
        logger.error("Features file not found: %s", FEAT_PATH, exc_info=True)
        sys.exit(1)


def cmd_summary(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("Date range:", df["date_time"].min(), "to", df["date_time"].max())
    print("traffic_volume:", round(df["traffic_volume"].mean(), 1))
    print("weather_description:", (df["weather_description"], 1))
    print(df["traffic_category"].value_counts().to_string())


def cmd_by_hour(df: pd.DataFrame, hour: int) -> None:
    if hour < 0 or hour > 23:
        logger.error("Invalid hour: %s", hour)
        print("Error: hour must be between 0 and 23")
        return
    sub = df[df["hour"] == hour]
    if sub.empty:
        print("No rows for hour", hour)
        return
    print(f"Hour {hour}: {len(sub)} records")
    print("traffic_volume:", round(sub["traffic_volume"].mean(), 1))
    


def cmd_recommend(df: pd.DataFrame, day: str) -> None:
    day = day.strip().lower()
    if day in ("weekday", "weekend"):
        sub = df[df["is_weekend"] == (0 if day == "weekday" else 1)]
    elif day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        sub = df[df["day_of_week"] == day]
    else:
        logger.error("Invalid day type: %s", day)
        print("Error: day must be 'weekday' or 'weekend'")
        return
    sub = df[df["is_weekend"] == (0 if day == "weekday" else 1)]
    by_hour = sub.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False)
    top = by_hour.head(3)
    print(f"Busiest hours for {day} traffic:")
    for h, v in top.items():
        print(f"  {int(h):02d}:00  avg traffic volume {v:.0f}")
    h0 = int(top.index[0])
    print(f"Suggestion: lesser traffic around {h0:02d}:00 on a {day}.")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Traffic volume CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("summary")
    p_h = sub.add_parser("by-hour")
    p_h.add_argument("--hour", type=int, required=True)
    p_r = sub.add_parser("recommend")
    p_r.add_argument("--day", type=str, required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    logger.info("CLI command=%s args=%s", args.command, vars(args))
    df = load_data()
    if args.command == "summary":
        cmd_summary(df)
    elif args.command == "by-hour":
        cmd_by_hour(df, args.hour)
    elif args.command == "recommend":
        cmd_recommend(df, args.day)


if __name__ == "__main__":
    main()