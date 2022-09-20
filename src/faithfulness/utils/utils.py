import sys
import json
import pathlib
import pickle
import re
import string
from enum import Enum
import pandas as pd
from pandas import DataFrame
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class F1Result(TypedDict):
    f1: float


class PRF1Result(F1Result):
    precision: float
    recall: float


def is_PRF1Result(obj) -> bool:
    return "precision" in obj and "recall" in obj and "f1" in obj


def is_F1Result(obj) -> bool:
    return "f1" in obj


class MetricVariant(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"


def calc_prf1(similarity_matrix) -> PRF1Result:
    precision = similarity_matrix.max(dim=1).values.mean().item()
    recall = similarity_matrix.max(dim=0).values.mean().item()
    if (precision + recall) > 0.0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        f1 = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def load_data(path):
    try:
        data = pickle.load(open(path, "rb"))
    except FileNotFoundError:
        data = []
    return data


def save_data(data, path):
    pickle.dump(data, open(path, "wb"))


def load_csv_data(data_path: pathlib.Path) -> DataFrame:
    if not data_path.exists():
        print(f"ERROR: Path {data_path} does not exist!")
        exit()

    if not data_path.is_file() or data_path.suffix != ".csv":
        print(f"ERROR: Path {data_path} does not point to a .sv file!")
        exit()

    with data_path.open(encoding="UTF-8", mode="r") as file:
        df = pd.read_csv(file)

    if df is None or len(df) <= 0:
        print(f"ERROR: Data loading failed! (1)")
        exit()

    return df


def load_json_data(data_path: pathlib.Path, examples: int = -1):
    if not data_path.exists():
        print(f"ERROR: Path {data_path} does not exist!")
        exit()

    if not data_path.is_file() or data_path.suffix != ".json":
        print(f"ERROR: Path {data_path} does not point to a .json file!")
        exit()

    with data_path.open(encoding="UTF-8", mode="r") as file:
        data = json.load(file)
        data = data[0: examples] if examples != -1 else data

    if data is None or len(data) <= 0:
        print(f"ERROR: Data loading failed! (1)")
        exit()

    return data


def ensure_dir_exists(path: pathlib.Path):
    if not path.exists():
        path.mkdir(parents=True)
    if not path.is_dir():
        print(f"ERROR: Path {path} does not point to a folder!")
        exit()
