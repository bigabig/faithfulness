import pickle
import re
import string
from enum import Enum


class MetricVariant(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"


def calc_prf1(similarity_matrix, named=False):
    precision = similarity_matrix.max(dim=1).values.mean().item()
    recall = similarity_matrix.max(dim=0).values.mean().item()
    if (precision + recall) > 0.0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        f1 = 0.0
    if named:
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return precision, recall, f1


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
