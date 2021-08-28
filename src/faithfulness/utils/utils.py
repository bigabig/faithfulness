import re
import string


def calc_prf1(similarity_matrix):
    precision = similarity_matrix.max(dim=1).values.mean()
    recall = similarity_matrix.max(dim=0).values.mean()
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision.item(), recall.item(), f1.item()


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
