from typing import List


class MetricInterface:
    """Compares two texts"""

    def score(self, summary_text: str, source_text: str):
        """Compares two texts"""
        raise NotImplementedError

    def score_batch(self, summaries: List[str], sources: List[str]):
        """Pairwise comparison of two texts"""
        raise NotImplementedError
