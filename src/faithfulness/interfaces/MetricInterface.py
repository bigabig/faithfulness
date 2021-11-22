from typing import List


class MetricInterface:
    """Compares two texts"""

    def score(self, summary_text: str, source_text: str, additional_output: bool):
        """Compares two texts"""
        raise NotImplementedError

    def score_batch(self, summaries: List[str], sources: List[str], additional_output: bool):
        """Pairwise comparison of two texts"""
        raise NotImplementedError
