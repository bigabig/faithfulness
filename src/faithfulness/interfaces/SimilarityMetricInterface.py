from typing import List
from faithfulness.interfaces.MetricInterface import MetricInterface


class SimilarityMetricInterface(MetricInterface):
    """Compares two texts"""

    def score(self, summary_text: str, source_text: str, additional_output: bool):
        """Compares two texts"""
        raise NotImplementedError

    def score_batch(self, summaries: List[str], sources: List[str], additional_output: bool):
        """Pairwise comparison of two texts"""
        raise NotImplementedError

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        """Compares and aligns phrases / answers / entities that occur in two texts"""
        raise NotImplementedError
