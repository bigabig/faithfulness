from typing import List


class SimilarityMetricInterface:
    """Compares two texts"""

    def score(self, summary_text: str, source_text: str):
        """Compares two texts"""
        raise NotImplementedError

    def score_batch(self, summaries: List[str], sources: List[str]):
        """Pairwise comparison of two texts"""
        raise NotImplementedError

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        """Compares and aligns phrases / answers / entities that occur in two texts"""
        raise NotImplementedError
