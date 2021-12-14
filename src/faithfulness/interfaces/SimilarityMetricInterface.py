from typing import List, Union
from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.utils.utils import PRF1Result, F1Result


class SimilarityMetricInterface(MetricInterface):
    """Compares two texts"""

    def score(self, summary_text: str, source_text: str) -> Union[PRF1Result, F1Result]:
        """Compares two texts"""
        raise NotImplementedError

    def score_batch(self, summaries: List[str], sources: List[str]) -> Union[List[PRF1Result], List[F1Result]]:
        """Pairwise comparison of two texts"""
        raise NotImplementedError

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]) -> AlignScoreResult:
        """Compares and aligns phrases / answers / entities that occur in two texts"""
        raise NotImplementedError
