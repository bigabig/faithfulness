from typing import List
from faithfulness.utils.utils import PRF1Result


class AlignScoreResult(PRF1Result):
    summary_source_alignment: List[int]
    source_summary_alignment: List[int]
    summary_source_similarities: List[List[float]]
    source_summary_similarities: List[List[float]]
