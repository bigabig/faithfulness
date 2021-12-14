from typing import List, Dict
from faithfulness.utils.utils import PRF1Result


# used in NER to have alignments for every NER label (key is ner label, value is alignment)
class GroupedAlignScoreResult(PRF1Result):
    summary_source_alignment: Dict[str, List[int]]
    source_summary_alignment: Dict[str, List[int]]
    summary_source_similarities: Dict[str, List[List[float]]]
    source_summary_similarities: Dict[str, List[List[float]]]
