from typing import List
import torch

from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import normalize_text, calc_prf1, F1Result


class ExactMatch(SimilarityMetricInterface):

    def score(self, summary_text: str, source_text: str) -> F1Result:
        return {"f1": 1.0 if normalize_text(summary_text) == normalize_text(source_text) else 0.0}

    def score_batch(self, summaries: List[str], sources: List[str]) -> List[F1Result]:
        return [self.score(summary, source) for summary, source in zip(summaries, sources)]

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]) -> AlignScoreResult:
        if len(summary_texts) == 0 or len(source_texts) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "summary_source_alignment": [],
                "source_summary_alignment": [],
                "summary_source_similarities": [],
                "source_summary_similarities": []
            }

        # normalize texts
        summary_texts = [normalize_text(x) for x in summary_texts]
        source_texts = [normalize_text(x) for x in source_texts]

        # compute similarities
        similarities = torch.tensor([[1.0 if source_text == summary_text else 0.0
                                      for source_text in source_texts]
                                     for summary_text in summary_texts])

        # compute alignment
        summary_source_alignment = similarities.argmax(dim=1).tolist()
        source_summary_alignment = similarities.argmax(dim=0).tolist()

        # compute scores
        prf1 = calc_prf1(similarities)

        return {
            "precision": prf1["precision"],
            "recall": prf1["recall"],
            "f1": prf1["f1"],
            "summary_source_alignment": summary_source_alignment,
            "source_summary_alignment": source_summary_alignment,
            "summary_source_similarities": similarities.tolist(),
            "source_summary_similarities": similarities.T.tolist(),
        }
