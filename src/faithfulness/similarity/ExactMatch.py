from typing import List
import torch
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import normalize_text, calc_prf1, MetricVariant


class ExactMatch(SimilarityMetricInterface):

    def get_variant(self) -> MetricVariant:
        return MetricVariant.F1

    def score(self, summary_text: str, source_text: str, additional_output: bool):
        return {"f1": 1.0 if normalize_text(summary_text) == normalize_text(source_text) else 0.0}

    def score_batch(self, summaries: List[str], sources: List[str], additional_output: bool):
        return {"f1": [self.score(summary, source, additional_output) for summary, source in zip(summaries, sources)]}

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        if len(summary_texts) == 0 or len(source_texts) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "alignment": [], "similarities": []}

        # normalize texts
        summary_texts = [normalize_text(x) for x in summary_texts]
        source_texts = [normalize_text(x) for x in source_texts]

        # compute similarities
        similarities = torch.tensor([[1.0 if source_text == summary_text else 0.0
                                      for source_text in source_texts]
                                     for summary_text in summary_texts])

        # compute alignment
        summary_source_alignment = similarities.argmax(dim=1).tolist()

        # compute scores
        precision, recall, f1 = calc_prf1(similarities)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "alignment": summary_source_alignment,
            "similarities": similarities.tolist()
        }
