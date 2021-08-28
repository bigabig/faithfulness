from typing import List
import torch
from faithfulness.similarity.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import normalize_text, calc_prf1


class ExactMatch(SimilarityMetricInterface):

    def score(self, summary_text: str, source_text: str):
        return 1 if normalize_text(summary_text) == normalize_text(source_text) else 0

    def score_batch(self, summaries: List[str], sources: List[str]):
        return [self.score(summary, source) for summary, source in zip(summaries, sources)]

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        # normalize texts
        summary_texts = [normalize_text(x) for x in summary_texts]
        source_texts = [normalize_text(x) for x in source_texts]

        # compute similarities
        similarities = torch.tensor([[1 if source_text == summary_text else 0
                                      for source_text in source_texts]
                                     for summary_text in summary_texts])

        # compute alignment
        summary_source_alignment = similarities.argmax(dim=1)

        # compute scores
        precision, recall, f1 = calc_prf1(similarities)

        return precision, recall, f1, summary_source_alignment, similarities.tolist()
