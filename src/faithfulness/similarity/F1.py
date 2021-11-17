from typing import List
from collections import Counter
import torch
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import normalize_text, calc_prf1


class F1(SimilarityMetricInterface):
    def score(self, summary_text: str, source_text: str):
        return self.f1_score(summary_text, source_text)

    def score_batch(self, summaries: List[str], sources: List[str]):
        return [self.f1_score(summary, source) for summary, source in zip(summaries, sources)]

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        # normalize texts
        summary_texts = [self.__get_tokens(x) for x in summary_texts]
        source_texts = [self.__get_tokens(x) for x in source_texts]

        # compute similarities
        similarities = torch.tensor([[self.f1_score_with_tokens(summary_text, source_text)
                                      for source_text in source_texts]
                                     for summary_text in summary_texts])

        # compute alignment
        summary_source_alignment = similarities.argmax(dim=1)

        # compute scores
        precision, recall, f1 = calc_prf1(similarities)

        return precision, recall, f1, summary_source_alignment, similarities.tolist()

    @staticmethod
    def __get_tokens(s):
        if not s:
            return []
        return normalize_text(s).split()

    @staticmethod
    def f1_score(a_gold, a_pred):
        gold_toks = F1.__get_tokens(a_gold)
        pred_toks = F1.__get_tokens(a_pred)
        return F1.f1_score_with_tokens(gold_toks, pred_toks)

    @staticmethod
    def f1_score_with_tokens(gold_toks, pred_toks):
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
