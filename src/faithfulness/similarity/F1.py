from typing import List
from collections import Counter
import torch
from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.utils import normalize_text, calc_prf1, PRF1Result


class F1(SimilarityMetricInterface):

    def score(self, summary_text: str, source_text: str) -> PRF1Result:
        return self.f1_score(summary_text, source_text)

    def score_batch(self, summaries: List[str], sources: List[str]) -> List[PRF1Result]:
        return [self.f1_score(summary, source) for summary, source in zip(summaries, sources)]

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
        summary_texts = [self.__get_tokens(x) for x in summary_texts]
        source_texts = [self.__get_tokens(x) for x in source_texts]

        # compute similarities
        similarities = torch.tensor([[self.f1_score_with_tokens(summary_text, source_text)["f1"]
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

    @staticmethod
    def __get_tokens(s):
        if not s:
            return []
        return normalize_text(s).split()

    @staticmethod
    def f1_score(a_pred, a_gold) -> PRF1Result:
        gold_toks = F1.__get_tokens(a_gold)
        pred_toks = F1.__get_tokens(a_pred)
        return F1.f1_score_with_tokens(pred_toks, gold_toks)

    @staticmethod
    def f1_score_with_tokens(pred_toks, gold_toks) -> PRF1Result:
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return {
                "precision": float(gold_toks == pred_toks),
                "recall": float(gold_toks == pred_toks),
                "f1": float(gold_toks == pred_toks)
            }
        if num_same == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        if (precision + recall) > 0.0:
            f1 = 2.0 * ((precision * recall) / (precision + recall))
        else:
            f1 = 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
