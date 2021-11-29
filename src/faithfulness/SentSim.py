from typing import List
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from tqdm import tqdm

from faithfulness.utils.utils import MetricVariant


class SentSim(MetricInterface):
    def __init__(self, metric: SimilarityMetricInterface, variant=MetricVariant.F1):
        self.metric = metric
        self.variant = variant

    def score(self, summary_sentences: List[str], source_sentences: List[str], additional_output: bool):
        result = self.metric.align_and_score(summary_sentences, source_sentences)
        if additional_output:
            return result
        return {self.variant.value: result[self.variant.value]}

    def score_batch(self, summaries_sentences: List[List[str]], sources_sentences: List[List[str]], additional_output: bool):
        results = {}
        for summary_sentence, source_sentence in tqdm(zip(summaries_sentences, sources_sentences),  desc="Calculating SentSim..."):
            result = self.score(summary_sentence, source_sentence, additional_output)
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]
        return results
