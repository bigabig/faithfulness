from typing import List, Type
from faithfulness.interfaces.UsesSimilarityMetricInterface import UsesSimilarityMetricInterface
from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from tqdm import tqdm


class SentSim(MetricInterface, UsesSimilarityMetricInterface):
    def __init__(self, metric: Type[SimilarityMetricInterface], metric_args=None):
        super(SentSim, self).__init__(metric=metric, metric_args=metric_args)
        self.load_metric()

    @staticmethod
    def needs_input() -> FaithfulnessInput:
        return FaithfulnessInput.SENTENCE

    def score(self, summary_sentences: List[str], source_sentences: List[str]) -> AlignScoreResult:
        return self.metric.align_and_score(summary_sentences, source_sentences)

    def score_batch(self, summaries_sentences: List[List[str]], sources_sentences: List[List[str]]) -> List[AlignScoreResult]:
        results = []
        for summary_sentence, source_sentence in tqdm(zip(summaries_sentences, sources_sentences), desc="Calculating SentSim..."):
            results.append(self.score(summary_sentence, source_sentence))
        return results
