from typing import List

import spacy

from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface


class SentSim(MetricInterface):

    def __init__(self, metric: SimilarityMetricInterface, spacymodel='en_core_web_lg'):
        print(f'Loading Spacy model {spacymodel}...')
        self.nlp = spacy.load(spacymodel)
        self.metric = metric

    def score(self, summary_text: str, source_text: str):
        # split sentences
        summary_sentences = self.__split_sentences(summary_text)
        source_sentences = self.__split_sentences(source_text)

        return self.metric.align_and_score(summary_sentences, source_sentences)

    def score_batch(self, summaries: List[str], sources: List[str]):
        pass

    def __split_sentences(self, text):
        return [x.text for x in self.nlp(text).sents]

