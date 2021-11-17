from typing import List

import spacy
import numpy as np

from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface


class NER(MetricInterface):

    def __init__(self, metric: SimilarityMetricInterface, spacymodel='en_core_web_lg'):
        print(f'Loading Spacy model {spacymodel}...')
        self.nlp = spacy.load(spacymodel)
        self.metric = metric

    def score(self, summary_text: str, source_text: str):
        # extract entities
        summary_entities = self.__extract_entities(summary_text)
        source_entities = self.__extract_entities(source_text)

        return self.__calc_score(summary_entities, source_entities)

    def score_batch(self, summaries: List[str], sources: List[str]):
        pass

    def __extract_entities(self, text):
        temp = self.nlp(text)

        result = {}
        for ent in temp.ents:
            label = ent.label_
            if label not in result.keys():
                result[label] = []
            result[label].append({"text": ent.text, "label": label, "start": ent.start_char, "end": ent.end_char})

        return result

    def __calc_score(self, summary_entities, source_entities):
        # todo: maybe offer 2 versions, one that loops over summary entities, one that loops over source entities?
        # todo: maybe also return precision, recall and alignment
        results = []
        for ner_label in summary_entities.keys():
            summary_ners = [x['text'] for x in summary_entities.get(ner_label, [])]
            source_ners = [x['text'] for x in source_entities.get(ner_label, [])]
            results.append(self.metric.align_and_score(summary_ners, source_ners)[2])  # we use f1 (we also get precision, recall, alignment and similarities as result)

        return np.array(results).mean().item()
