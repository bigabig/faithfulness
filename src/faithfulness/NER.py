from typing import List

import spacy
import numpy as np

from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from tqdm import tqdm

from faithfulness.utils.utils import MetricVariant


class NER(MetricInterface):

    def __init__(self, metric: SimilarityMetricInterface, spacymodel='en_core_web_lg', variant=MetricVariant.F1, ner_variant=MetricVariant.F1):
        print(f'Loading Spacy model {spacymodel}...')
        self.nlp = spacy.load(spacymodel)
        self.metric = metric
        self.variant = variant
        self.ner_variant = ner_variant

    def score(self, summary_text: str, source_text: str):
        # extract entities
        summary_entities = self.__extract_entities(summary_text)
        source_entities = self.__extract_entities(source_text)

        if self.ner_variant == MetricVariant.F1:
            precision = self.__calc_score(summary_entities, summary_entities, source_entities)
            recall = self.__calc_score(source_entities, summary_entities, source_entities)
            return 2 * ((precision * recall) / (precision + recall))
        elif self.ner_variant == MetricVariant.PRECISION:
            return self.__calc_score(summary_entities, summary_entities, source_entities)
        else:
            return self.__calc_score(source_entities, summary_entities, source_entities)

    def score_batch(self, summaries: List[str], sources: List[str]):
        return [
            self.score(summary, source)
            for (summary, source) in tqdm(zip(summaries, sources))
        ]

    def __extract_entities(self, text):
        temp = self.nlp(text)

        result = {}
        for ent in temp.ents:
            label = ent.label_
            if label not in result.keys():
                result[label] = []
            result[label].append({"text": ent.text, "label": label, "start": ent.start_char, "end": ent.end_char})

        return result

    def __calc_score(self, entities, summary_entities, source_entities):
        results = []
        for ner_label in entities.keys():
            summary_ners = [x['text'] for x in summary_entities.get(ner_label, [])]
            source_ners = [x['text'] for x in source_entities.get(ner_label, [])]
            results.append(self.metric.align_and_score(summary_ners, source_ners)[self.variant])  # default: f1

        return np.array(results).mean().item()
