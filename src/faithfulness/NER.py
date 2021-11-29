from typing import List

import spacy
import numpy as np

from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from tqdm import tqdm

from faithfulness.utils.utils import MetricVariant


class NER(MetricInterface):

    def __init__(self, metric: SimilarityMetricInterface, spacymodel='en_core_web_lg', metric_variant=MetricVariant.F1, ner_variant=MetricVariant.F1):
        print(f'Loading Spacy model {spacymodel}...')
        self.nlp = spacy.load(spacymodel)
        self.metric = metric
        self.metric_variant = metric_variant
        self.ner_variant = ner_variant

    def score(self, summary_text: str, source_text: str, additional_output: bool):
        # extract entities
        summary_entities = self.__extract_entities(summary_text)
        source_entities = self.__extract_entities(source_text)

        result = {}
        if additional_output:
            result["summary_entities"] = summary_entities
            result["source_entities"] = source_entities

        if self.ner_variant == MetricVariant.F1:
            summary_source_result = self.__calc_score(summary_entities, summary_entities, source_entities, additional_output)
            source_summary_result = self.__calc_score(source_entities, summary_entities, source_entities, additional_output)

            scores = {
                "precision": summary_source_result[self.metric_variant.value],
                "recall": source_summary_result[self.metric_variant.value],
            }

            scores["f1"] = 2 * ((scores["precision"] * scores["recall"]) / (scores["precision"] + scores["recall"])) if (scores["precision"] + scores["recall"]) > 0.0 else 0.0

            if additional_output:
                scores["summary_source_alignment"] = summary_source_result["alignment"],
                scores["summary_source_similarities"] = summary_source_result["similarities"],
                scores["source_summary_alignment"] = source_summary_result["alignment"],
                scores["source_summary_similarities"] = source_summary_result["similarities"],
                result.update(scores)
            else:
                result.update({self.ner_variant.value: scores[self.ner_variant.value]})

        elif self.ner_variant == MetricVariant.PRECISION:
            result.update(self.__calc_score(summary_entities, summary_entities, source_entities, additional_output))
        elif self.ner_variant == MetricVariant.RECALL:
            result.update(self.__calc_score(source_entities, summary_entities, source_entities, additional_output))

        return result

    def score_batch(self, summaries: List[str], sources: List[str], additional_output):
        results = {}
        for summary, source in tqdm(zip(summaries, sources), desc="Calculating NER..."):
            result = self.score(summary, source, additional_output)
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]
        return results

    def __extract_entities(self, text):
        temp = self.nlp(text)

        result = {}
        for ent in temp.ents:
            label = ent.label_
            if label not in result.keys():
                result[label] = []
            result[label].append({"text": ent.text, "label": label, "start": ent.start_char, "end": ent.end_char})

        return result

    def __calc_score(self, entities, summary_entities, source_entities, additional_output: bool):
        results = {}
        for ner_label in entities.keys():
            summary_ners = [x['text'] for x in summary_entities.get(ner_label, [])]
            source_ners = [x['text'] for x in source_entities.get(ner_label, [])]
            result = self.metric.align_and_score(summary_ners, source_ners)
            for key, value in result.items():
                if key == self.metric_variant.value:
                    results[self.metric_variant.value] = [*results.get(self.metric_variant.value, []), value]
                elif additional_output and key in ["alignment", "similarities"]:
                    # TODO handle alignment (correctly) in combination with ner label
                    results[key] = [*results.get(key, []), value]

        # if a summary has no named entities, we assume it is faithful!
        if len(entities.keys()) == 0:
            results[self.metric_variant.value] = 1.0
            if additional_output:
                results["alignment"] = []
                results["similarities"] = []
            return results

        results[self.metric_variant.value] = np.array(results[self.metric_variant.value]).mean().item()
        return results
