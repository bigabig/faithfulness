import sys
import spacy
import numpy as np
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from tqdm import tqdm
from typing import List, Dict, Type
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from faithfulness.interfaces.UsesSimilarityMetricInterface import UsesSimilarityMetricInterface
from faithfulness.types.GroupedAlignScoreResult import GroupedAlignScoreResult


class Entity(TypedDict):
    text: str
    label: str
    start: int
    end: int


class NERResult(GroupedAlignScoreResult):
    summary_entities:  Dict[str, List[Entity]]
    source_entities:  Dict[str, List[Entity]]


class NER(MetricInterface, UsesSimilarityMetricInterface):

    def __init__(self, metric: Type[SimilarityMetricInterface], metric_args=None, spacymodel='en_core_web_lg'):
        super(NER, self).__init__(metric, metric_args)
        print(f'Loading Spacy model {spacymodel}...')
        self.nlp = spacy.load(spacymodel)
        self.load_metric()

    @staticmethod
    def needs_input() -> FaithfulnessInput:
        return FaithfulnessInput.DOCUMENT

    def score(self, summary_text: str, source_text: str) -> NERResult:
        # extract entities
        summary_entities = self.__extract_entities(summary_text)
        source_entities = self.__extract_entities(source_text)

        # find common entities labels
        common_tags = set.intersection(set(summary_entities.keys()), set(source_entities.keys()))
        # missing_tags = set(summary_entities.keys()).difference(set(source_entities.keys()))

        tmp = self.__calc_score(list(common_tags), summary_entities, source_entities)
        return {
            "precision": tmp["precision"],
            "recall": tmp["recall"],
            "f1": tmp["f1"],
            "summary_source_alignment": tmp["summary_source_alignment"],
            "source_summary_alignment": tmp["source_summary_alignment"],
            "summary_source_similarities": tmp["summary_source_similarities"],
            "source_summary_similarities": tmp["source_summary_similarities"],
            "summary_entities": summary_entities,
            "source_entities": source_entities
        }

    def score_batch(self, summaries: List[str], sources: List[str]) -> List[NERResult]:
        return [self.score(summary, source) for summary, source in tqdm(zip(summaries, sources), desc="Calculating NER...")]

    def __extract_entities(self, text) -> Dict[str, List[Entity]]:
        temp = self.nlp(text)

        result: Dict[str, List[Entity]] = {}
        for ent in temp.ents:
            label = ent.label_
            if label not in result.keys():
                result[label] = []
            result[label].append({"text": ent.text, "label": label, "start": ent.start_char, "end": ent.end_char})

        return result

    def __calc_score(self, entities: List[str], summary_entities: Dict[str, List[Entity]], source_entities: Dict[str, List[Entity]]) -> GroupedAlignScoreResult:
        results: GroupedAlignScoreResult = {}

        # if a summary has no named entities, we assume it is faithful!
        if len(entities) == 0:
            results["precision"] = 1.0
            results["recall"] = 1.0
            results["f1"] = 1.0
            results["summary_source_alignment"] = {}
            results["source_summary_alignment"] = {}
            results["summary_source_similarities"] = {}
            results["source_summary_similarities"] = {}
            return results

        for ner_label in entities:
            summary_ners = [x['text'] for x in summary_entities.get(ner_label, [])]
            source_ners = [x['text'] for x in source_entities.get(ner_label, [])]
            result = self.metric.align_and_score(summary_ners, source_ners)
            for key, value in result.items():
                if key in ["precision", "recall", "f1"]:
                    results[key] = [*results.get(key, []), value]
                else:  # key in ["summary_source_alignment", "source_summary_alignment", "summary_source_similarities", "source_summary_similarities"
                    if key not in results:
                        results[key] = {}
                    results[key][ner_label] = value

        # we keep summary / source alignments and similarities grouped by entity label
        # but we aggregate (calculate mean) of the precisions recalls and f1s
        results["precision"] = np.array(results["precision"]).mean().item()
        results["recall"] = np.array(results["recall"]).mean().item()
        results["f1"] = np.array(results["f1"]).mean().item()

        return results
