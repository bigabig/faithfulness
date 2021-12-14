import json
import csv
import pathlib
from builtins import map
from typing import List, Union
from faithfulness.BERTScore import BERTScore, BERTScoreResult
from faithfulness.Entailment import Entailment, EntailmentResult, is_EntailmentResult
from faithfulness.NER import NER, NERResult
from faithfulness.OpenIE import OpenIE, OpenIEResult
from faithfulness.QGQA import QGQA, QGQAResult
from faithfulness.SRL import SRL, SRLResult
from faithfulness.SentSim import SentSim
from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.utils.correlation import pearson, spearman
from faithfulness.utils.utils import ensure_dir_exists, PRF1Result, load_json_data


class Experimentor:
    def __init__(self, data_path: pathlib.Path, out_path: pathlib.Path, metric: Union[SentSim, QGQA, BERTScore, Entailment, NER, OpenIE, SRL], experiment_name: str,
                 data_faithfulness_key="faithfulness",
                 data_summary_key="summary",
                 data_source_key="source",
                 data_summary_sentences_key="summary_sentences",
                 data_source_sentences_key="source_sentences",
                 examples=-1):

        self.data_path = data_path

        ensure_dir_exists(out_path)
        self.out_file_json = out_path / (experiment_name + ".json")
        self.out_file_csv = out_path / (experiment_name + ".csv")

        self.metric = metric
        self.experiment_name = experiment_name
        self.examples = examples

        self.data = None
        self.experiment_input = None
        self.data_faithfulness_key = data_faithfulness_key
        self.data_summary_key = data_summary_key
        self.data_source_key = data_source_key
        self.data_summary_sentences_key = data_summary_sentences_key
        self.data_source_sentences_key = data_source_sentences_key

    def experiment(self):
        self.__load_experiment_data()

        # Calculate faithfulness, then add results to input data
        if isinstance(self.metric, SentSim):
            results: List[AlignScoreResult] = self.metric.score_batch(self.experiment_input[0], self.experiment_input[1])
            for x, result in zip(self.data, results):
                x["precision"] = result["precision"]
                x["recall"] = result["recall"]
                x["f1"] = result["f1"]
                x["summary_source_alignment"] = result["summary_source_alignment"]
                x["source_summary_alignment"] = result["source_summary_alignment"]
                x["summary_source_similarities"] = result["summary_source_similarities"]
                x["source_summary_similarities"] = result["source_summary_similarities"]

        if isinstance(self.metric, QGQA):
            results: List[QGQAResult] = self.metric.score_batch(self.experiment_input[0], self.experiment_input[1])
            for x, result in zip(self.data, results):
                x["precision"] = result["precision"]
                x["recall"] = result["recall"]
                x["f1"] = result["f1"]
                x["questions"] = result["questions"]

        if isinstance(self.metric, BERTScore):
            results: List[BERTScoreResult] = self.metric.score_batch(self.experiment_input[0], self.experiment_input[1])
            for x, result in zip(self.data, results):
                x["precision"] = result["precision"]
                x["recall"] = result["recall"]
                x["f1"] = result["f1"]
                x["similarities"] = result["similarities"]
                x["summary_tokens"] = result["summary_tokens"]
                x["source_tokens"] = result["source_tokens"]

        if isinstance(self.metric, Entailment):
            results: Union[List[EntailmentResult], List[PRF1Result]] = self.metric.score_batch(self.experiment_input[0], self.experiment_input[1])
            if is_EntailmentResult(results[0]):
                for x, result in zip(self.data, results):
                    x["precision"] = result["precision"]
                    x["recall"] = result["recall"]
                    x["f1"] = result["f1"]
                    x["summary_alignment"] = result["summary_alignment"]
                    x["source_alignment"] = result["source_alignment"]
                    x["summary_entailment"] = result["summary_entailment"]
                    x["source_entailment"] = result["source_entailment"]
            else:
                for x, result in zip(self.data, results):
                    x["precision"] = result["precision"]
                    x["recall"] = result["recall"]
                    x["f1"] = result["f1"]

        if isinstance(self.metric, NER):
            results: List[NERResult] = self.metric.score_batch(self.experiment_input[0], self.experiment_input[1])
            for x, result in zip(self.data, results):
                x["precision"] = result["precision"]
                x["recall"] = result["recall"]
                x["f1"] = result["f1"]
                x["summary_entities"] = result["summary_entities"]
                x["source_entities"] = result["source_entities"]
                x["summary_source_alignment"] = result["summary_source_alignment"]
                x["source_summary_alignment"] = result["source_summary_alignment"]
                x["summary_source_similarities"] = result["summary_source_similarities"]
                x["source_summary_similarities"] = result["source_summary_similarities"]

        if isinstance(self.metric, OpenIE):
            results: List[OpenIEResult] = self.metric.score_batch(self.experiment_input[0], self.experiment_input[1])
            for x, result in zip(self.data, results):
                x["precision"] = result["precision"]
                x["recall"] = result["recall"]
                x["f1"] = result["f1"]
                x["summary_source_alignment"] = result["summary_source_alignment"]
                x["source_summary_alignment"] = result["source_summary_alignment"]
                x["summary_source_similarities"] = result["summary_source_similarities"]
                x["source_summary_similarities"] = result["source_summary_similarities"]
                x["source_triples"] = result["source_triples"]
                x["summary_triples"] = result["summary_triples"]
                x["new_summary_sentences"] = result["summary_sentences"]
                x["new_source_sentences"] = result["source_sentences"]

        if isinstance(self.metric, SRL):
            results: List[SRLResult] = self.metric.score_batch(self.experiment_input[0], self.experiment_input[1])
            for x, result in zip(self.data, results):
                x["precision"] = result["precision"]
                x["recall"] = result["recall"]
                x["f1"] = result["f1"]
                x["summary_source_alignment"] = result["summary_source_alignment"]
                x["source_summary_alignment"] = result["source_summary_alignment"]
                x["summary_source_similarities"] = result["summary_source_similarities"]
                x["source_summary_similarities"] = result["source_summary_similarities"]
                x["new_summary_sentences"] = result["summary_sentences"]
                x["new_source_sentences"] = result["source_sentences"]
                x["summary_phrases"] = result["summary_phrases"]
                x["source_phrases"] = result["source_phrases"]

        # scale precision recall and f1 between 0 - 1
        for score in ["precision", "recall", "f1"]:
            scores = [x[score] for x in self.data]
            min_score = min(scores)
            max_score = max(scores)
            print(f"Scacling from {min_score} - {max_score} to 0 - 1")
            for x in self.data:
                print(f"Before: {x[score]}")
                x[score] = (x[score] - min_score) / (max_score - min_score)
                print(f"After: {x[score]}")

        # Save results as json file
        with self.out_file_json.open(encoding="UTF-8", mode="w") as file:
            json.dump(self.data, file, default=lambda a: a.__dict__)

        # calculate correlation
        self.correlation()

    def correlation(self):
        data = load_json_data(self.out_file_json)

        # check that precsion, recall f1 exists
        tmp = data[0]
        if "precision" not in tmp or "recall" not in tmp or "f1" not in tmp:
            print("ERROR: no results for precision, recall and f1")
            exit()

        # Read data
        precisions, recalls, f1s, faithfulness = zip(*[(x["precision"], x["recall"], x["f1"], x[self.data_faithfulness_key]) for x in data])

        # Calculate correlation
        table = [["Method", "Pearson", "Spearman"]]
        for scores, label in zip([precisions, recalls, f1s], ["_precision", "_recall", "_f1"]):
            table.append(
                [self.experiment_name + label, pearson(scores, faithfulness), spearman(scores, faithfulness)])

        # Save correlation as csv file
        with self.out_file_csv.open(encoding="UTF-8", mode="w") as file:
            writer = csv.writer(file)
            writer.writerows(table)

    def __load_experiment_data(self):
        self.data = load_json_data(self.data_path, self.examples)

        if self.metric.needs_input() == FaithfulnessInput.SENTENCE:
            self.experiment_input = list(zip(*[(x[self.data_summary_sentences_key], x[self.data_source_sentences_key], x[self.data_faithfulness_key]) for x in self.data]))
        elif self.metric.needs_input() == FaithfulnessInput.DOCUMENT:
            self.experiment_input = list(zip(*[(x[self.data_summary_key], x[self.data_source_key], x[self.data_faithfulness_key]) for x in self.data]))

        if self.experiment_input is None:
            print(f"ERROR: Data loading failed! (2)")
            exit()
