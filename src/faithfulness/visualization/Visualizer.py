import json
import pathlib
from typing import Type
from faithfulness.BERTScore import BERTScore
from faithfulness.Entailment import Entailment
from faithfulness.NER import NER
from faithfulness.OpenIE import OpenIE
from faithfulness.QGQA import QGQA
from faithfulness.SRL import SRL
from faithfulness.SentSim import SentSim
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.utils.utils import load_json_data


class Visualizer:
    def __init__(self, file_path: pathlib.Path, metric: Type[MetricInterface]):
        self.file_path = file_path
        self.metric: Type[MetricInterface] = metric

    def visualize(self):
        data = load_json_data(self.file_path)
        result = []
        method = ""

        if self.metric == BERTScore:
            print("BERTSCORE")
            from faithfulness.visualization.visualize_bertscore import visualize
            result = [visualize(x) for x in data]
            method = "bertscore"

        elif self.metric == Entailment:
            print("ENTAILMENT SENT")
            from faithfulness.visualization.visualize_entailment import visualize
            result = [visualize(x) for x in data]
            method = "entailment_sent"

        elif self.metric == NER:
            print("NER")
            from faithfulness.visualization.visualize_ner import visualize
            result = [visualize(x) for x in data]
            method = "ner"

        elif self.metric == OpenIE:
            print("OpenIE")
            from faithfulness.visualization.visualize_openie import visualize
            result = [visualize(x) for x in data]
            method = "openie"

        elif self.metric == QGQA:
            print("QGQA")
            from faithfulness.visualization.visualize_qgqa import visualize
            result = [visualize(x) for x in data]
            method = "qgqa"

        elif self.metric == SentSim:
            print("SentSim")
            from faithfulness.visualization.visualize_sentsim import visualize
            result = [visualize(x) for x in data]
            method = "sentsim"

        elif self.metric == SRL:
            print("SRL")
            from faithfulness.visualization.visualize_srl import visualize
            result = [visualize(x) for x in data]
            method = "srl"

        # Save output
        out_file = self.file_path.with_name(self.file_path.name.replace(".json", "_ui.json"))
        with out_file.open(encoding="UTF-8", mode="w") as file:
            json.dump({
                "method": method,
                "data": result,
            }, file)
