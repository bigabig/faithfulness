from pathlib import Path
from faithfulness.BERTScore import BERTScore
from faithfulness.Entailment import Entailment
from faithfulness.NER import NER
from faithfulness.OpenIE import OpenIE
from faithfulness.QGQA import QGQA
from faithfulness.SRL import SRL
from faithfulness.SentSim import SentSim
from faithfulness.visualization.Visualizer import Visualizer

for variant in ["good", "bad"]:
    input_file = Path(f"./xsum/analysis/bertscore_sent_examples_{variant}.json")
    Visualizer(file_path=input_file, metric=BERTScore).visualize()

    input_file = Path(f"./xsum/analysis/entailment_sent_examples_{variant}.json")
    Visualizer(file_path=input_file, metric=Entailment).visualize()

    input_file = Path(f"./xsum/analysis/ner_em_examples_{variant}.json")
    Visualizer(file_path=input_file, metric=NER).visualize()

    input_file = Path(f"./xsum/analysis/openie_f1_examples_{variant}.json")
    Visualizer(file_path=input_file, metric=OpenIE).visualize()

    input_file = Path(f"./xsum/analysis/qgqa_f1_examples_{variant}.json")
    Visualizer(file_path=input_file, metric=QGQA).visualize()

    input_file = Path(f"./xsum/analysis/sentsim_f1_examples_{variant}.json")
    Visualizer(file_path=input_file, metric=SentSim).visualize()

    input_file = Path(f"./xsum/analysis/srl_grouped_sentcos_examples_{variant}.json")
    Visualizer(file_path=input_file, metric=SRL).visualize()
