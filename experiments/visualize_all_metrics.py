from pathlib import Path
from faithfulness.BERTScore import BERTScore
from faithfulness.Entailment import Entailment
from faithfulness.NER import NER
from faithfulness.OpenIE import OpenIE
from faithfulness.QGQA import QGQA
from faithfulness.SRL import SRL
from faithfulness.SentSim import SentSim
from faithfulness.visualization.Visualizer import Visualizer

input_file = Path("./xsum/bertscore/bertscore_doc.json")
Visualizer(file_path=input_file, metric=BERTScore).visualize()

input_file = Path("./xsum/entailment/entailment_sent.json")
Visualizer(file_path=input_file, metric=Entailment).visualize()

input_file = Path("./xsum/ner/ner_em.json")
Visualizer(file_path=input_file, metric=NER).visualize()

input_file = Path("./xsum/openie/openie_em.json")
Visualizer(file_path=input_file, metric=OpenIE).visualize()

input_file = Path("./xsum/qgqa/qgqa_em.json")
Visualizer(file_path=input_file, metric=QGQA).visualize()

input_file = Path("./xsum/sentsim/sentsim_em.json")
Visualizer(file_path=input_file, metric=SentSim).visualize()

input_file = Path("./xsum/srl/srl_em.json")
Visualizer(file_path=input_file, metric=SRL).visualize()
