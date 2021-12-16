from pathlib import Path
from faithfulness.Baseline import Baseline, BaselineMethod
from faithfulness.Entailment import Entailment, EntailmentMethod
from faithfulness.BERTScore import BERTScore, BERTScoreMethod
from faithfulness.Experimentor import Experimentor
from faithfulness.NER import NER
from faithfulness.OpenIE import OpenIE
from faithfulness.QGQA import QGQA
from faithfulness.SRL import SRL, SRLMethod
from faithfulness.SentSim import SentSim
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.similarity.F1 import F1
from faithfulness.similarity.SentCos import SentCos

examples = -1

# ----------------------------------------------------------------------------------------------------------------------
# OPEN INFORMATION EXTRACTION
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/openie")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = OpenIE(metric=SentCos, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="openie_sentcos",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = OpenIE(metric=F1, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="openie_f1",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = OpenIE(metric=ExactMatch, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="openie_em",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = OpenIE(metric=BERTScore, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="openie_bertscore",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
