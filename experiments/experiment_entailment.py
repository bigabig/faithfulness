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
# ENTAILMENT
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/entailment")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = Entailment(method=EntailmentMethod.SENT, max_length=512)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="entailment_sent",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = Entailment(method=EntailmentMethod.DOC, max_length=512)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="entailment_doc",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
