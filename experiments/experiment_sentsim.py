from pathlib import Path

from faithfulness.BERTScore import BERTScore
from faithfulness.Experimentor import Experimentor
from faithfulness.SentSim import SentSim
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.similarity.F1 import F1
from faithfulness.similarity.SentCos import SentCos

examples = -1

# ----------------------------------------------------------------------------------------------------------------------
# SENTSIM
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/sentsim")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = SentSim(metric=SentCos)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="sentsim_sentcos",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SentSim(metric=F1)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="sentsim_f1",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SentSim(metric=ExactMatch)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="sentsim_em",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SentSim(metric=BERTScore)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="sentsim_bertscore",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
