from pathlib import Path

from faithfulness.BERTScore import BERTScore
from faithfulness.Experimentor import Experimentor
from faithfulness.QGQA import QGQA
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.similarity.F1 import F1
from faithfulness.similarity.SentCos import SentCos

examples = -1

# ----------------------------------------------------------------------------------------------------------------------
# Question Generation & Question Answering
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/qgqa")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = QGQA(metric=SentCos, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="qgqa_sentcos",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = QGQA(metric=F1, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="qgqa_f1",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = QGQA(metric=ExactMatch, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="qgqa_em",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

# do not use BERTScore SENT variant, the score_batch method of BERTScore SENT expects List[List[str]] but QGQA will only input List[str] of answers to compare
faithfulness_metric = QGQA(metric=BERTScore, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="qgqa_bertscore",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
