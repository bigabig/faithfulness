from pathlib import Path

from faithfulness.BERTScore import BERTScore, BERTScoreMethod
from faithfulness.Experimentor import Experimentor

examples = -1

# ----------------------------------------------------------------------------------------------------------------------
# BERTSCORE
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/bertscore")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = BERTScore(method=BERTScoreMethod.DOC)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="bertscore_doc",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = BERTScore(method=BERTScoreMethod.SENT)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="bertscore_sent",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
