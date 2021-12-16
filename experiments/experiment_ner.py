from pathlib import Path

from faithfulness.BERTScore import BERTScore
from faithfulness.Experimentor import Experimentor
from faithfulness.NER import NER
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.similarity.F1 import F1
from faithfulness.similarity.SentCos import SentCos

examples = -1

# ----------------------------------------------------------------------------------------------------------------------
# NAMED ENTITY RECOGNITION
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/ner")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = NER(metric=SentCos)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="ner_sentcos",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = NER(metric=F1)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="ner_f1",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = NER(metric=ExactMatch)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="ner_em",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = NER(metric=BERTScore)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="ner_bertscore",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
