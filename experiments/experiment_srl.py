from pathlib import Path

from faithfulness.BERTScore import BERTScore
from faithfulness.Experimentor import Experimentor
from faithfulness.SRL import SRL, SRLMethod
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.similarity.F1 import F1
from faithfulness.similarity.SentCos import SentCos

examples = -1

# ----------------------------------------------------------------------------------------------------------------------
# SEMANTIC ROLE LABELING
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/srl")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = SRL(metric=SentCos, method=SRLMethod.NO_GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_ungrouped_sentcos",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SRL(metric=F1, method=SRLMethod.NO_GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_ungrouped_f1",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SRL(metric=ExactMatch, method=SRLMethod.NO_GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_ungrouped_em",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SRL(metric=BERTScore, method=SRLMethod.NO_GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_ungrouped_bertscore",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SRL(metric=SentCos, method=SRLMethod.GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_grouped_sentcos",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SRL(metric=F1, method=SRLMethod.GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_grouped_f1",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SRL(metric=ExactMatch, method=SRLMethod.GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_grouped_em",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = SRL(metric=BERTScore, method=SRLMethod.GROUPS, save_path=output_path, batch_mode=True)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="srl_grouped_bertscore",
                 examples=examples)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
