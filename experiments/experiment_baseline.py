from pathlib import Path
from faithfulness.Baseline import Baseline, BaselineMethod
from faithfulness.Experimentor import Experimentor

examples = -1

# ----------------------------------------------------------------------------------------------------------------------
# BASELINES
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/baseline")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = Baseline(method=BaselineMethod.ONE)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="baseline_one",
                 examples=examples,
                 scale=False)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = Baseline(method=BaselineMethod.ZERO)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="baseline_zero",
                 examples=examples,
                 scale=False)
e.experiment()
e.correlation()
e.binary_evaluation()

faithfulness_metric = Baseline(method=BaselineMethod.RANDOM)
e = Experimentor(data_path=input_path,
                 out_path=output_path,
                 metric=faithfulness_metric,
                 experiment_name="baseline_random",
                 examples=examples,
                 scale=False)
e.experiment()
e.correlation()
e.binary_evaluation()

print("FINISHED!")
