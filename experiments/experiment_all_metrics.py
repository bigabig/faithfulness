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


examples = 50
dataset_name = "xsum"
input_path = Path(f"./prepared_{dataset_name}.json")
metrics = ["baseline", "srl", "entailment", "bertscore", "openie", "sentsim", "qgqa", "ner"]

# ----------------------------------------------------------------------------------------------------------------------
# BASELINES
# ----------------------------------------------------------------------------------------------------------------------

if "baseline" in metrics:
    output_path = Path(f"./{dataset_name}/baseline")

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

# ----------------------------------------------------------------------------------------------------------------------
# SEMANTIC ROLE LABELING
# ----------------------------------------------------------------------------------------------------------------------

if "srl" in metrics:

    output_path = Path(f"./{dataset_name}/srl")

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

# ----------------------------------------------------------------------------------------------------------------------
# OPEN INFORMATION EXTRACTION
# ----------------------------------------------------------------------------------------------------------------------

if "openie" in metrics:

    output_path = Path(f"./{dataset_name}/openie")

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

# ----------------------------------------------------------------------------------------------------------------------
# NAMED ENTITY RECOGNITION
# ----------------------------------------------------------------------------------------------------------------------

if "ner" in metrics:

    output_path = Path(f"./{dataset_name}/ner")

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

# ----------------------------------------------------------------------------------------------------------------------
# ENTAILMENT
# ----------------------------------------------------------------------------------------------------------------------

if "entailment" in metrics:

    output_path = Path(f"./{dataset_name}/entailment")

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

# ----------------------------------------------------------------------------------------------------------------------
# BERTSCORE
# ----------------------------------------------------------------------------------------------------------------------

if "bertscore" in metrics:

    output_path = Path(f"./{dataset_name}/bertscore")

    faithfulness_metric = BERTScore(method=BERTScoreMethod.DOC, show_tqdm=True)
    e = Experimentor(data_path=input_path,
                     out_path=output_path,
                     metric=faithfulness_metric,
                     experiment_name="bertscore_doc",
                     examples=examples)
    e.experiment()
    e.correlation()
    e.binary_evaluation()

    faithfulness_metric = BERTScore(method=BERTScoreMethod.SENT, show_tqdm=True)
    e = Experimentor(data_path=input_path,
                     out_path=output_path,
                     metric=faithfulness_metric,
                     experiment_name="bertscore_sent",
                     examples=examples)
    e.experiment()
    e.correlation()
    e.binary_evaluation()

# ----------------------------------------------------------------------------------------------------------------------
# SENTSIM
# ----------------------------------------------------------------------------------------------------------------------

if "sentsim" in metrics:

    output_path = Path(f"./{dataset_name}/sentsim")

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

# ----------------------------------------------------------------------------------------------------------------------
# QGQA
# ----------------------------------------------------------------------------------------------------------------------

if "qgqa" in metrics:

    output_path = Path(f"./{dataset_name}/qgqa")

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
