from pathlib import Path

from faithfulness.Baseline import Baseline, BaselineMethod
from faithfulness.Entailment import Entailment, EntailmentMethod
from faithfulness.BERTScore import BERTScore, BERTScoreMethod
from faithfulness.Experimentor import Experimentor
from faithfulness.NER import NER
from faithfulness.OpenIE import OpenIE
from faithfulness.QGQA import QGQA
from faithfulness.SRL import SRL
from faithfulness.SentSim import SentSim
from faithfulness.similarity.ExactMatch import ExactMatch
from faithfulness.similarity.F1 import F1
from faithfulness.similarity.SentCos import SentCos

examples = 10

# ----------------------------------------------------------------------------------------------------------------------
# BASELINES
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/baseline")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = Baseline(method=BaselineMethod.ONE)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="baseline_one",
             examples=examples,
             scale=False).experiment()

faithfulness_metric = Baseline(method=BaselineMethod.ZERO)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="baseline_zero",
             examples=examples,
             scale=False).experiment()

faithfulness_metric = Baseline(method=BaselineMethod.RANDOM)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="baseline_random",
             examples=examples,
             scale=False).experiment()

# print("FINISHED!")
# exit()

# ----------------------------------------------------------------------------------------------------------------------
# SEMANTIC ROLE LABELING
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/srl")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = SRL(metric=SentCos, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="srl_sentcos",
             examples=examples).experiment()

faithfulness_metric = SRL(metric=F1, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="srl_f1",
             examples=examples).experiment()

faithfulness_metric = SRL(metric=ExactMatch, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="srl_em",
             examples=examples).experiment()

faithfulness_metric = SRL(metric=BERTScore, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="srl_bertscore",
             examples=examples).experiment()

# print("FINISHED!")
# exit()

# ----------------------------------------------------------------------------------------------------------------------
# OPEN INFORMATION EXTRACTION
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/openie")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = OpenIE(metric=SentCos, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="openie_sentcos",
             examples=examples).experiment()

faithfulness_metric = OpenIE(metric=F1, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="openie_f1",
             examples=examples).experiment()

faithfulness_metric = OpenIE(metric=ExactMatch, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="openie_em",
             examples=examples).experiment()

faithfulness_metric = OpenIE(metric=BERTScore, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="openie_bertscore",
             examples=examples).experiment()

# print("FINISHED!")
# exit()

# ----------------------------------------------------------------------------------------------------------------------
# NAMED ENTITY RECOGNITION
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/ner")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = NER(metric=SentCos)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="ner_sentcos",
             examples=examples).experiment()

faithfulness_metric = NER(metric=F1)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="ner_f1",
             examples=examples).experiment()

faithfulness_metric = NER(metric=ExactMatch)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="ner_em",
             examples=examples).experiment()

faithfulness_metric = NER(metric=BERTScore)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="ner_bertscore",
             examples=examples).experiment()

# print("FINISHED!")
# exit()

# ----------------------------------------------------------------------------------------------------------------------
# ENTAILMENT
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/entailment")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = Entailment(method=EntailmentMethod.SENT, max_length=512)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="entailment_sent",
             examples=examples).experiment()

faithfulness_metric = Entailment(method=EntailmentMethod.DOC, max_length=512)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="entailment_doc",
             examples=examples).experiment()

# print("FINISHED!")
# exit()

# ----------------------------------------------------------------------------------------------------------------------
# BERTSCORE
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/bertscore")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = BERTScore(method=BERTScoreMethod.DOC)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="bertscore_doc",
             examples=examples).experiment()

faithfulness_metric = BERTScore(method=BERTScoreMethod.SENT)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="bertscore_sent",
             examples=examples).experiment()

# print("FINISHED!")
# exit()

# ----------------------------------------------------------------------------------------------------------------------
# SENTSIM
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/sentsim")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = SentSim(metric=SentCos)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="sentsim_sentcos",
             examples=examples).experiment()

faithfulness_metric = SentSim(metric=F1)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="sentsim_f1",
             examples=examples).experiment()

faithfulness_metric = SentSim(metric=ExactMatch)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="sentsim_em",
             examples=examples).experiment()

faithfulness_metric = SentSim(metric=BERTScore)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="sentsim_bertscore",
             examples=examples).experiment()

# print("FINISHED!")
# exit()

# ----------------------------------------------------------------------------------------------------------------------
# QGQA
# ----------------------------------------------------------------------------------------------------------------------

output_path = Path("./xsum/qgqa")
input_path = Path("./prepared_xsum.json")

faithfulness_metric = QGQA(metric=SentCos, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="qgqa_sentcos",
             examples=examples).experiment()

faithfulness_metric = QGQA(metric=F1, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="qgqa_f1",
             examples=examples).experiment()

faithfulness_metric = QGQA(metric=ExactMatch, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="qgqa_em",
             examples=examples).experiment()

# do not use BERTScore SENT variant, the score_batch method of BERTScore SENT expects List[List[str]] but QGQA will only input List[str] of answers to compare
faithfulness_metric = QGQA(metric=BERTScore, save_path=output_path, batch_mode=True)
Experimentor(data_path=input_path,
             out_path=output_path,
             metric=faithfulness_metric,
             experiment_name="qgqa_bertscore",
             examples=examples).experiment()

# print("FINISHED!")
# exit()
