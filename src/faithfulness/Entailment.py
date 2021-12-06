from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import TruncationStrategy

from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.utils.Datasets import SimpleDataset, SummarizationDataset
import torch.nn.functional as F
from enum import Enum
from tqdm import tqdm
from faithfulness.utils.utils import MetricVariant


class EntailmentMethod(Enum):
    SENT = 1,
    DOC = 2,


class Entailment(MetricInterface):

    def __init__(self, method=EntailmentMethod.SENT, modelname='facebook/bart-large-mnli', batch_size=1, max_length=None, variant=MetricVariant.F1):

        print(f'Loading entailment model {modelname}...')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(modelname)
        model.to(device)

        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.variant = variant

        self.batch_size = batch_size
        self.max_length = max_length if max_length is not None else tokenizer.model_max_length

        self.num_labels = model.config._num_labels
        if modelname == "roberta-large-mnli":
            self.entailment_id = model.config.label2id["ENTAILMENT"]
        if modelname == "facebook/bart-large-mnli":
            self.entailment_id = model.config.label2id["entailment"]

    def score(self, summary: Union[str, List[str]], source: Union[str, List[str]], additional_output: bool):
        if self.method == EntailmentMethod.SENT:
            return self.__sentencewise_entailment(summary, source, additional_output)
        if self.method == EntailmentMethod.DOC:
            return self.__documentwise_entailment(summary, source, additional_output)

    def score_batch(self, summaries: Union[List[str], List[List[str]]], sources: Union[List[str], List[List[str]]], additional_output: bool):
        if self.method == EntailmentMethod.SENT:
            return self.__sentencewise_entailment_batch(summaries, sources, additional_output)
        if self.method == EntailmentMethod.DOC:
            return self.__documentwise_entailment_batch(summaries, sources)

    def __sentencewise_entailment(self, summary_sentences: List[str], source_sentences: List[str], additional_output: bool):
        # create all source-summary sentence pairs
        pairs = []
        for source_sentence in source_sentences:
            for summary_sentence in summary_sentences:
                pairs.append({"source": source_sentence, "summary": summary_sentence})
        dataloader = DataLoader(SimpleDataset(pairs), batch_size=self.batch_size, shuffle=False)

        variants = [MetricVariant.PRECISION, MetricVariant.RECALL] if self.variant == MetricVariant.F1 else [self.variant]

        # use model to predict outputs
        score = {}
        entailment = {}
        alignment = {}

        for variant in variants:
            scores = []

            for data in dataloader:
                source_batch = data['source']
                summary_batch = data['summary']

                a = source_batch if variant == MetricVariant.PRECISION else summary_batch
                b = summary_batch if variant == MetricVariant.PRECISION else source_batch
                truncation_strategy = TruncationStrategy.ONLY_FIRST if variant == MetricVariant.PRECISION else TruncationStrategy.ONLY_SECOND

                # create model inputs
                inputs = self.tokenizer(a, b, return_tensors="pt", padding=True, truncation=truncation_strategy, max_length=self.max_length)
                inputs.to(self.device)

                # predict output
                output = self.model(**inputs)
                output = output['logits'].detach()

                # convert to probabilities
                output = F.softmax(output, dim=1)

                # extract entailment probabilities & append
                scores.extend(output[:, self.entailment_id].tolist())

            # reshape to (#source_sentences, #summary_sentences, #scores)
            scores = torch.tensor(scores).reshape(len(source_sentences), len(summary_sentences))

            # entailment: e.g. [0.9, 0.8] means that summary sentence 0 is entailed with 90% by a source sentence and summary sentence 1 is entailed with 80% by a source sentence
            # alignment: e.g. [0, 4] means that summary sentence 0 is entailed by source sentence 0 and summary sentence 1 is entailed by source sentence 4
            e, a = scores.max(dim=0 if variant == MetricVariant.PRECISION else 1)

            score[variant.value] = e.mean().item()
            entailment[variant.value] = e.tolist()
            alignment[variant.value] = a.tolist()

        if self.variant == MetricVariant.PRECISION or self.variant == MetricVariant.RECALL:
            if additional_output:
                return {
                    "score": score[self.variant.value],
                    "alignment": alignment[self.variant.value],
                    "entailment": entailment[self.variant.value],
                }
            return {"score": score[self.variant.value]}

        else:
            precision = score[MetricVariant.PRECISION.value]
            recall = score[MetricVariant.RECALL.value]
            if (precision + recall) > 0.0:
                f1 = 2 * ((precision * recall) / (precision + recall))
            else:
                f1 = 0.0
            if additional_output:
                return {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "summary_alignment": alignment[MetricVariant.PRECISION.value],
                    "source_alignment": alignment[MetricVariant.RECALL.value],
                    "summary_entailment": entailment[MetricVariant.PRECISION.value],
                    "source_entailment": entailment[MetricVariant.RECALL.value],
                }
            return {"score": f1}

    def __sentencewise_entailment_batch(self, summaries: List[List[str]], sources: List[List[str]], additional_output: bool):
        results = {}
        for (summary, source) in tqdm(zip(summaries, sources)):
            result = self.__sentencewise_entailment(summary, source, additional_output)
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]
        return results

    def __documentwise_entailment(self, summary, source, additional_output: bool):
        variants = [MetricVariant.PRECISION, MetricVariant.RECALL] if self.variant == MetricVariant.F1 else [self.variant]

        score = {}
        for variant in variants:
            a = source if variant == MetricVariant.PRECISION else summary
            b = summary if variant == MetricVariant.PRECISION else source
            truncation_strategy = TruncationStrategy.ONLY_FIRST if variant == MetricVariant.PRECISION else TruncationStrategy.ONLY_SECOND

            # predict entailment
            inputs = self.tokenizer(a, b, return_tensors="pt", truncation=truncation_strategy, max_length=self.max_length)
            inputs.to(self.device)
            output = self.model(**inputs)  # e.g. for roberta-mnli [contradiction_score, neutral_score, entailment_score]
            outputs = output['logits'].detach()

            # calculate entailment score
            all_scores = F.softmax(outputs, dim=1)
            score[variant.value] = all_scores[:, self.entailment_id].mean().item()

        if self.variant == MetricVariant.F1:
            precision = score[MetricVariant.PRECISION.value]
            recall = score[MetricVariant.RECALL.value]
            f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
            if additional_output:
                return {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            else:
                return {"score": f1}
        return {"score": score[self.variant.value]}

    def __documentwise_entailment_batch(self, summaries: List[str], sources: List[str], additional_output: bool):
        # load data into dataset
        assert len(summaries) == len(sources)
        dataloader = DataLoader(SummarizationDataset(summaries, sources), batch_size=self.batch_size, shuffle=False)

        variants = [MetricVariant.PRECISION, MetricVariant.RECALL] if self.variant == MetricVariant.F1 else [self.variant]

        results = {}
        for variant in variants:
            results[variant.value] = []

        for batch in tqdm(dataloader,  desc="Calculating documentwise entailment..."):
            batch_summaries = batch["summaries"]
            batch_sources = batch["sources"]

            for variant in variants:
                a = batch_sources if variant == MetricVariant.PRECISION else batch_summaries
                b = batch_summaries if variant == MetricVariant.PRECISION else batch_sources
                truncation_strategy = TruncationStrategy.ONLY_FIRST if variant == MetricVariant.PRECISION else TruncationStrategy.ONLY_SECOND

                # tokenize truncated input
                inputs = self.tokenizer(a, b, return_tensors="pt", padding=True, truncation=truncation_strategy.ONLY_FIRST, max_length=self.max_length)
                inputs.to(self.device)

                # calc entailment
                output = self.model(**inputs)
                outputs = output['logits'].detach()  # e.g. for roberta-mnli [contradiction_score, neutral_score, entailment_score]
                scores = F.softmax(outputs, dim=1)  # convert to probabilities
                results[variant.value].extend(scores[:, self.entailment_id].tolist())

        if self.variant == MetricVariant.F1:
            precisions = results[MetricVariant.PRECISION.value]
            recalls = results[MetricVariant.RECALL.value]
            f1s = [2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0 for precision, recall in zip(precisions, recalls)]
            if additional_output:
                return {
                    "precision": precisions,
                    "recall": recalls,
                    "f1": f1s
                }
            else:
                return {"score": f1s}
        return {"score": results[self.variant.value]}
