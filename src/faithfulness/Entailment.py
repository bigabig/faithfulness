from typing import List, Union, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import TruncationStrategy
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.utils.Datasets import SimpleDataset, SummarizationDataset
import torch.nn.functional as F
from enum import Enum
from tqdm import tqdm
from faithfulness.utils.utils import PRF1Result


class EntailmentMethod(Enum):
    SENT = 1,
    DOC = 2,


class EntailmentResult(PRF1Result):
    summary_alignment: List[int]
    source_alignment: List[int]
    summary_entailment: List[float]
    source_entailment: List[float]


def is_EntailmentResult(obj) -> bool:
    return "summary_entailment" in obj and "source_entailment" in obj


class Entailment(MetricInterface):

    def __init__(self, method=EntailmentMethod.SENT, modelname: str = 'facebook/bart-large-mnli', batch_size: int = 1, max_length: Optional[int] = None):

        print(f'Loading entailment model {modelname}...')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(modelname)
        model.to(device)

        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = batch_size
        self.max_length = max_length if max_length is not None else tokenizer.model_max_length

        self.num_labels = model.config._num_labels
        if modelname == "roberta-large-mnli":
            self.entailment_id = model.config.label2id["ENTAILMENT"]
        if modelname == "facebook/bart-large-mnli":
            self.entailment_id = model.config.label2id["entailment"]

    def needs_input(self) -> FaithfulnessInput:
        return FaithfulnessInput.SENTENCE if self.method == EntailmentMethod.SENT else FaithfulnessInput.DOCUMENT

    def score(self, summary: Union[str, List[str]], source: Union[str, List[str]]) -> Union[EntailmentResult, PRF1Result]:
        if self.method == EntailmentMethod.SENT:
            return self.__sentencewise_entailment(summary, source)
        if self.method == EntailmentMethod.DOC:
            return self.__documentwise_entailment(summary, source)

    def score_batch(self, summaries: Union[List[str], List[List[str]]], sources: Union[List[str], List[List[str]]]) -> Union[List[EntailmentResult], List[PRF1Result]]:
        if self.method == EntailmentMethod.SENT:
            return self.__sentencewise_entailment_batch(summaries, sources)
        if self.method == EntailmentMethod.DOC:
            return self.__documentwise_entailment_batch(summaries, sources)

    def __sentencewise_entailment(self, summary_sentences: List[str], source_sentences: List[str]) -> EntailmentResult:
        # create all source-summary sentence pairs
        pairs = []
        for source_sentence in source_sentences:
            for summary_sentence in summary_sentences:
                pairs.append({"source": source_sentence, "summary": summary_sentence})
        dataloader = DataLoader(SimpleDataset(pairs), batch_size=self.batch_size, shuffle=False)

        # use model to predict outputs
        score = {}
        entailment = {}
        alignment = {}

        for variant in [True, False]:
            scores = []

            for data in dataloader:
                source_batch = data['source']
                summary_batch = data['summary']

                a = source_batch if variant else summary_batch
                b = summary_batch if variant else source_batch
                truncation_strategy = TruncationStrategy.ONLY_FIRST if variant else TruncationStrategy.ONLY_SECOND

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
            e, a = scores.max(dim=0 if variant else 1)

            score[variant] = e.mean().item()
            entailment[variant] = e.tolist()
            alignment[variant] = a.tolist()

        precision = score[True]
        recall = score[False]
        if (precision + recall) > 0.0:
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            f1 = 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "summary_alignment": alignment[True],
            "source_alignment": alignment[False],
            "summary_entailment": entailment[True],
            "source_entailment": entailment[False],
        }

    def __sentencewise_entailment_batch(self, summaries: List[List[str]], sources: List[List[str]]) -> List[EntailmentResult]:
        results: List[EntailmentResult] = []
        for (summary, source) in tqdm(zip(summaries, sources)):
            results.append(self.__sentencewise_entailment(summary, source))
        return results

    def __documentwise_entailment(self, summary, source) -> PRF1Result:
        score = {}
        for variant in [True, False]:
            a = source if variant else summary
            b = summary if variant else source
            truncation_strategy = TruncationStrategy.ONLY_FIRST if variant else TruncationStrategy.ONLY_SECOND

            # predict entailment
            inputs = self.tokenizer(a, b, return_tensors="pt", truncation=truncation_strategy, max_length=self.max_length)
            inputs.to(self.device)
            output = self.model(**inputs)  # e.g. for roberta-mnli [contradiction_score, neutral_score, entailment_score]
            outputs = output['logits'].detach()

            # calculate entailment score
            all_scores = F.softmax(outputs, dim=1)
            score[variant] = all_scores[:, self.entailment_id].mean().item()

        precision = score[True]
        recall = score[False]
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def __documentwise_entailment_batch(self, summaries: List[str], sources: List[str]) -> List[PRF1Result]:
        # load data into dataset
        assert len(summaries) == len(sources)
        dataloader = DataLoader(SummarizationDataset(summaries, sources), batch_size=self.batch_size, shuffle=False)

        results = {}
        for variant in [True, False]:
            results[variant] = []

        for batch in tqdm(dataloader,  desc="Calculating documentwise entailment..."):
            batch_summaries = batch["summaries"]
            batch_sources = batch["sources"]

            for variant in [True, False]:
                a = batch_sources if variant else batch_summaries
                b = batch_summaries if variant else batch_sources
                truncation_strategy = TruncationStrategy.ONLY_FIRST if variant else TruncationStrategy.ONLY_SECOND

                # tokenize truncated input
                inputs = self.tokenizer(a, b, return_tensors="pt", padding=True, truncation=truncation_strategy, max_length=self.max_length)
                inputs.to(self.device)

                # calc entailment
                output = self.model(**inputs)
                outputs = output['logits'].detach()  # e.g. for roberta-mnli [contradiction_score, neutral_score, entailment_score]
                scores = F.softmax(outputs, dim=1)  # convert to probabilities
                results[variant].extend(scores[:, self.entailment_id].tolist())

        precisions = results[True]
        recalls = results[False]
        return [{"precision": precision,
                 "recall": recall,
                 "f1": 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
                 } for precision, recall in zip(precisions, recalls)]
