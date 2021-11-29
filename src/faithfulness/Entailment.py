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


class EntailmentMethod(Enum):
    SENT = 1,
    DOC = 2,


class Entailment(MetricInterface):

    def __init__(self, method=EntailmentMethod.SENT, modelname='facebook/bart-large-mnli', batch_size=1, max_length=None):

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

    def score(self, summary: Union[str, List[str]], source: Union[str, List[str]], additional_output: bool):
        if self.method == EntailmentMethod.SENT:
            return self.__sentencewise_entailment(summary, source, additional_output)
        if self.method == EntailmentMethod.DOC:
            return self.__documentwise_entailment(summary, source)

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

        # use model to predict outputs
        outputs = None
        for data in dataloader:
            source_batch = data['source']
            summary_batch = data['summary']

            # create model inputs
            inputs = self.tokenizer(source_batch, summary_batch, return_tensors="pt", padding=True, truncation=TruncationStrategy.ONLY_FIRST, max_length=self.max_length)
            inputs.to(self.device)

            # predict output
            output = self.model(**inputs)

            # concatenate output to all outputs
            if outputs is None:
                outputs = output['logits'].detach()
            else:
                outputs = torch.cat((outputs, output['logits'].detach()))

        # convert logits into percentages
        scores = F.softmax(outputs, dim=1)  # shape (#sentence pairs, #labels)

        # reshape to (#source_sentences, #summary_sentences, #scores)
        scores = scores.reshape(len(source_sentences), len(summary_sentences), self.num_labels)

        # extract entailment probabilities
        entailment_scores = scores[:, :, self.entailment_id]

        # for every summary sentence find source sentences that have the highest entailment score
        values, indices = entailment_scores.max(dim=0)
        max_entailment_scores = values  # e.g. [0.9, 0.8] means that summary sentence 0 is entailed with 90% by a source sentence and summary sentence 1 is entailed with 80% by a source sentence
        entailed_by = indices  # e.g. [0, 4] means that summary sentence 0 is entailed by source sentence 0 and summary sentence 1 is entailed by source sentence 4

        if additional_output:
            return {
                "score": max_entailment_scores.mean().item(),
                "alignment": entailed_by.tolist(),
                "entailment": max_entailment_scores.tolist(),
            }
        return {"score": max_entailment_scores.mean().item()}

    def __sentencewise_entailment_batch(self, summaries: List[List[str]], sources: List[List[str]], additional_output: bool):
        results = {}
        for (summary, source) in tqdm(zip(summaries, sources)):
            result = self.__sentencewise_entailment(summary, source, additional_output)
            for key, value in result.items():
                results[key] = [*results.get(key, []), value]
        return results

    def __documentwise_entailment(self, summary, source):
        # predict entailment
        inputs = self.tokenizer(source, summary, return_tensors="pt", truncation=TruncationStrategy.ONLY_FIRST, max_length=self.max_length)
        inputs.to(self.device)
        output = self.model(**inputs)  # e.g. for roberta-mnli [contradiction_score, neutral_score, entailment_score]
        outputs = output['logits'].detach()

        # calculate entailment score
        all_scores = F.softmax(outputs, dim=1)
        score = all_scores[:, self.entailment_id].mean().item()

        return {"score": score}

    def __documentwise_entailment_batch(self, summaries: List[str], sources: List[str]):
        # load data into dataset
        assert len(summaries) == len(sources)
        dataloader = DataLoader(SummarizationDataset(summaries, sources), batch_size=self.batch_size, shuffle=False)

        result = []
        for batch in tqdm(dataloader,  desc="Calculating documentwise entailment..."):
            batch_summaries = batch["summaries"]
            batch_sources = batch["sources"]

            # tokenize truncated input
            inputs = self.tokenizer(batch_sources, batch_summaries, return_tensors="pt", padding=True, truncation=TruncationStrategy.ONLY_FIRST, max_length=self.max_length)
            inputs.to(self.device)

            # calc entailment
            output = self.model(**inputs)
            outputs = output['logits'].detach()  # e.g. for roberta-mnli [contradiction_score, neutral_score, entailment_score]
            scores = F.softmax(outputs, dim=1)  # convert to probabilities
            result.extend(scores[:, self.entailment_id].tolist())

        return {"score": result}
