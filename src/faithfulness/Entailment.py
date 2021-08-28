import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from faithfulness.utils.Datasets import SimpleDataset
import torch.nn.functional as F
from enum import Enum


class EntailmentMethod(Enum):
    SENT = 1,
    DOC = 2,


class Entailment:

    def __init__(self, method=EntailmentMethod.SENT, modelname='roberta-large-mnli', spacymodel='en_core_web_lg'):

        print(f'Loading entailment model {modelname}...')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(modelname)
        model.to(device)

        if method == EntailmentMethod.SENT:
            print(f'Loading Spacy model {spacymodel}...')
            self.nlp = spacy.load(spacymodel)

        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.num_labels = model.config._num_labels
        self.entailment_id = model.config.label2id["ENTAILMENT"]

    def eval(self, summary, source):
        if self.method == EntailmentMethod.SENT:
            return self.__sentencewise_entailment(summary, source)

        if self.method == EntailmentMethod.DOC:
            return self.__documentwise_entailment(summary, source)

    def __split_sentences(self, text):
        if self.method != EntailmentMethod.SENT:
            print(f"ERROR: Sentence splitting is only supported when using 'EntailmentMethod.SENT'.")
            exit()

        return [x.text for x in self.nlp(text).sents]

    def __sentencewise_entailment(self, summary, source):
        # split sentences
        summary_sentences = self.__split_sentences(summary)
        source_sentences = self.__split_sentences(source)

        # create all source-summary sentence pairs
        # all pairs will have a maximum sequence length of 512 tokens
        # to achieve this, the source sentence may be shortened! the summary will always be full length
        pairs = []
        for source_sentence in source_sentences:
            for summary_sentence in summary_sentences:
                source_tokens = self.tokenizer.encode(source_sentence, add_special_tokens=False)
                summary_tokens = self.tokenizer.encode(summary_sentence, add_special_tokens=False)
                if len(source_tokens) + len(summary_tokens) > self.tokenizer.max_len_sentences_pair:
                    source_tokens = source_tokens[:(len(source_tokens) + len(summary_tokens) - self.tokenizer.max_len_sentences_pair)]
                    pairs.append({"source": self.tokenizer.decode(source_tokens), "summary": summary_sentence})
                else:
                    pairs.append({"source": source_sentence, "summary": summary_sentence})

        dataset = SimpleDataset(pairs)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # use model to predict outputs
        outputs = None
        for data in dataloader:
            source_batch = data['source']
            summary_batch = data['summary']

            # create model inputs
            inputs = self.tokenizer(source_batch, summary_batch, return_tensors="pt", padding=True)
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

        return max_entailment_scores.mean().item(), entailed_by.tolist(), max_entailment_scores

    # todo: add correct document wise entailment!
    def __documentwise_entailment(self, summary, source):
        # tokenize input
        source_tokens = self.tokenizer(source, add_special_tokens=False)['input_ids']
        summary_tokens = self.tokenizer(summary, add_special_tokens=False)['input_ids']

        # make sure that input is maximum 512 tokens
        # we cut the source document to fit into this limitation
        # - 3 for start, end and sep token.
        if len(source_tokens) + len(summary_tokens) > self.tokenizer.max_len_sentences_pair:
            source_tokens = source_tokens[:(len(source_tokens) + len(summary_tokens) - self.tokenizer.max_len_sentences_pair)]
            pair = {"source": self.tokenizer.decode(source_tokens), "summary": summary}
        else:
            pair = {"source": source, "summary": summary}

        # predict entailment
        inputs = self.tokenizer(pair['source'], pair['summary'], return_tensors="pt")
        inputs.to(self.device)
        output = self.model(**inputs)  # e.g. for roberta-mnli [contradiction_score, neutral_score, entailment_score]
        outputs = output['logits'].detach()

        # calculate entailment score
        all_scores = F.softmax(outputs, dim=1)
        score = all_scores[:, self.entailment_id].mean().item()

        return score
