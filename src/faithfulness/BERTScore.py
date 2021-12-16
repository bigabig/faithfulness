from enum import Enum
from typing import List, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from faithfulness.types.AlignScoreResult import AlignScoreResult
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.Datasets import SimpleDataset, SummarizationDataset
from faithfulness.utils.utils import calc_prf1, PRF1Result
from tqdm import tqdm
import itertools


class PRF1SimResult(PRF1Result):
    similarities: List[List[float]]


class BERTScoreResult(PRF1SimResult):
    summary_tokens: List[int]
    source_tokens: List[int]


class BERTScoreMethod(Enum):
    SENT = 1,  # sentences longer than 512 tokens are truncated
    DOC = 2,  # documents longer than 512 tokens are truncated


class BERTScore(SimilarityMetricInterface):

    def __init__(self, modelname="roberta-large-mnli", layer=8, method=BERTScoreMethod.DOC, batch_size=2, show_tqdm=False):
        """
        Evaluate summaries and sources using BERTScore
        :param modelname: this model calculates token embeddings
        :param layer: this layer's embeddings are used as token embeddings
        :param method: calculate BERTScore for whole documents (truncated if longer than the model's max input length) or per sentence (truncated if sentence is longer than the model's max inmput length). Does NOT affect align_and_score().
        :param batch_size: when using score_batch, how many documents / sentences should be processed in parallel (depends on your GPU)
        """

        print(f"Init BERTScore model {modelname}...")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        model = AutoModel.from_pretrained(modelname)
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = device
        self.method = method
        self.batch_size = batch_size
        self.show_tqdm = show_tqdm

    def needs_input(self) -> FaithfulnessInput:
        return FaithfulnessInput.SENTENCE if self.method == BERTScoreMethod.SENT else FaithfulnessInput.DOCUMENT

    def score(self, summary: Union[str, List[str]], source: Union[str, List[str]]) -> BERTScoreResult:
        """
        Calculate BERTScore for a summary and corresponding source document.
        :param summary: string (a summary document) [If BERTScoreMethod.DOC] or List[str] (a summary document consisting of multiple sentences) [If BERTScoreMethod.SENT]
        :param source: string (a source document) [If BERTScoreMethod.DOC] or List[str] (a source document consisting of multiple sentences) [If BERTScoreMethod.SENT]
        :return: {precision, recall, f1, similarities, summary_tokens, source_tokens}
        """
        if self.method == BERTScoreMethod.DOC:
            return self.__score_document(summary, source)
        if self.method == BERTScoreMethod.SENT:
            return self.__score_sentences(summary, source)

    def __score_document(self, summary_text: str, source_text: str) -> BERTScoreResult:
        """
        Calculate BERTScore for a summary and corresponding source text, each consisting of a single string.
        Documents longer than 512 tokens (depending on the model) are truncated.
        :param summary_text: summary consisting of a single string
        :param source_text: source consisting of a single string
        :return: {precision, recall, f1, similarities, summary_tokens, source_tokens}
        """
        # embed summary & source
        summary_embeddings, summary_tokens = self.__embed(summary_text)
        source_embeddings, source_tokens = self.__embed(source_text)

        # calculate bertscore
        result = BERTScore.__calc_bertscore(summary_embeddings, source_embeddings)
        return {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "similarities": result["similarities"],
            "summary_tokens": summary_tokens,
            "source_tokens": source_tokens
        }

    def __score_sentences(self, summary_sentences: List[str], source_sentences: List[str]) -> BERTScoreResult:
        """
        Calculate BERTScore for a summary and corresponding source text, each consisting of multiple sentences (a list of strings)
        Sentences longer than 512 tokens (depending on the model) are truncated.
        :param summary_sentences: summary consisting of multiple sentences (List of strings)
        :param source_sentences: summary consisting of multiple sentences (List of strings)
        :return: {precision, recall, f1, similarities, summary_tokens, source_tokens}
        """
        # embed summary & source
        summary_embeddings, summary_tokens = self.__embed_batch(summary_sentences, False)
        source_embeddings, source_tokens = self.__embed_batch(source_sentences, False)

        # calculate bertscore
        result = BERTScore.__calc_bertscore(summary_embeddings, source_embeddings)
        return {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "similarities": result["similarities"],
            "summary_tokens": summary_tokens,
            "source_tokens": source_tokens
        }

    def score_batch(self, summaries: Union[List[str], List[List[str]]], sources: Union[List[str], List[List[str]]]) -> List[BERTScoreResult]:
        """
        Calculate BERTScore for a batch of summaries and their corresponding source documents.
        :param summaries: batch of summary documents (List[str]) [If BERTScoreMethod.DOC] or batch of summary documents with sentences (List[List[str]]) [If BERTScoreMethod.SENT]
        :param sources: batch of source documents (List[str]) [If BERTScoreMethod.DOC] or batch of source documents with sentences (List[List[str]]) [If BERTScoreMethod.SENT]
        :return: {precision, recall, f1, similarities, summary_tokens, source_tokens}
        """
        if self.method == BERTScoreMethod.DOC:
            return self.__score_batch_document(summaries, sources)
        if self.method == BERTScoreMethod.SENT:
            return self.__score_batch_sentences(summaries, sources)

    def __score_batch_document(self, summaries: List[str], sources: List[str]) -> List[BERTScoreResult]:
        """
        Calculate BERTScore for batches of summaries and sources (a document is expected to be a simple string)
        Documents longer than 512 tokens (depending on the model) are truncated.
        :param summaries: list of summary documents / texts
        :param sources: list of source documents / texts
        :return: List of {precision, recall, f1, similarities, summary_tokens, source_tokens}
        """
        assert len(summaries) == len(sources)

        dataloader = DataLoader(SummarizationDataset(summaries, sources), batch_size=32, shuffle=False)

        results: List[BERTScoreResult] = []
        tqdm_args = {"desc": "Calculating bertscore..."} if self.show_tqdm else {}
        for batch in tqdm(dataloader, *tqdm_args):
            batch_summaries = batch["summaries"]
            batch_sources = batch["sources"]
            summaries_embeddings, summaries_tokens = self.__embed_batch(batch_summaries)
            sources_embeddings, sources_tokens = self.__embed_batch(batch_sources)

            # compare summary and source pairwise
            for summary_embeddings, source_embeddings, summary_tokens, source_tokens in zip(summaries_embeddings, sources_embeddings, summaries_tokens, sources_tokens):
                result = BERTScore.__calc_bertscore(summary_embeddings, source_embeddings)
                results.append({
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "similarities": result["similarities"],
                    "summary_tokens": summary_tokens,
                    "source_tokens": source_tokens
                })
        return results

    def __score_batch_sentences(self, summaries_sentences: List[List[str]], sources_sentences: List[List[str]]) -> List[BERTScoreResult]:
        """
        Calculate BERTScore for batches of summaries and sources, consisting of multiple sentences:
        1. summaries and sources are split into chunks / batches to save resources
        2. in every batch, every summary & source sentence is assigned an id and collected in a large batch of summary & source sentences.
           __embed_batch_grouped is used to calculate token embeddings for every sentence and collects them in one tensor per summary / source.
        3. BERTScore is calculated using the source & summary embeddings.
        Sentences longer than 512 tokens (depending on the model) are truncated.
        :param summaries_sentences: batch of summaries containing of lists of sentences
        :param sources_sentences: batch of sources containing of lists of sentences
        :return: List of {precision, recall, f1, similarities, summary_tokens, source_tokens}
        """
        assert len(summaries_sentences) == len(sources_sentences)

        bs = 32
        sum_batches = [summaries_sentences[x:x + bs] for x in range(0, len(summaries_sentences), bs)]
        src_batches = [sources_sentences[x:x + bs] for x in range(0, len(sources_sentences), bs)]

        results: List[BERTScoreResult] = []
        tqdm_args = {"desc": "Calculating bertscore..."} if self.show_tqdm else {}
        for sum_batch, src_batch in tqdm(zip(sum_batches, src_batches),  *tqdm_args):
            all_summaries_sentences = []
            ids_summaries_sentences = []
            for idx, x in enumerate(sum_batch):
                all_summaries_sentences.extend(x)
                ids_summaries_sentences.extend([idx for _ in range(len(x))])

            all_sources_sentences = []
            ids_sources_sentences = []
            for idx, x in enumerate(src_batch):
                all_sources_sentences.extend(x)
                ids_sources_sentences.extend([idx for _ in range(len(x))])

            summaries_embeddings, summaries_tokens = self.__embed_batch_grouped(all_summaries_sentences, ids_summaries_sentences)
            sources_embeddings, sources_tokens = self.__embed_batch_grouped(all_sources_sentences, ids_sources_sentences)

            # compare summary and source pairwise
            for summary_embeddings, source_embeddings, summary_tokens, source_tokens in zip(summaries_embeddings, sources_embeddings, summaries_tokens, sources_tokens):
                result = BERTScore.__calc_bertscore(summary_embeddings, source_embeddings)
                results.append({
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "similarities": result["similarities"],
                    "summary_tokens": summary_tokens,
                    "source_tokens": source_tokens
                })
        return results

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]) -> AlignScoreResult:
        """
        Calculate a similarity metric for provided texts (phrases, sentences...).
        The texts are aligned using the maximum similarities. (e.g. [1, 3]: summary 0 -> source 1, summary1 -> source 3)
        Precision, recall and f1 are computed for the similarity matrix.
        Texts longer than 512 tokens (depending on the model) are truncated.
        :param summary_texts: list of texts, phrases, sentences...
        :param source_texts: other list of texts, phrases, sentences to be compared, aligned and scored
        :return: [precision, recall, f1, summary_source_alignment, similarities]
        """
        if len(summary_texts) == 0 or len(source_texts) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "summary_source_alignment": [],
                "source_summary_alignment": [],
                "summary_source_similarities": [],
                "source_summary_similarities": []
            }

        summaries_embeddings, summaries_tokens = self.__embed_batch(summary_texts)
        sources_embeddings, sources_tokens = self.__embed_batch(source_texts)

        # calculate a bertscore similarity matrix
        # that compares every summary and source <sentence, phrase, text...>
        similarities = torch.tensor(
            [[BERTScore.__calc_bertscore(summary_embeddings, source_embeddings)["f1"]  # use f1 bertscore
              for source_embeddings in sources_embeddings]
             for summary_embeddings in summaries_embeddings])

        summary_source_alignment = similarities.argmax(dim=1).tolist()
        source_summary_alignment = similarities.argmax(dim=0).tolist()

        # compute scores
        prf1 = calc_prf1(similarities)

        return {
            "precision": prf1["precision"],
            "recall": prf1["recall"],
            "f1": prf1["f1"],
            "summary_source_alignment": summary_source_alignment,
            "source_summary_alignment": source_summary_alignment,
            "summary_source_similarities": similarities.tolist(),
            "source_summary_similarities": similarities.T.tolist(),
        }

    @staticmethod
    def __calc_bertscore(summary_embeddings: torch.Tensor, source_embeddings: Tensor) -> PRF1SimResult:
        """
        Calculates BERTScore, given the token embeddings (2D Tensors of shape [#tokens, #embedding_dims] of two texts
        :param summary_embeddings: token embeddings of text1
        :param source_embeddings: token embeddings of text2
        :return: {precision, recall, f1, similarities}
        """
        if not summary_embeddings.numel() or not source_embeddings.numel():
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "similarities": []
            }

        similarities = summary_embeddings.mm(source_embeddings.T)
        result = calc_prf1(similarities)
        return {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "similarities": similarities.tolist()
        }

    def __embed(self, text: str):
        """
        Calculates token embeddings for a single text (e.g. summary, sentence, phrases...)
        Texts longer than 512 tokens (depending on the model) are truncated.
        :param text: the text to embedd
        :return: 2D Tensor of shape [#tokens, #embedding_dims]
        """
        # tokenize text
        model_inputs = self.tokenizer(text, return_tensors="pt", truncation=True)  # this also adds the special tokens necessary for the model
        model_inputs.to(self.device)

        # apply model
        outputs = self.model(**model_inputs, output_hidden_states=True)

        # extract tokens & embeddings
        embeddings = outputs.hidden_states[self.layer].detach()  # use the embeddings of the specified layer
        embeddings = embeddings.squeeze()[1:-1]  # do not use the CLS and SEP token (beginning and end token)
        tokens = model_inputs['input_ids'].squeeze()[1:-1].tolist()

        # normalize embeddings
        embeddings = torch.stack([x / x.norm() for x in embeddings])

        return embeddings, tokens

    def __embed_batch(self, texts: List[str], group_by_text=True):
        """
        Calculate token embeddings for a batch of texts (e.g. a summary) and returns one 2D Tensor per text. [if group_by_text=True]
        Calculates token embeddings for a batch of sentences [if group_by_text=False) and returns one 2D tensor containing all token embeddings.
        Texts longer than 512 tokens (depending on the model) are truncated.
        :param texts: a list of texts or sentences
        :param group_by_text: return one tensor per text (True) or collect all embeddings in one tensor (False)
        :return: List of 2D Tensors [if group_by_text=True] or 2D Tensor [else]
        """
        # construct dataset
        dataloader = DataLoader(SimpleDataset(texts), batch_size=self.batch_size, shuffle=False)

        all_tokens = []
        all_token_embeddings = []
        for data in dataloader:
            model_inputs = self.tokenizer(data, return_tensors="pt", truncation=True, padding=True)  # this also adds the special tokens necessary for the model
            model_inputs.to(self.device)
            outputs = self.model(**model_inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[self.layer].detach()  # use the embeddings of the specified layer

            for token_embeddings, tokens, attention_mask in zip(embeddings, model_inputs['input_ids'], model_inputs['attention_mask']):
                idx = attention_mask.sum().item()
                temp = token_embeddings[1:idx - 1]  # do not use CLS, SEP and PAD token
                temp = temp / temp.norm(dim=1).unsqueeze(dim=1).expand(temp.shape)  # normalize embeddings
                if group_by_text:
                    all_token_embeddings.append(temp)  # do not use CLS, SEP and PAD token
                    all_tokens.append(tokens[1:idx - 1].tolist())  # do not use CLS, SEP and PAD token
                else:
                    all_token_embeddings.extend(temp)  # do not use CLS, SEP and PAD token
                    all_tokens.extend(tokens[1:idx - 1].tolist())  # do not use CLS, SEP and PAD token

        return all_token_embeddings if group_by_text else torch.stack(all_token_embeddings), all_tokens

    def __embed_batch_grouped(self, texts: List[str], ids: List[int]):
        """
        Calculate token embeddings for sentences and groups them by document. Every sentences is associated with a document (e.g. a summary containing of many sentences) indicated by the id.
        Texts longer than 512 tokens (depending on the model) are truncated.
        :param texts: a list of texts, e.g. summary sentences
        :param ids: a list of numbers indicating to which summary a certain sentences belongs to
        :return: a list of 2D tensors. A tensor represents token embeddings of a summary. There is one tensor per summary id in ids. (Output is grouped by id)
        """
        # construct dataset
        dataloader = DataLoader(SimpleDataset([{"text": text, "id": idx} for text, idx in zip(texts, ids)]), batch_size=self.batch_size, shuffle=False)

        all_tokens = {}
        all_token_embeddings = {}
        for batch in dataloader:
            text_batch = batch["text"]
            id_batch = batch["id"].tolist()
            model_inputs = self.tokenizer(text_batch, return_tensors="pt", truncation=True, padding=True)  # this also adds the special tokens necessary for the model
            model_inputs.to(self.device)
            outputs = self.model(**model_inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[self.layer].detach()  # use the embeddings of the specified layer

            for token_embeddings, tokens, attention_mask, text_id in zip(embeddings, model_inputs['input_ids'], model_inputs['attention_mask'], id_batch):
                idx = attention_mask.sum().item()
                temp = token_embeddings[1:idx - 1]  # do not use CLS, SEP and PAD token
                temp = temp / temp.norm(dim=1).unsqueeze(dim=1).expand(temp.shape)  # normalize embeddings
                all_token_embeddings[text_id] = [*all_token_embeddings.get(text_id, []), temp]  # do not use CLS, SEP and PAD token
                all_tokens[text_id] = [*all_tokens.get(text_id, []), tokens[1:idx - 1].tolist()]  # do not use CLS, SEP and PAD token

        # combine list of tensors to a tensor
        for key, value in all_token_embeddings.items():
            all_token_embeddings[key] = torch.cat(value)

        for key, value in all_tokens.items():
            all_tokens[key] = list(itertools.chain.from_iterable(all_tokens[key]))

        return list(all_token_embeddings.values()), list(all_tokens.values())
