from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from faithfulness.similarity.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.Datasets import SimpleDataset
from faithfulness.utils.utils import calc_prf1


class BERTScore(SimilarityMetricInterface):

    def __init__(self, modelname="roberta-large-mnli", layer=8):

        print(f"Init BERTScore model {modelname}...")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(modelname)  # roberta-large-mnli
        model = AutoModel.from_pretrained(modelname)
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = device
        self.batch_size = 8

    def score(self, summary_text: str, source_text: str):
        # embed summary & source
        summary_embeddings, summary_tokens = self.__embed(summary_text)
        source_embeddings, source_tokens = self.__embed(source_text)

        # calculate bertscore
        return self.__calc_bertscore(summary_embeddings, source_embeddings)[2]  # returns f1

    @staticmethod
    def __calc_bertscore(summary_embeddings, source_embeddings):
        similarities = summary_embeddings.mm(source_embeddings.T)
        return calc_prf1(similarities)

    def __embed(self, text: str):
        # tokenize text
        model_inputs = self.tokenizer(text, return_tensors="pt",
                                      truncation=True)  # this also adds the special tokens necessary for the model
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

    def __embed_batch(self, texts: List[str]):
        # construct dataset
        dataloader = DataLoader(SimpleDataset(texts), batch_size=8, shuffle=False)

        all_tokens = []
        all_token_embeddings = []
        for data in dataloader:
            model_inputs = self.tokenizer(data, return_tensors="pt", truncation=True,
                                          padding=True)  # this also adds the special tokens necessary for the model
            model_inputs.to(self.device)
            outputs = self.model(**model_inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[self.layer].detach()  # use the embeddings of the specified layer

            for token_embeddings, tokens, attention_mask in zip(embeddings, model_inputs['input_ids'],
                                                                model_inputs['attention_mask']):
                idx = attention_mask.sum().item()
                # vielleicht shiften
                # (token_embeddings[1:idx-1] / token_embeddings[1:idx-1].norm(dim=1).unsqueeze(dim=1).expand(token_embeddings[1:idx-1].shape))[1].sum()
                temp = token_embeddings[1:idx - 1]  # do not use CLS, SEP and PAD token
                temp = temp / temp.norm(dim=1).unsqueeze(dim=1).expand(temp.shape)  # normalize embeddings
                all_token_embeddings.append(temp)  # do not use CLS, SEP and PAD token
                all_tokens.append(tokens[1:idx - 1].tolist())  # do not use CLS, SEP and PAD token

        return all_token_embeddings, all_tokens

    def score_batch(self, summaries: List[str], sources: List[str]):
        assert len(summaries) == len(sources)

        summaries_embeddings, summaries_tokens = self.__embed_batch(summaries)
        sources_embeddings, sources_tokens = self.__embed_batch(sources)

        # compare summary and source pairwise
        # returns just f1 [2]
        return [self.__calc_bertscore(summary_embeddings, source_embeddings)[2] for
                summary_embeddings, source_embeddings in zip(summaries_embeddings, sources_embeddings)]

    def align_and_score(self, summary_texts: List[str], source_texts: List[str]):
        summaries_embeddings, summaries_tokens = self.__embed_batch(summary_texts)
        sources_embeddings, sources_tokens = self.__embed_batch(source_texts)

        # calculate a bertscore similarity matrix
        # that compares every summary and source <sentence, phrase, text...>
        similarities = torch.tensor(
            [[self.__calc_bertscore(summary_embeddings, source_embeddings)[2]  # use f1 bertscore
              for source_embeddings in sources_embeddings]
             for summary_embeddings in summaries_embeddings])

        summary_source_alignment = similarities.argmax(dim=1).tolist()
        # source_summary_alignment = similarities.argmax(dim=0).tolist()
        precision, recall, f1 = calc_prf1(similarities)

        return precision, recall, f1, summary_source_alignment, similarities.tolist()
