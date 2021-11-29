import pickle
from typing import List
import spacy
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.similarity.F1 import F1
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.Datasets import QGDataset
from tqdm import tqdm

from faithfulness.utils.utils import load_data, save_data


class QGQA(MetricInterface):

    def __init__(self, metric: SimilarityMetricInterface = F1(), qamodelname='deepset/roberta-large-squad2', qgmodelname='valhalla/t5-base-qg-hl', spacymodel='en_core_web_lg', batch_mode=False, save_path=""):
        self.metric = metric
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_mode = batch_mode
        if not batch_mode:
            print(f"Loading Spacy model {spacymodel}...")
            self.spacy_model = spacy.load(spacymodel)

            print(f"Loading QG model {qgmodelname}...")
            qg_tokenizer = AutoTokenizer.from_pretrained(qgmodelname)
            qg_model = T5ForConditionalGeneration.from_pretrained(qgmodelname)
            qg_model.to(self.device)
            self.qg_tokenizer = qg_tokenizer
            self.qg_model = qg_model

            print(f"Loading QA Model {qamodelname}...")
            self.qa_model = pipeline('question-answering', model=qamodelname, tokenizer=qamodelname,
                               device=0 if torch.cuda.is_available() else -1)
        else:
            self.spacymodelname = spacymodel
            self.qamodelname = qamodelname
            self.qgmodelname = qgmodelname

        self.save_path = save_path
        self.batch_size = 2
        self.num_questions = 5
        self.max_questions_per_text = 10

        # todo: multiple answer similarities?
        # todo: select if questions should be generated based on summary or source

    def score(self, summary: str, source: str, additional_output: bool = True):
        """
        :param summary:
        :param source:
        :param additional_output: has no effect
        :return:
        """
        if self.batch_mode:
            print("ERROR: Please set batch_mode to False, to use this method")
            exit()

        answers_candidates = self.__extract_answer_candidates(summary, self.spacy_model)
        questions, question_scores, answers = self.__generate_questions(summary, answers_candidates, self.qg_model, self.qg_tokenizer)
        filtered_questions = self.__filter_questions(summary, questions, question_scores, answers, self.qa_model)

        if len(filtered_questions) == 0:
            return 1.0  # we assume the summary is faithful, if there are no questions

        answered_questions = self.__answer_questions(filtered_questions, source, self.qa_model)

        return self.__compute_score(answered_questions)

    def score_batch(self, summaries: List[str], sources: List[str], additional_output: bool):
        """
        :param summaries:
        :param sources:
        :param additional_output: has no effect
        :return:
        """
        if not self.batch_mode:
            print("ERROR: Please set batch_mode to True, to use this method")
            exit()

        filtered_questions = load_data(self.save_path + "_filtered_questions.pkl")
        if len(filtered_questions) == 0:
            print(f"Loading Spacy model {self.spacymodelname}...")
            model = spacy.load(self.spacymodelname)

            answers_candidates = []
            for summary in tqdm(summaries, desc="Extracting answer candidates..."):
                answers_candidates.append(self.__extract_answer_candidates(summary, model))

            print(f"Loading QG model {self.qgmodelname}...")
            tokenizer = AutoTokenizer.from_pretrained(self.qgmodelname)
            model = T5ForConditionalGeneration.from_pretrained(self.qgmodelname)
            model.to(self.device)

            questions, question_scores, answers = [], [], []
            for summary, acs in tqdm(zip(summaries, answers_candidates), desc="Generating questions..."):
                q, qs, a = self.__generate_questions(summary, acs, model, tokenizer)
                questions.append(q)
                question_scores.append(qs)
                answers.append(a)

            print(f"Loading QA Model {self.qamodelname}...")
            model = pipeline('question-answering', model=self.qamodelname, tokenizer=self.qamodelname,
                             device=0 if torch.cuda.is_available() else -1)

            filtered_questions = []
            for summary, qs, qs_scores, ans in tqdm(zip(summaries, questions, question_scores, answers),
                                                    desc="Filtering questions..."):
                filtered_questions.append(self.__filter_questions(summary, qs, qs_scores, ans, model))

            save_data(filtered_questions, self.save_path + "_filtered_questions.pkl")

        print("Continuing from checkpoint...")
        print(f"Loading QA Model {self.qamodelname}...")
        model = pipeline('question-answering', model=self.qamodelname, tokenizer=self.qamodelname, device=0 if torch.cuda.is_available() else -1)

        answered_questions = load_data(self.save_path + "_answered_questions.pkl")
        if len(answered_questions) == 0:
            answered_questions = []
            for fqs, source in tqdm(zip(filtered_questions, sources), desc="Answering questions..."):
                answered_questions.append(self.__answer_questions(fqs, source, model))
            save_data(answered_questions, self.save_path + "_answered_questions.pkl")
        else:
            print("Continuing from checkpoint...")

        results = {}
        for aq in tqdm(answered_questions, desc="Computing scores..."):
            r = self.__compute_score(aq)
            for key, value in r.items():
                results[key] = [*results.get(key, []), value]

        return results

    @staticmethod
    def __extract_answer_candidates(text, model):
        answer_candidates = {}
        temp = model(text)

        # noun phrases
        for phrase in temp.noun_chunks:
            start = phrase.start_char
            end = phrase.end_char
            text = phrase.text
            answer_candidates[text] = {"start": start, "end": end, "text": text}

        # named entities
        for ent in temp.ents:
            start = ent.start_char
            end = ent.end_char
            # label = ent.label_  # actually not  important...
            text = ent.text
            answer_candidates[text] = {"start": start, "end": end, "text": text}

        return list(answer_candidates.values())

    def __generate_questions(self, text, answer_candidates, model, tokenizer):
        # generate input pairs for the model
        inputs = set()  # use set to prevent duplicates!
        for answer in answer_candidates:
            ans_start = answer['start']
            ans_end = answer['end']
            inputs.add((f"{text[:ans_start]}<hl>{text[ans_start:ans_end]}</hl>{text[ans_end:]}</s>", text[ans_start:ans_end]))

        # construct dataset
        dataset = QGDataset(sentences=[tup[0] for tup in inputs], answers=[tup[1] for tup in inputs])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # predict questions
        questions = []
        question_scores = []
        answers = []
        for d in dataloader:
            sentence_batch = d['sentence']
            answer_batch = d['answer']

            inputs = tokenizer.batch_encode_plus(
                sentence_batch,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                pad_to_max_length=True,
                return_tensors="pt"
            )

            outputs = model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                max_length=60,
                num_beams=10,
                num_return_sequences=self.num_questions,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                output_scores=True,
                return_dict_in_generate=True,
            )

            questions.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.sequences])
            question_scores.extend(outputs.sequences_scores.tolist())
            for answer in answer_batch:
                answers.extend([answer] * self.num_questions)

        return questions, question_scores, answers

    def __filter_questions(self, text, questions, scores, answers, model):
        filtered = {}
        for question, score, answer in zip(questions, scores, answers):
            # filter less than 3 tokens
            # filter duplicates, make sure that questions contain '?' and always take the question with highest score
            if len(question.split()) > 2 and "?" in question and (question not in filtered.keys() or filtered[question][2] < score):
                filtered[question] = (question, answer, score)
        filtered = list(filtered.values())

        # answer the question with a qa model given the text that was used to generate the questions
        # # filter out the question, if the predicted answer does not match the expected answer
        # # (idea: if the qa model is not able to predict the correct answer on the original text, it also won't predict the correct answer given another text e.g. the source document. Thus, it is a bad question!)
        results = []
        bad_questions = []
        for question, answer, score in filtered:
            output = model(question=question, context=text)

            tmp = {
                'question': question,
                'expected_answer': answer,
                'score': score,
                'answer': output['answer'],
                'answer_start': output['start'],
                'answer_end': output['end']
            }
            # compare answer with expected answer
            if F1.f1_score(answer, output['answer']) >= 0.9:
                results.append(tmp)
            else:
                bad_questions.append(tmp)

        # todo: maybe add bad questions back to the result object, so that we have a minimum number of questions?
        # take at maximum 'max_questions_per_text' most likely (based on score) questions to evaluate a text
        return sorted(results, key=lambda x: x['score'], reverse=True)[:self.max_questions_per_text]

    @staticmethod
    def __answer_questions(questions, text, model):
        for q in questions:
            output = model(question=q['question'], context=text)
            q['text_answer'] = output['answer']
            q['text_answer_start'] = output['start']
            q['text_answer_end'] = output['end']

        return questions

    def __compute_score(self, answered_questions):
        expected_answers = [question['expected_answer'] for question in answered_questions]
        original_text_answers = [question['answer'] for question in answered_questions]
        other_text_answers = [question['text_answer'] for question in answered_questions]

        # if a summary has no questions, we assume it is faithful!
        if len(answered_questions) == 0:
            score = 1.0
        else:
            variant = self.metric.get_variant()
            scores1 = self.metric.score_batch(original_text_answers, other_text_answers, False)[variant.value]
            scores2 = self.metric.score_batch(expected_answers, other_text_answers, False)[variant.value]
            scores = torch.tensor([scores1, scores2]).max(dim=0)[0]

            # if a summary has no questions, we assume it is faithful!
            if len(scores) == 0:
                print("LOL")
                print(len(expected_answers), len(original_text_answers), len(other_text_answers))

            # save results
            for score, question in zip(scores.tolist(), answered_questions):
                question['answer_similarity'] = score

            score = scores.mean().item()

        return {
            "score": score,
            "questions": answered_questions,
        }
