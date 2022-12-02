import pathlib
import sys
import spacy
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from faithfulness.interfaces.FaithfulnessInput import FaithfulnessInput
from faithfulness.interfaces.MetricInterface import MetricInterface
from faithfulness.interfaces.UsesSimilarityMetricInterface import UsesSimilarityMetricInterface
from faithfulness.similarity.F1 import F1
from faithfulness.interfaces.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.Datasets import QGDataset
from tqdm import tqdm
from faithfulness.utils.utils import load_data, save_data, ensure_dir_exists, is_PRF1Result, is_F1Result
from typing import List, Type, Dict, Tuple, Optional
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class AnswerCandidate(TypedDict):
    start: int
    end: int
    text: str


class FilteredQuestion(TypedDict):
    question: str
    expected_answer: str
    score: float
    answer: str
    answer_start: int
    answer_end: int


class AnsweredQuestion(FilteredQuestion):
    text_answer: str
    text_answer_start: int
    text_answer_end: int


class ScoredQuestion(AnsweredQuestion):
    answer_similarity_precision: float
    answer_similarity_recall: float
    answer_similarity_f1: float


class QGQAResult(TypedDict):
    precision: float
    recall: float
    f1: float
    questions: List[ScoredQuestion]


class QGQA(MetricInterface, UsesSimilarityMetricInterface):

    def __init__(self, metric: Type[SimilarityMetricInterface], save_path: pathlib.Path, metric_args=None, qamodelname='deepset/roberta-large-squad2', qgmodelname='valhalla/t5-base-qg-hl', spacymodel='en_core_web_lg', batch_mode=False):
        super(QGQA, self).__init__(metric, metric_args)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_mode = batch_mode

        self.spacymodelname = spacymodel
        self.qamodelname = qamodelname
        self.qgmodelname = qgmodelname

        self.metric: Optional[SimilarityMetricInterface] = None
        self.spacy_model = None
        self.qg_model = None
        self.qg_tokenizer = None
        self.qa_model = None

        if not batch_mode:
            self.__load_spacy_model()
            self.__load_qg_model()
            self.__load_qa_model()
            self.load_metric()

        self.save_path = save_path
        ensure_dir_exists(save_path)

        self.batch_size = 2
        self.num_questions = 5
        self.max_questions_per_text = 10

        # todo: select if questions should be generated based on summary or source

    @staticmethod
    def needs_input() -> FaithfulnessInput:
        return FaithfulnessInput.DOCUMENT

    def score(self, summary: str, source: str) -> QGQAResult:
        """
        :param summary:
        :param source:
        :return:
        """
        if self.batch_mode:
            print("ERROR: Please set batch_mode to False, to use this method")
            exit()

        answers_candidates = self.__extract_answer_candidates(summary)
        questions, question_scores, answers = self.__generate_questions(summary, answers_candidates)
        filtered_questions = self.__filter_questions(summary, questions, question_scores, answers)

        if len(filtered_questions) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "questions": []}  # we assume the summary is faithful, if there are no questions

        answered_questions = self.__answer_questions(filtered_questions, source)

        return self.__compute_score(answered_questions)

    def score_batch(self, summaries: List[str], sources: List[str]) -> List[QGQAResult]:
        """
        :param summaries:
        :param sources:
        :return:
        """
        if not self.batch_mode:
            print("ERROR: Please set batch_mode to True, to use this method")
            exit()

        answers_candidates: List[List[AnswerCandidate]] = load_data(self.save_path / "answer_candidates.pkl")
        if len(answers_candidates) == 0:
            self.__load_spacy_model()
            for summary in tqdm(summaries, desc="Extracting answer candidates..."):
                answers_candidates.append(self.__extract_answer_candidates(summary))
            save_data(answers_candidates, self.save_path / "answer_candidates.pkl")

        questions: List[Tuple[List[str], List[float], List[str]]] = load_data(self.save_path / "questions.pkl")
        if len(questions) == 0:
            self.__load_qg_model()
            for summary, acs in tqdm(zip(summaries, answers_candidates), desc="Generating questions..."):
                questions.append(self.__generate_questions(summary, acs))
            save_data(questions, self.save_path / "questions.pkl")
        all_questions, question_scores, answers = map(list, zip(*questions))

        filtered_questions: List[List[FilteredQuestion]] = load_data(self.save_path / "filtered_questions.pkl")
        if len(filtered_questions) == 0:
            self.__load_qa_model()
            for summary, qs, qs_scores, ans in tqdm(zip(summaries, all_questions, question_scores, answers), desc="Filtering questions..."):
                filtered_questions.append(self.__filter_questions(summary, qs, qs_scores, ans))
            save_data(filtered_questions, self.save_path / "filtered_questions.pkl")

        answered_questions: List[List[AnsweredQuestion]] = load_data(self.save_path / "answered_questions.pkl")
        if len(answered_questions) == 0:
            self.__load_qa_model()
            for fqs, source in tqdm(zip(filtered_questions, sources), desc="Answering questions..."):
                answered_questions.append(self.__answer_questions(fqs, source))
            save_data(answered_questions, self.save_path / "answered_questions.pkl")

        self.load_metric()
        results: List[QGQAResult] = []
        for aq in tqdm(answered_questions, desc="Computing scores..."):
            results.append(self.__compute_score(aq))

        return results

    def __extract_answer_candidates(self, text) -> List[AnswerCandidate]:
        answer_candidates: Dict[str, AnswerCandidate] = {}
        temp = self.spacy_model(text)

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
            text = ent.text
            answer_candidates[text] = {"start": start, "end": end, "text": text}

        return list(answer_candidates.values())

    def __generate_questions(self, text: str, answer_candidates: List[AnswerCandidate]) -> (List[str], List[float], List[str]):
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
        questions: List[str] = []
        question_scores: List[float] = []
        answers: List[str] = []
        for d in dataloader:
            sentence_batch = d['sentence']
            answer_batch = d['answer']

            inputs = self.qg_tokenizer.batch_encode_plus(
                sentence_batch,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                pad_to_max_length=True,
                return_tensors="pt"
            )

            outputs = self.qg_model.generate(
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

            questions.extend([self.qg_tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.sequences])
            question_scores.extend(outputs.sequences_scores.tolist())
            for answer in answer_batch:
                answers.extend([answer] * self.num_questions)

        return questions, question_scores, answers

    def __filter_questions(self, text: str, questions: List[str], scores: List[float], answers: List[str]) -> List[FilteredQuestion]:
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
        results: List[FilteredQuestion] = []
        bad_questions: List[FilteredQuestion] = []
        for question, answer, score in filtered:
            output = self.qa_model(question=question, context=text)

            tmp: FilteredQuestion = {
                'question': question,
                'expected_answer': answer,
                'score': score,
                'answer': output['answer'],
                'answer_start': output['start'],
                'answer_end': output['end']
            }
            # compare answer with expected answer
            if F1.f1_score(answer, output['answer'])["f1"] >= 0.9:
                results.append(tmp)
            else:
                bad_questions.append(tmp)

        # todo: maybe add bad questions back to the result object, so that we have a minimum number of questions?
        # take at maximum 'max_questions_per_text' most likely (based on score) questions to evaluate a text
        return sorted(results, key=lambda x: x['score'], reverse=True)[:self.max_questions_per_text]

    def __answer_questions(self, questions: List[FilteredQuestion], text: str) -> List[AnsweredQuestion]:
        result: List[AnsweredQuestion] = []
        for q in questions:
            output = self.qa_model(question=q['question'], context=text)
            result.append(dict(q, text_answer=output['answer'], text_answer_start=output["start"], text_answer_end=output["end"]))

        return result

    def __compute_score(self, answered_questions: List[AnsweredQuestion]) -> QGQAResult:
        expected_answers = [question['expected_answer'] for question in answered_questions]
        original_text_answers = [question['answer'] for question in answered_questions]
        other_text_answers = [question['text_answer'] for question in answered_questions]

        result: List[ScoredQuestion] = []
        # if a summary has no questions, we assume it is faithful!
        if len(answered_questions) == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            scores1 = self.metric.score_batch(original_text_answers, other_text_answers)
            scores2 = self.metric.score_batch(expected_answers, other_text_answers)

            precisions, recalls, f1s = [], [], []
            if is_PRF1Result(scores1[0]) and is_PRF1Result(scores2[0]):
                precisions = torch.tensor([[x["precision"] for x in scores1], [x["precision"] for x in scores2]]).max(dim=0)[0].tolist()
                recalls = torch.tensor([[x["recall"] for x in scores1], [x["recall"] for x in scores2]]).max(dim=0)[0].tolist()
                f1s = torch.tensor([[x["f1"] for x in scores1], [x["f1"] for x in scores2]]).max(dim=0)[0].tolist()
            elif is_F1Result(scores1[0]) and is_F1Result(scores2[0]):
                f1s = torch.tensor([[x["f1"] for x in scores1], [x["f1"] for x in scores2]]).max(dim=0)[0].tolist()
                precisions = f1s
                recalls = f1s

            # save results
            for precision, recall, f1, question in zip(precisions, recalls, f1s, answered_questions):
                result.append(dict(question,
                                   answer_similarity_precision=precision,
                                   answer_similarity_recall=recall,
                                   answer_similarity_f1=f1))

            precision = sum(precisions) / len(precisions)
            recall = sum(recalls) / len(recalls)
            f1 = sum(f1s) / len(f1s)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "questions": result,
        }

    def __load_spacy_model(self, unload_other_models=False):
        if self.spacy_model is None:
            print(f"Loading Spacy model {self.spacymodelname}...")
            self.spacy_model = spacy.load(self.spacymodelname)
        if unload_other_models:
            self.__unload_other_models("spacy")

    def __load_qg_model(self, unload_other_models=False):
        if self.qg_model is None:
            print(f"Loading QG model {self.qgmodelname}...")
            qg_tokenizer = AutoTokenizer.from_pretrained(self.qgmodelname)
            qg_model = T5ForConditionalGeneration.from_pretrained(self.qgmodelname)
            qg_model.to(self.device)
            self.qg_tokenizer = qg_tokenizer
            self.qg_model = qg_model
        if unload_other_models:
            self.__unload_other_models("qg")

    def __load_qa_model(self, unload_other_models=False):
        if self.qa_model is None:
            print(f"Loading QA Model {self.qamodelname}...")
            self.qa_model = pipeline('question-answering', model=self.qamodelname, tokenizer=self.qamodelname, device=0 if torch.cuda.is_available() else -1)
        if unload_other_models:
            self.__unload_other_models("qa")

    def load_metric(self, unload_other_models=False):
        if self.metric is None:
            self.metric = self.metric_type(*self.metric_args)
        if unload_other_models:
            self.__unload_other_models("metric")

    def __unload_other_models(self, modelname):
        models = {"spacy", "qg", "qa", "metric"}
        models.remove(modelname)
        for model in list(models):
            if model == "spacy":
                self.spacy_model = None
            elif model == "qg":
                self.qg_model = None
            elif model == "qa":
                self.qa_model = None
            elif model == "metric":
                self.metric = None
