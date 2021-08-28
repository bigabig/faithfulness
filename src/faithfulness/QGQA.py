import spacy
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from faithfulness.similarity.F1 import F1
from faithfulness.similarity.SimilarityMetricInterface import SimilarityMetricInterface
from faithfulness.utils.Datasets import QGDataset


class QGQA:

    def __init__(self, metric: SimilarityMetricInterface = F1(), qamodelname='deepset/roberta-large-squad2', qgmodelname='valhalla/t5-base-qg-hl', spacymodel='en_core_web_lg'):
        self.metric = metric

        print(f"Loading Spacy model {spacymodel}...")
        self.nlp = spacy.load(spacymodel)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print(f"Loading QG model {qgmodelname}...")
        qg_tokenizer = AutoTokenizer.from_pretrained(qgmodelname)
        qg_model = T5ForConditionalGeneration.from_pretrained(qgmodelname)
        qg_model.to(self.device)
        self.qg_tokenizer = qg_tokenizer
        self.qg_model = qg_model

        print(f"Loading QA Model {qamodelname}...")
        self.qa = pipeline('question-answering', model=qamodelname, tokenizer=qamodelname, device=0 if torch.cuda.is_available() else -1)

        self.batch_size = 2
        self.num_questions = 5
        self.max_questions_per_text = 10

        # todo: multiple answer similarities?
        # todo: select if questions should be generated based on summary or source

    def __extract_answer_candidates(self, text):
        answer_candidates = {}
        temp = self.nlp(text)

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

    def __generate_questions(self, text, answer_candidates):
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

    def __filter_questions(self, text, questions, scores, answers):
        filtered = {}
        for question, score, answer in zip(questions, scores, answers):
            # filter less than 3 tokens
            # filter duplicates, make sure that questions contain '?' and always take the question with highest score
            if len(question.split()) > 2 and "?" in question and (question not in filtered.keys() or filtered[question][2] < score):
                filtered[question] = (question, answer, score)
        filtered = list(filtered.values())

        # answer the question with a qa model given the text that was used to generate the questions
        # filter out the question, if the predicted answer does not match the expected answer
        # (idea: if the qa model is not able to predict the correct answer on the original text, it also won't predict the correct answer given another text e.g. the source document. Thus, it is a bad question!)
        results = []
        bad_questions = []
        for question, answer, score in filtered:
            qa_input = {'question': question,
                        'context': text}
            result = self.qa(qa_input)

            tmp = {
                'question': question,
                'expected_answer': answer,
                'score': score,
                'answer': result['answer'],
                'answer_start': result['start'],
                'answer_end': result['end']
            }

            # compare answer with expected answer
            if F1.f1_score(answer, result['answer']) >= 0.9:
                results.append(tmp)
            else:
                bad_questions.append(tmp)

        # todo: maybe add bad questions back to the result object, so that we have a minimum number of questions?

        # take at maximum 'max_questions_per_text' most likely (based on score) questions to evaluate a text
        return sorted(results, key=lambda x: x['score'], reverse=True)[:self.max_questions_per_text]

    def __answer_questions(self, questions, text):
        for q in questions:
            question = q['question']

            # predict answer
            qa_input = {'question': question,
                        'context': text}

            result = self.qa(qa_input)

            # save results
            q['text_answer'] = result['answer']
            q['text_answer_start'] = result['start']
            q['text_answer_end'] = result['end']

        return questions

    def __compute_score(self, answered_questions):
        expected_answers = [question['expected_answer'] for question in answered_questions]
        original_text_answers = [question['answer'] for question in answered_questions]
        other_text_answers = [question['text_answer'] for question in answered_questions]

        scores1 = self.metric.score_batch(original_text_answers, other_text_answers)
        scores2 = self.metric.score_batch(expected_answers, other_text_answers)
        scores = torch.tensor([scores1, scores2]).max(dim=0)[0]

        # save results
        for score, question in zip(scores.tolist(), answered_questions):
            question['answer_similarity'] = score

        return scores.mean().item(), answered_questions

    def score(self, summary, source):
        answers_candidates = self.__extract_answer_candidates(summary)
        questions, question_scores, answers = self.__generate_questions(summary, answers_candidates)
        filtered_questions = self.__filter_questions(summary, questions, question_scores, answers)

        if len(filtered_questions) == 0:
            return 1.0  # we assume the summary is faithful, if there are no questions

        answered_questions = self.__answer_questions(filtered_questions, source)

        return self.__compute_score(answered_questions)
