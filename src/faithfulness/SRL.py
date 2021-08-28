from typing import List
import spacy
import torch
from allennlp.predictors.predictor import Predictor
from faithfulness.similarity.SimilarityMetricInterface import SimilarityMetricInterface


class SRL:

    def __init__(self, metric: SimilarityMetricInterface, srl_model_path="models/structured-prediction-srl-bert.2020.12.15.tar.gz", spacymodel='en_core_web_lg'):
        print(f'Loading Spacy model {spacymodel}...')
        self.nlp = spacy.load(spacymodel)
        self.metric = metric

        print(f"Loading semantic role labeling model ...")
        device = 0 if torch.cuda.is_available() else -1
        self.model = Predictor.from_path(srl_model_path, cuda_device=device)

    def eval(self, summary_text: str, source_text: str):
        summary_sentences = self.__split_sentences(summary_text)
        source_sentences = self.__split_sentences(source_text)

        summary_phrases = self.__get_phrases(summary_sentences)
        source_phrases = self.__get_phrases(source_sentences)

        # Find common and missing tags
        common_tags = set.intersection(set(summary_phrases.keys()), set(source_phrases.keys()))
        missing_tags = set(summary_phrases).difference(set(source_phrases))

        # todo: maybe also return precision, recall and alignment
        result = [self.metric.align_and_score([argument['phrase'] for argument in summary_phrases[tag]],
                                              [argument['phrase'] for argument in source_phrases[tag]])[2]  # we use f1[0] (we also get precision, recall, alignment and similarities as result)
                  for tag in common_tags]
        result.extend([0.0 for _ in missing_tags])

        if len(result) > 0:
            return sum(result) / len(result)
        else:
            return 0.0

    def __split_sentences(self, text: str) -> List[str]:
        return [x.text for x in self.nlp(text).sents]

    def __get_phrases(self, sentences):
        # Predict frames
        sentences = [{"sentence": sentence} for sentence in sentences]
        predictions = self.model.predict_batch_json(sentences)

        # Parse frames
        all_frames = []
        for pred in predictions:
            words = pred['words']
            for x in pred['verbs']:
                tags = x['tags']
                frame = self.__parse_frame(words, tags)
                frame = self.__merge_arguments(frame)
                all_frames.append(frame)

        # filter frames: we are only interested in frames that have more than one argument
        all_frames = [frame for frame in all_frames if len(frame) > 1]

        # converts phrases to a dict indexed by tag
        phrases = self.__order_phrases_by_tag(all_frames)

        # filters and groups phrases by tag
        phrases = self.__filter_and_group_phrases_by_tag(phrases)

        return phrases

    @staticmethod
    def __parse_frame(words, tags):
        assert len(words) == len(tags)

        frame = []
        current = None
        for idx, (word, tag) in enumerate(zip(words, tags)):
            last_word = idx == len(words) - 1
            splitted = tag.split('-')

            if len(splitted) > 1:
                category = "-".join(splitted[1:])
            else:
                category = splitted[0]

            if current is not None and current['tag'] == category:
                current['words'].append(word)
            elif current is not None and current['tag'] != category:
                current['phrase'] = " ".join(current['words'])
                if not last_word:
                    current['phrase'] = current['phrase'] + " "

                frame.append(current)
                current = {
                    'tag': category,
                    'words': [word]
                }
            else:  # current is None
                current = {
                    'tag': category,
                    'words': [word]
                }

        current['phrase'] = " ".join(current['words'])
        frame.append(current)

        return frame

    @staticmethod
    def __multi_delete(list_, idxs):
        indexes = sorted(idxs, reverse=True)
        for index in indexes:
            del list_[index]
        return list_

    @staticmethod
    def __merge_arguments(frame):
        delete_list = []
        for idx, argument in enumerate(frame):
            if argument['tag'].startswith("C"):
                splitted = argument['tag'].split('-')
                tag = "-".join(splitted[1:])

                merge_with = None
                for y in frame:
                    if y['tag'] == tag:
                        merge_with = y
                        break

                if merge_with is None:
                    argument['tag'] = tag
                else:
                    merge_with['words'].extend(argument['words'])
                    merge_with['phrase'] = " ".join(merge_with['words'])
                    merge_with['phrase'] = merge_with['phrase'] + " "
                    delete_list.append(idx)

        return SRL.__multi_delete(frame, delete_list)

    @staticmethod
    def __order_phrases_by_tag(frames):
        result = {}
        for f in frames:
            for frame in f:
                if frame['tag'] not in result.keys():
                    result[frame['tag']] = []
                result[frame['tag']].append(frame)
        return result

    @staticmethod
    def __filter_and_group_phrases_by_tag(phrases):
        attributes = []
        negations = []
        directions = []
        reasons = []
        locations = []
        subjects = []
        verbs = []
        objects = []
        hows = []
        whens = []

        for tag in phrases.keys():
            if tag in ['ARG2', 'ARG3', 'ARG4', 'ARGM-PRD']:
                attributes.extend(phrases[tag])
            elif tag in ['ARGM-NEG']:
                negations.extend(phrases[tag])
            elif tag in ['ARGM-DIR']:
                directions.extend(phrases[tag])
            elif tag in ['ARGM-CAU', 'ARGM-PRP', 'ARGM-PNC', 'ARGM-GOL']:
                reasons.extend(phrases[tag])
            elif tag in ['ARGM-LOC']:
                locations.extend(phrases[tag])
            elif tag in ['ARG0', 'ARGM-COM', 'ARGA']:
                subjects.extend(phrases[tag])
            elif tag in ['V']:
                verbs.extend(phrases[tag])
            elif tag in ['ARG1']:
                objects.extend(phrases[tag])
            elif tag in ['ARGM-MNR', 'ARGM-EXT', 'ARGM-ADV', 'ARGM-ADJ']:
                hows.extend(phrases[tag])
            elif tag in ['ARGM-TMP']:
                whens.extend(phrases[tag])

        result = {}
        if len(attributes) > 0:
            result['attributes'] = attributes
        if len(negations) > 0:
            result['negations'] = negations
        if len(directions) > 0:
            result['directions'] = directions
        if len(reasons) > 0:
            result['reasons'] = reasons
        if len(locations) > 0:
            result['locations'] = locations
        if len(subjects) > 0:
            result['subjects'] = subjects
        if len(verbs) > 0:
            result['verbs'] = verbs
        if len(objects) > 0:
            result['objects'] = objects
        if len(hows) > 0:
            result['hows'] = hows
        if len(whens) > 0:
            result['whens'] = whens
        return result
