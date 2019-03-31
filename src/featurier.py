from src.utils import *
from xml.etree import ElementTree
from tqdm import tqdm
from nltk.corpus import verbnet
from sklearn.preprocessing import LabelEncoder
import numpy as np
import spacy
import re


class Featurier(object):
    def __init__(self, config, task):
        self.conf = config
        self.task = task
        self.nlp = spacy.load("en_core_web_md")
        self.text_ids = list()
        self.le = LabelEncoder()
        self.abstract = ""

    # 载入 XML 文件
    @property
    def load(self):
        tree = ElementTree.parse(self.conf.train_set_path.format(self.task))
        root = tree.getroot()
        texts = root.findall("text")

        units = list()
        for text in tqdm(texts, desc="Load the train data"):
            # Store the id of text
            self.text_ids.append(text.get("id"))

            unit = list()
            # Process the title
            title = text.find("title")
            unit.append(self._parse(title))

            # Process the abstract
            abstract = text.find("abstract")
            abs_cont = self._parse(abstract)
            sentences = [sentence.text for sentence in self.nlp(self.abstract).sents]

            for sentence in sentences:
                bag = list()

                sentence = sentence.strip()
                for (word, flag) in abs_cont:
                    if len(sentence) > 0 and re.search(re.escape(word), sentence) is not None:
                        bag.append((word, flag))
                        if word == " ":
                            sentence = sentence[0:].strip()
                        else:
                            sentence = sentence[len(word):].strip()

                abs_cont = abs_cont[len(bag):]
                unit.append(bag)
            units.append(unit)

        # Encode the text_ids
        self.le.fit(self.text_ids)

        return units

    # 解析 XML 文件
    def _parse(self, element):
        contents = list()
        abstract = ""
        if element is not None:
            entities = list(map(get_entity, element.findall("entity")))
            for phrase in element.itertext():
                abstract += phrase
                entity = [entity for entity in entities if phrase in entity]
                if len(entity) > 0:
                    contents.append(entity[0])
                else:
                    tokens = self.nlp(phrase.strip())
                    contents += [(token.text, "text") for token in tokens]
        self.abstract = abstract
        return contents

    # 构建词汇特征
    @staticmethod
    def _get_pairs(sentence):
        """获取实体对"""
        entities = list()
        for (word, flag) in sentence:
            if flag != 'text':
                entities.append((word, flag))

        return make_pair(entities)

    @staticmethod
    def _get_distance(sentence, pair):
        """获取实体之间的距离"""
        return sentence.index(pair[1]) - sentence.index(pair[0])

    @staticmethod
    def _get_indicator(sentence, pair, indicator):
        """获取指示词特征"""
        segment = sentence[sentence.index(pair[0]) + 1:sentence.index(pair[1])]
        for (word, flag) in segment:
            if indicator == word:
                return 1

        return 0

    def _get_levin_class(self, sentence):
        head = [sent.root.text for sent in self.nlp(merge_pairs(sentence)).sents]
        classes = verbnet.classids(head[0])

        if len(classes) == 0:
            return -1
        else:
            return int(classes[0].split("-")[1].split('.')[0])

    # 构建实体特征
    def _get_text_pos(self, pair):
        return self.le.transform([pair[0][1].split(".")[0]])[0]

    @staticmethod
    def _get_sent_pos(text, sentence):
        return text.index(sentence)

    @staticmethod
    def _get_start_ent(sentence, pair):
        return sentence.index(pair[0])

    @staticmethod
    def _get_end_ent(sentence, pair):
        return sentence.index(pair[1])

    def _get_similarity(self, pair):
        return self.nlp(pair[0][0]).similarity(self.nlp(pair[1][0]))

    @staticmethod
    def _get_simi_bucket(similarity):
        if 0 <= similarity <= 0.25:
            return 0
        elif 0.25 < similarity <= 0.5:
            return 1
        elif 0.5 < similarity <= 0.75:
            return 2
        else:
            return 3

    def construct(self):
        print('Spacy model loaded...', 'Begin to load the train data...')
        units = self.load
        print('Train data loaded...', 'Begin to extract the features...')

        features = {}
        for unit in tqdm(units, desc="Featurier"):
            for sentence in unit:
                feature = {}
                pairs = self._get_pairs(sentence)

                for pair in pairs:
                    feature[pair[0][1] + "-" + pair[1][1]] = {}
                    feature[pair[0][1] + "-" + pair[1][1]]["distance"] = self._get_distance(sentence, pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["hasIn"] = self._get_indicator(sentence, pair, "in")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasOf"] = self._get_indicator(sentence, pair, "of")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasWith"] = self._get_indicator(sentence, pair, "with")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasThan"] = self._get_indicator(sentence, pair, "than")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasAnd"] = self._get_indicator(sentence, pair, "and")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasWith"] = self._get_indicator(sentence, pair, "with")
                    feature[pair[0][1] + "-" + pair[1][1]]["levinClass"] = self._get_levin_class(sentence)
                    feature[pair[0][1] + "-" + pair[1][1]]["textPos"] = self._get_text_pos(pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["sentPos"] = self._get_sent_pos(unit, sentence)
                    feature[pair[0][1] + "-" + pair[1][1]]["startEntity"] = self._get_start_ent(sentence, pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["endEntity"] = self._get_end_ent(sentence, pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["similarity"] = self._get_similarity(pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["simiBucket"] = self._get_simi_bucket(feature[pair[0][1] + "-" + pair[1][1]]["similarity"])

                features.update(feature)
        return features
