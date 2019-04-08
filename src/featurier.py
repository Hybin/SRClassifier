from utils import *
from xml.etree import ElementTree
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
import warnings

warnings.filterwarnings("ignore")


class Featurier(object):
    def __init__(self, config, task, mode='train'):
        self.conf = config
        self.task = task
        self.mode = mode
        self.nlp = spacy.load("en_core_web_md")
        self.text_ids = list()
        self.le = LabelEncoder()
        self.abstract = ""
        self.relations = {"USAGE": 5, "TOPIC": 4, "MODEL-FEATURE": 1, "PART_WHOLE": 2, "RESULT": 3, "COMPARE": 0}
        self.container = dict()
        self.units = dict()
        self.segments = dict()

    # 载入数据
    def load(self):
        if self.mode == 'train':
            data_path = self.conf.train_set_path
            label_path = self.conf.train_label_path
        else:
            data_path = self.conf.test_set_path
            label_path = self.conf.test_label_path

        """载入 XML 数据"""
        tree = ElementTree.parse(data_path.format(self.task))
        root = tree.getroot()
        texts = root.findall("text")

        units = list()
        for text in tqdm(texts, desc="Load the train data"):
            # Store the id of text
            self.text_ids.append(text.get("id"))

            unit = list()
            # Process the title
            title = text.find("title")
            title_cont = self._parse(title, text.get("id"))
            unit.append(title_cont)

            # Process the abstract
            abstract = text.find("abstract")
            abs_cont = self._parse(abstract, text.get("id"))
            self.units[text.get("id")] = title_cont + abs_cont
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
            units.append({text.get("id"): unit})

        # Encode the text_ids
        self.le.fit(self.text_ids)

        """载入关系数据"""
        with open(label_path.format(self.task), 'r') as fp:
            data = fp.readlines()
            labels = [(item[0:item.index("(")], item[item.index("(") + 1:-2]) for item in data]
            relations = dict()
            for (label, pair) in labels:
                unit_id = pair.split(",")[0].split(".")[0]
                if unit_id not in relations.keys():
                    relations[unit_id] = list()

                relations[unit_id].append((label, pair))

        return units, relations

    # 解析 XML 文件
    def _parse(self, element, unit_id):
        contents = list()
        abstract = ""
        if element is not None:
            entities = get_entity(element)
            if unit_id not in self.container.keys():
                self.container[unit_id] = list()

            self.container[unit_id] += entities

            for phrase in element.itertext():
                abstract += phrase
                entity = [entity for entity in entities if phrase in entity]
                if len(entity) > 0:
                    contents.append(entity[0])
                    entities.remove(entity[0])
                else:
                    tokens = self.nlp(phrase.strip())
                    items = [(token.text, "text") for token in tokens]
                    for (item, flag) in items:
                        if re.search(re.escape(".") + "[A-Z][a-z]+", item) is not None:
                            i = items.index((item, flag))
                            items.remove(items[i])
                            items.insert(i, (item[0], 'text'))
                            items.insert(i + 1, (item[1:], flag))

                    contents += items
            contents += entities
        # Check if the punctuation is correct
        while re.search(re.escape(".") + "[A-Z][a-z]+", abstract) is not None:
            pos = re.search(re.escape(".") + "[A-Z][a-z]+", abstract).span()[0] + 1
            abstract = abstract[0:pos] + " " + abstract[pos:]
        self.abstract = abstract
        return contents

    # 构建词汇特征
    def _get_pairs(self, labels, unit_id):
        """获取关系类型（训练数据）及实体对"""
        relations = list()
        pairs = list()
        entities = self.container[unit_id]
        for (label, pair) in labels[unit_id]:
            # Get the relation
            if label != "":
                relations.append(self.relations[label])
            else:
                relations.append(-1)

            # Get the pair
            indice = pair.split(",")
            if 'REVERSE' in indice:
                ent_pair = (find_pair(indice[1], entities), find_pair(indice[0], entities))
            else:
                ent_pair = (find_pair(indice[0], entities), find_pair(indice[1], entities))

            pairs.append(ent_pair)

        return relations, pairs

    @staticmethod
    def _get_distance(unit, pair):
        """获取实体之间的距离"""
        distance = abs(unit.index(pair[1]) - unit.index(pair[0]))
        return distance

    @staticmethod
    def _get_indicator(sentence, pair, indicator):
        """获取指示词特征"""
        segment = sentence[sentence.index(pair[0]) + 1:sentence.index(pair[1])]
        for (word, flag) in segment:
            if indicator == word:
                return 1

        return 0

    def _get_verb(self, sentence, pair):
        """判断实体之间是否存在动词"""
        segment = sentence[sentence.index(pair[0]) + 1:sentence.index(pair[1])]
        for (word, flag) in segment:
            doc = self.nlp(word)
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            if len(verbs) > 0:
                return 1

        return 0

    def _build_tf_idf(self, pairs, sentence):
        for pair in pairs:
            segment = sentence[sentence.index(pair[0]) + 1:sentence.index(pair[1])]
            content = ""
            for (word, flag) in segment:
                content += word + " "
            self.segments[pair[0][1] + "-" + pair[1][1]] = content.strip()

        vectorizer = TfidfVectorizer()
        vectorizer.fit(list(self.segments.values()))

        return vectorizer

    # 构建实体特征
    def _get_text_pos(self, pair):
        return self.le.transform([pair[0][1].split(".")[0]])[0]

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
        units, labels = self.load()
        print('Train data loaded...', 'Begin to extract the features...')

        features = {}

        for unit in tqdm(units, desc="Featurier"):
            feature = {}
            unit_id = list(unit.keys())[0]
            container = self.units[unit_id]
            if unit_id in labels.keys():
                relations, pairs = self._get_pairs(labels, unit_id)
                vectorizer = self._build_tf_idf(pairs, container)

                for pair in pairs:
                    feature[pair[0][1] + "-" + pair[1][1]] = {}
                    feature[pair[0][1] + "-" + pair[1][1]]["distance"] = self._get_distance(container, pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["hasVerb"] = self._get_verb(container, pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["hasIn"] = self._get_indicator(container, pair, "in")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasTo"] = self._get_indicator(container, pair, "to")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasOn"] = self._get_indicator(container, pair, "on")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasIs"] = self._get_indicator(container, pair, "is")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasOf"] = self._get_indicator(container, pair, "of")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasWith"] = self._get_indicator(container, pair, "with")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasThan"] = self._get_indicator(container, pair, "than")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasAnd"] = self._get_indicator(container, pair, "and")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasWith"] = self._get_indicator(container, pair, "with")
                    feature[pair[0][1] + "-" + pair[1][1]]["hasFrom"] = self._get_indicator(container, pair, "from")
                    feature[pair[0][1] + "-" + pair[1][1]]["textPos"] = self._get_text_pos(pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["startEntity"] = self._get_start_ent(container, pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["endEntity"] = self._get_end_ent(container, pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["similarity"] = self._get_similarity(pair)
                    feature[pair[0][1] + "-" + pair[1][1]]["simiBucket"] = self._get_simi_bucket(
                        feature[pair[0][1] + "-" + pair[1][1]]["similarity"])
                    feature[pair[0][1] + "-" + pair[1][1]]["context"] = vectorizer.transform([self.segments[pair[0][1] + "-" + pair[1][1]]]).tocsc().toarray()[0]
                    feature[pair[0][1] + "-" + pair[1][1]]["relation"] = relations[pairs.index(pair)]

            features.update(feature)

        return features
