from featurier import Featurier
from classifier import Classifier
import pandas as pd
from tqdm import tqdm


class Pipeline(object):
    def __init__(self, config, task, mode='train'):
        self.conf = config
        self.task = task
        self.mode = mode
        self.featurier = Featurier(self.conf, self.task, self.mode)
        self.classifier = None
        self.pred = None

    def load(self):
        data = self.featurier.construct()
        origin = self.featurier.units
        df = pd.DataFrame(data=list(data.values()))
        return df, origin

    def train(self):
        train, origin = self.load()
        self.classifier = Classifier(self.conf, self.task, train, None)
        self.classifier.train()

    def test(self):
        test, origin = self.load()
        self.classifier = Classifier(self.conf, self.task, None, test)
        self.pred = self.classifier.test()

        print("Predicting...")
        predictions = list()
        for pair in tqdm(self.pred.to_dict("records"), desc="Predict"):
            unit_id = self.featurier.le.inverse_transform([pair["textPos"]])[0]
            unit = origin[unit_id]
            start_ent = unit[pair["startEntity"]]
            end_ent = unit[pair["endEntity"]]

            if pair["startEntity"] > pair["endEntity"]:
                predictions.append(pair["relation"] + "(" + end_ent[1] + "," + start_ent[1] + ",REVERSE)")
            else:
                predictions.append(pair["relation"] + "(" + start_ent[1] + "," + end_ent[1] + ")")

        with open(self.conf.predict_path.format(self.task), "w") as fp:
            for prediction in predictions:
                fp.write(prediction + "\n")

    def process(self):
        if self.mode == 'train':
            self.train()
        else:
            self.test()
