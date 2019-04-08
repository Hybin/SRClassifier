from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Classifier(object):
    def __init__(self, conf, task, train=None, test=None):
        self.conf = conf
        self.task = task
        self.train_ = train
        self.test_ = test
        self.features = ["hasWith", "hasIn", "simiBucket", "textPos", "hasOf", "hasAnd", "startEntity", "distance",
                         "hasFrom", "endEntity", "similarity", "hasThan", "hasVerb"]
        self.labels = ["relation"]
        self.num_round = 500
        self.eval_set = list()
        self.early_stopping_rounds = 20
        self.classifier = XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=1000,
            gamma=4,
            verbosity=1,
            objective='multi:softmax',
            num_class=6,
            booster='gbtree',
            n_jobs=4,
            seed=27
        )

    def train(self):
        train_X, test_X, train_y, test_y = train_test_split(self.train_[self.features],
                                                            self.train_[self.labels],
                                                            test_size=0.4,
                                                            random_state=42)

        self.eval_set = [(train_X.values, train_y.values), (test_X.values, test_y.values)]
        self.classifier.fit(train_X.values, train_y.values,
                            eval_metric='merror',
                            eval_set=self.eval_set,
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose=True)

        self.classifier.save_model(self.conf.model_path.format(self.task))
        return 'Model has been saved!'

    def test(self):
        test_set = self.test_[self.features].values
        self.classifier.load_model(self.conf.model_path.format(self.task))
        self.classifier._le = LabelEncoder().fit(['USAGE', 'TOPIC', 'MODEL-FEATURE', 'PART_WHOLE', 'RESULT', 'COMPARE'])
        pred = self.classifier.predict(test_set)
        predictions = pd.concat([self.test_[self.features], pd.DataFrame(pred, columns=["relation"])], axis=1)

        return predictions


