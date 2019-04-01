from xgboost.sklearn import XGBClassifier


class Classifier(object):
    def __init__(self, train, validation, test, conf):
        self.conf = conf
        self.train_ = train
        self.validation = validation
        self.test_ = test
        self.params = {

        }
        self.num_round = 500
        self.eval_set = list()
        self.early_stopping_rounds = 10
        self.classifier = None

    def train(self):
        self.classifier = XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=1000,
            verbosity=1,
            objective='multi:softmax',
            num_class=6,
            booster='gbtree',
            n_jobs=4,
            gamma=0.,  # 惩罚项中叶子结点个数前的参数
            min_child_weight=1,  # 叶子节点最小权重
            scale_pos_weight=1,  # 解决样本个数不平衡的问题
            seed=27
        )

        train_X = ''
        train_y = ''

        self.classifier.fit(train_X, train_y,
                            eval_metric='merror',
                            eval_set=self.eval_set,
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose=True)

        self.classifier.save_model(self.conf.model_path)

    def test(self):
        self.classifier.load_model(self.conf.model_path)


