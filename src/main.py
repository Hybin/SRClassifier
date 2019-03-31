from src.config import Config
from src.featurier import Featurier

if __name__ == '__main__':
    config = Config()

    print('Load the spacy model...')
    featurier = Featurier(config, '2')

    features = featurier.construct()

    for key, val in features.items():
        print(key, val)

