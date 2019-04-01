from src.config import Config
from src.featurier import Featurier
import pandas as pd

if __name__ == '__main__':
    config = Config()

    print('Load the spacy model...')
    featurier = Featurier(config, '2', 'test')

    features = featurier.construct()

    df = pd.DataFrame(data=list(features.values()))

    print(df.to_string())
