from config import Config
from pipeline import Pipeline
import argparse

if __name__ == '__main__':

    # Accept the optional arguments
    parser = argparse.ArgumentParser(description='Semantic Relations Classification.')
    parser.add_argument('--task', help='number of the sub task')
    parser.add_argument('--type', help='train or test')
    args = parser.parse_args()

    # Load the config
    config = Config()

    print('Load the spacy model...')

    pipeline = Pipeline(config, args.task, args.type)

    pipeline.process()

    print("Complete!")
