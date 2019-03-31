# SRClassifier
**Semantic Relation Classifier / 语义关系分类器**

### Preparation
With the `requirements.txt`, it is necessary to install python packages before 
running the program. You could follow the tip of IDE (like Pycharm) to install
them or open your terminal and input the commands:
```bash
$ cd /path/to/SRClassifier/
$ pip3 install -r requirements.txt
```
At the same time, to pre-process the train file or test files, some resource are
required, like `verbnet` inside the nltk and `en_core_web_sm` inside the spacy.
So, open your terminal again:
```bash
$ # for en_core_web_sm
$ python3 -m spacy download en_core_web_md
$ # for verbnet
$ python3
>>> import nltk
>>> nltk.download('verbnet')
>>> exit()
```
