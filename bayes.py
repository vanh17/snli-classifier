from data_processing import data_bayes
from collections import Counter
import sklearn
import math

class bayes:
    def __init__(self, datas: Iterable[Sequene[Text], Sequene[Text], Text]):
        """ create class object to hold important information
        vocaublary as a counter to hold vocabulary for each class that need to classify
        also create the count sume to keep track of the total words in each class
        """
        self.vocabulary = Counter()
        self.vocabulary["entailment"] = Counter()
        self.vocabulary["neutral"] = Counter()
        self.vocabulary["contradiction"] = Counter()
        self.sumE = 0
        self.sumN = 0
        self.sumC = 0
        for data in datas:
            premise, hypothesis, label = data
            for word in premise+hypothesis:
                self.vocabulary[label][word] = self.vocabulary[label].get(word, 0) + 1



