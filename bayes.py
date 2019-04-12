from data_processing import data_bayes
from collections import Counter
import sklearn
import math

class bayes:
    def __init__(self):
        """ create class object to hold important information
        vocaublary as a counter to hold vocabulary for each class that need to classify
        also create the count sume to keep track of the total words in each class
        """
        self.classCount = Counter()
        self.vocabulary = Counter()
        self.vocabulary["entailment"] = Counter()
        self.vocabulary["neutral"] = Counter()
        self.vocabulary["contradiction"] = Counter()
        self.priorE = 0.0
        self.priorN = 0.0
        self.priorC = 0.0
        self.condprob = Counter()

    def train(self, datas: Iterable[Sequene[Text], Sequene[Text], Text]):
    	N = 0
    	for data in datas:
    		N = N + 1
            premise, hypothesis, label = data
            self.classCount[label] = self.classCount.get(label, 0) + 1
            # count unigram, bigram
            for word in premise+hypothesis:
                self.vocabulary[label][word] = self.vocabulary[label].get(word, 0) + 1
        self.priorE = math.log(self.classCount["entailment"] / N)
        self.priorN = math.log(self.classCount["neutral"] / N)
        self.priorC = math.log(self.classCount["contradiction"] / N)
        for label in ["entailment", "neutral", "contradiction"]:
            total = sum(self.vocabulary[label].values())
            count = len(self.vocabulary[label].keys())
            for t in self.vocabulary[label]:
            	self.condprob[t] = self.condprob.get(t, Counter())
                self.condprob[t][label] = math.log((self.vocabulary[label][t] + 1) / (total + count))
    def predict(self, texts:Iterable[Sequene[Text]]):
               






