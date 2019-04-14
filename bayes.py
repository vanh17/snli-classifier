from collections import Counter
from typing import Iterator, Iterable, Tuple, Text, Union, Sequence
import sklearn
import math
import nltk
import numpy as np

possible_labels = ["contradiction", "entailment", "neutral"]
class Bayes:
    def __init__(self):
        """ create class object to hold important information
        vocaublary as a counter to hold vocabulary for each class that need to classify
        also create the count sume to keep track of the total words in each class
        """
        self.classCount = Counter()
        self.vocabulary = Counter()
        # self.globVoc = set()
        self.vocabulary["entailment"] = Counter()
        self.vocabulary["neutral"] = Counter()
        self.vocabulary["contradiction"] = Counter()
        self.priors = Counter()
        self.condprob = Counter()

    def train(self, datas: Iterable[Tuple[Tuple[Sequence[Text], Sequence[Text], Text, Sequence[Text], Sequence[Text]], Text]]):
        N = 0
        for data in datas:
            N = N + 1
            sentences, label = data
            premise, hypothesis, id_train, prePos, hypoPos = sentences
            self.classCount[label] = self.classCount.get(label, 0) + 1
            # count unigram
            # use only unigram in hypothesis because that is most important 
            # sentence to determine if the pair is entailment, neutral or contradiction
            for word in premise+hypothesis:
                self.vocabulary[label] = self.vocabulary.get(label, Counter())
                self.vocabulary[label][word] = self.vocabulary[label].get(word, 0) + 1
                # self.globVoc.add(word)
            # same as unigram but this is for bigram
            for itr in range(len(hypothesis)-1):
                t = hypothesis[itr]+"_"+hypothesis[itr+1]
                self.vocabulary[label] = self.vocabulary.get(label, Counter())
                self.vocabulary[label][t] = self.vocabulary[label].get(t, 0) + 1
                # self.globVoc.add(t)
            # cross unigram
            for pre in range(len(prePos)):
                for hypo in range(len(hypoPos)):
                    if prePos[pre][1] == hypoPos[hypo][1]:
                        t = prePos[pre][0]+"_"+hypoPos[hypo][0]
                        self.vocabulary[label] = self.vocabulary.get(label, Counter())
                        self.vocabulary[label][t] = self.vocabulary[label].get(t, 0) + 1
                        # self.globVoc.add(t)
            # cross bigram
            for pre in range(len(prePos)-1):
                for hypo in range(len(hypoPos)-1):
                    if prePos[pre+1][1] == hypoPos[hypo+1][1]:
                        t = prePos[pre][0]+"_"+prePos[pre+1][0]+"_"+hypoPos[hypo][0]+"_"+hypoPos[hypo+1][0]
                        self.vocabulary[label] = self.vocabulary.get(label, Counter())
                        self.vocabulary[label][t] = self.vocabulary[label].get(t, 0) + 1
                        # self.globVoc.add(t)
        self.priors["entailment"] = math.log(self.classCount["entailment"] / N)
        self.priors["neutral"] = math.log(self.classCount["neutral"] / N)
        self.priors["contradiction"] = math.log(self.classCount["contradiction"] / N)
        self.condprob["UNKNOWN"] = Counter()
        for label in possible_labels:
            total = sum(self.vocabulary[label].values())
            count = len(self.vocabulary[label].keys())
            for t in self.vocabulary[label]:
                self.condprob[t] = self.condprob.get(t, Counter())
                self.condprob[t][label] = math.log((self.vocabulary[label][t] + 1) / (total + count))
            self.condprob["UNKNOWN"][label] = math.log(1 / (total + count))
    
    def predict(self, texts: Iterable[Tuple[Sequence[Text], Sequence[Text], Text, Sequence[Text], Sequence[Text]]]) -> Sequence[Text]:
        preds = []
        probDict = [0] * len(possible_labels)
        for text in texts:
            premise, hypothesis, id_test, prePos, hypoPos = text
            for l in range(len(possible_labels)):
                probDict[l] = self.priors[possible_labels[l]]
                # use only unigram, bigram in hypothesis because that is most important 
                # sentence to determine if the pair is entailment, neutral or contradiction
                for word in premise+hypothesis:
                    if word in self.vocabulary[possible_labels[l]]:
                        probDict[l] += self.condprob[word][possible_labels[l]]
                # bigram
                for itr in range(len(hypothesis)-1):
                    t = hypothesis[itr]+"_"+hypothesis[itr+1]
                    if t in self.vocabulary[possible_labels[l]]:
                        probDict[l] += self.condprob[t][possible_labels[l]]
                # cross unigram
                for pre in range(len(prePos)):
                    for hypo in range(len(hypoPos)):
                        if prePos[pre][1] == hypoPos[hypo][1]:
                            t = prePos[pre][0]+"_"+hypoPos[hypo][0] 
                            if t in self.vocabulary[possible_labels[l]]:
                               probDict[l] += self.condprob[t][possible_labels[l]]
                # cross bigram
                for pre in range(len(prePos)-1):
                    for hypo in range(len(hypoPos)-1):
                        if prePos[pre+1][1] == hypoPos[hypo+1][1]:
                            t = prePos[pre][0]+"_"+prePos[pre+1][0]+"_"+hypoPos[hypo][0]+"_"+hypoPos[hypo+1][0]
                            if t in self.vocabulary[possible_labels[l]]:
                                probDict[l] += self.condprob[t][possible_labels[l]]
                            else:
                                probDict[l] += self.condprob["UNKNOWN"][possible_labels[l]]
            preds.append(possible_labels[np.argmax(probDict)])
        return preds