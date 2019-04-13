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
        self.globVoc = set()
        self.vocabulary["entailment"] = Counter()
        self.vocabulary["neutral"] = Counter()
        self.vocabulary["contradiction"] = Counter()
        self.priors = Counter()
        self.condprob = Counter()

    def train(self, datas: Iterable[Tuple[Tuple[Sequence[Text], Sequence[Text]], Text]]):
    	N = 0
    	for data in datas:
    		N = N + 1
    		sentences, label = data
    		premise, hypothesis = sentences
    		self.classCount[label] = self.classCount.get(label, 0) + 1
    		# count unigram, bigram
    		for word in premise+hypothesis:
    			self.vocabulary[label] = self.vocabulary.get(label, Counter())
    			self.vocabulary[label][word] = self.vocabulary[label].get(word, 0) + 1
    			self.globVoc.add(word)
    	self.priors["entailment"] = math.log(self.classCount["entailment"] / N)
    	self.priors["neutral"] = math.log(self.classCount["neutral"] / N)
    	self.priors["contradiction"] = math.log(self.classCount["contradiction"] / N)
    	for label in possible_labels:
        	total = sum(self.vocabulary[label].values())
        	count = len(self.vocabulary[label].keys())
        	for t in self.vocabulary[label]:
        		self.condprob[t] = self.condprob.get(t, Counter())
        		self.condprob[t][label] = math.log((self.vocabulary[label][t] + 1) / (total + count))
    
    def predict(self, texts: Iterable[Tuple[Sequence[Text], Sequence[Text]]]) -> Sequence[Text]:
    	preds = []
    	probDict = [0] * len(possible_labels)
    	for text in texts:
    		premise, hypothesis = text
    		for l in range(len(possible_labels)):
    			probDict[l] = self.priors[possible_labels[l]]
    			for word in premise+hypothesis:
    				if word in self.globVoc:
    					probDict[l] += self.condprob[word][possible_labels[l]]
    		preds.append(possible_labels[np.argmax(probDict)])
    		if hypothesis == nltk.word_tokenize("A woman in a blue tank top holding a car."):
    			print("A woman in a blue tank top holding a car. " + possible_labels[np.argmax(probDict)])
    		if hypothesis == nltk.word_tokenize("Two girls playing hopscotch in an open court."):
    			print("Two girls playing hopscotch in an open court. " + possible_labels[np.argmax(probDict)])
    	return preds	
                        









