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
    		for word in hypothesis:
    			self.vocabulary[label] = self.vocabulary.get(label, Counter())
    			self.vocabulary[label][word] = self.vocabulary[label].get(word, 0) + 1
    			self.globVoc.add(word)
            # same as unigram but this is for bigram
            for itr in range(len(hypothesis)-1):
                self.vocabulary[label] = self.vocabulary.get(label, Counter())
                self.vocabulary[label][hypothesis[itr]+"_"+hypothesis[itr+1]] = self.vocabulary[label].get(hypothesis[itr]+"_"+hypothesis[itr+1], 0) + 1
                self.globVoc.add(hypothesis[itr]+"_"+hypothesis[itr+1])
            # cross unigram
            for pre in range(len(prePos)):
                for hypo in range(len(hypoPos)):
                    if prePos[pre][1] == hypoPos[hypo][1]:
                        self.vocabulary[label] = self.vocabulary.get(label, Counter())
                        self.vocabulary[label][prePos[pre][0]+"_"+hypoPos[hypo][0]] = self.vocabulary[label].get(prePos[pre][0]+"_"+hypoPos[hypo][0], 0) + 1
                        self.globVoc.add(prePos[pre][0]+"_"+hypoPos[hypo][0])
    	self.priors["entailment"] = math.log(self.classCount["entailment"] / N)
    	self.priors["neutral"] = math.log(self.classCount["neutral"] / N)
    	self.priors["contradiction"] = math.log(self.classCount["contradiction"] / N)
    	for label in possible_labels:
        	total = sum(self.vocabulary[label].values())
        	count = len(self.vocabulary[label].keys())
        	for t in self.vocabulary[label]:
        		self.condprob[t] = self.condprob.get(t, Counter())
        		self.condprob[t][label] = math.log((self.vocabulary[label][t] + 1) / (total + count))
    
    def predict(self, texts: Iterable[Tuple[Sequence[Text], Sequence[Text], Text, Sequence[Text], Sequence[Text]]]) -> Sequence[Text]:
    	preds = []
    	probDict = [0] * len(possible_labels)
    	for text in texts:
    		premise, hypothesis, id_test, prePos, hypoPos = text
    		for l in range(len(possible_labels)):
    			probDict[l] = self.priors[possible_labels[l]]
                # use only unigram, bigram in hypothesis because that is most important 
                # sentence to determine if the pair is entailment, neutral or contradiction
    			for word in hypothesis:
    				if word in self.globVoc:
    					probDict[l] += self.condprob[word][possible_labels[l]]
                # bigram
                for itr in range(len(hypothesis)-1):
                    if hypothesis[itr]+"_"+hypothesis[itr+1] in self.globVoc:
                        probDict[l] += self.condprob[hypothesis[itr]+"_"+hypothesis[itr+1]][possible_labels[l]]
                # cross unigram
                for pre in range(len(prePos)):
                    for hypo in range(len(hypoPos)):
                        if prePos[pre][1] == hypoPos[hypo][1]:
                            if prePos[pre][0]+"_"+hypoPos[hypo][0] in self.globVoc:
                               probDict[l] += self.condprob[prePos[pre][0]+"_"+hypoPos[hypo][0]][possible_labels[l]]    
    		preds.append(possible_labels[np.argmax(probDict)])
    	return preds	
                        









