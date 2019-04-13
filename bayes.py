from collections import Counter
import sklearn
import math

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
    			self.vocabulary[label][word] = self.vocabulary[label].get(word, 0) + 1
    			self.globVoc.add(word)
    	self.priors["entailment"] = math.log(self.classCount["entailment"] / N)
    	self.priors["neutral"] = math.log(self.classCount["neutral"] / N)
    	self.priors["contradiction"] = math.log(self.classCount["contradiction"] / N)
    	for label in ["entailment", "neutral", "contradiction"]:
        	total = sum(self.vocabulary[label].values())
        	count = len(self.vocabulary[label].keys())
        	for t in self.vocabulary[label]:
        		self.condprob[t] = self.condprob.get(t, Counter())
        		self.condprob[t][label] = math.log((self.vocabulary[label][t] + 1) / (total + count))
    
    def predict(self, texts: Iterable[Tuple[Sequence[Text], Sequence[Text]]]) -> Sequence[Text]:
    	preds = []
    	probDict = Counter()
    	for text in texts:
        	premise, hypothesis = text
        	for l in ["entailment", "neutral", "contradiction"]:
				probDict[l] = self.priors[l]
				for word in premise+hypothesis:
					if word in self.globVoc:
						probDict[l] += self.condprob[word][l]
			preds.append(max(probDict))
		return preds	
                        









