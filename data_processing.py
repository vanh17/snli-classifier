import os
import re
import string
import math
import io
import nltk
from collections import Counter
from typing import Iterator, Iterable, Tuple, Text, Union, Sequence
from nltk.stem import WordNetLemmatizer, PorterStemmer

class Data_bayes:
    def read_to_turple(self, filename: Text, is_lemmatized=False, is_stemmed=False) -> Iterable[Tuple[Tuple[Sequence[Text], Sequence[Text]], Text]]:
        """Generate (premise, hypothesis, label) turple for each sentence in the training, dev, test data.
            the structure for each line in the file name is followed:
            gold_label	premise_binary_parse	hypothesis_binary_parse	premise_parse 	hypothesis_parse
            The first line of the file is the tittle for each item, so we will ignore this line
            Also ignore the lines where the first item in the line is _, because that means all the annotators 
            could not agree on the label for such pairs.
        """
        with io.open(filename, "r") as data:
            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer()
            for line in data:
                line = line.split('\t')
                if line[0] == "gold_label" or line[0] == "_":
                    continue
                label = line[0]
                if not is_stemmed and not is_lemmatized:
                    premise = [c for c in nltk.word_tokenize(line[4])]
                    hypothesis = [c for c in nltk.word_tokenize(line[5])]
                if is_stemmed:
                    premise = [stemmer.stem(c) for c in nltk.word_tokenize(line[4])]
                    hypothesis = [stemmer.stem(c) for c in nltk.word_tokenize(line[5])]
                if is_lemmatized:
                    premise = [lemmatizer.lemmatize(c) for c in nltk.word_tokenize(line[4])]
                    hypothesis = [lemmatizer.lemmatize(c) for c in nltk.word_tokenize(line[5])]
                yield ((premise, hypothesis), label)

# class Data_mLSTM:
#     def __init__(self):


#     def data(self, filename: Text):
