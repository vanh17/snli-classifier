import pprint
import bayes
import sys
import argparse
import data_processing
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Define the argparser here to keep track of the user command
# Will have the default values for each of the args. So that the use can just need to call
# python test_bayes.py
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='data/snli_1.0/snli_1.0_train.txt')
parser.add_argument('--test_path', type=str, default='data/snli_1.0/snli_1.0_dev.txt')
parser.add_argument('--is_lemmatized', type=bool, default=False)
parser.add_argument('--is_stemmed', type=bool, default=False)

def main():
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)

    data_bayes = data_processing.Data_bayes()
    train_dataset = data_bayes.read_to_turple(args.train_path, args.is_lemmatized, args.is_stemmed)
    test_dataset = data_bayes.read_to_turple(args.test_path, args.is_lemmatized, args.is_stemmed)
    sentences, labels = zip(*test_dataset)

    # init Bayes
    clf = bayes.Bayes()

    # train Bayes
    clf.train(train_dataset)

    # predict Bayes
    predicted_indices = clf.predict(sentences)
    f1 = f1_score(labels, predicted_indices, average=None)
    f1_macro = f1_score(labels, predicted_indices, average="macro")
    accuracy = accuracy_score(labels, predicted_indices)

    print("accuracy: " + str(accuracy*100))
    print("F1_macro: " + str(f1_macro*100))
    print("F1_contradiction: " + str(f1[0]*100))
    print("F1_entailment: " + str(f1[1]*100))
    print("F1_neutral: " + str(f1[2]*100))

if __name__ == '__main__':
    main()