import pprint
import rnn
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
parser.add_argument('--test_path', type=str, default='data/snli_1.0/snli_1.0_test.txt')
parser.add_argument('--is_lemmatized', type=bool, default=False)
parser.add_argument('--is_stemmed', type=bool, default=False)

def main():
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)

    # intial the data_parser for RNN model
    data_rnn = data_processing.Data_RNN()
    # create the training data in the needed format
    # this will be the list, one entry for each sentence in the training data. One entry is a string
    # Also need to create one for the test data
    train_dataset = data_rnn.read_to_turple(args.train_path, args.is_lemmatized, args.is_stemmed)
    test_dataset = data_rnn.read_to_turple(args.test_path, args.is_lemmatized, args.is_stemmed)
    # create three variables to hold the list from the tuple creates by the train_data/test_data
    premisesTrain, hypothesesTrain, labelsTrain = zip(*train_dataset)
    premises, hypotheses, labels = zip(*test_dataset)

    # init RNN
    clf = rnn.RNN()

    # train RNN
    clf.train(premisesTrain, hypothesesTrain, labelsTrain)

    # predict RNN
    predicted_indices = clf.predict(premises, hypotheses)
    f1 = f1_score(clf.to_labels(labels), predicted_indices, average=None)
    f1_macro = f1_score(clf.to_labels(labels), predicted_indices, average="macro")
    accuracy = accuracy_score(clf.to_labels(labels), predicted_indices)

    print("accuracy: " + str(accuracy*100))
    print("F1_macro: " + str(f1_macro*100))
    print("F1_contradiction: " + str(f1[0]*100))
    print("F1_entailment: " + str(f1[1]*100))
    print("F1_neutral: " + str(f1[2]*100))

if __name__ == '__main__':
    main()