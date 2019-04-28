# snli-classifier
This is a classifier for Standford Natural Language Inference Dataset.

Before starting to the project, please make sure to run this command. This will create suitable environment to run the project.

`pip install -r requirements.txt`

To run the project, please unzip the snli_1.0.zip in folder data. To unzip it, make sure you are inside the directory data:

`cd data`

To unzip run this:

`unzip snli_1.0.zip`

When it finishes unzipping the file. Please go to the project main directory by this command.

At the project main directory, you will see there are two files called test_bayes.py and test_rnn.py. The two files are the main file to run naive Bayes approach and recurrent neural network LSTM approach respectively.

To run the naive Bayes, please use this command line:

`python test_bayes.py` 

By default, the script will train on `data/snli_1.0/snli_1.0_train.txt` and evaluate on `data/snli_1.0/snli_1.0_dev.txt`. However, you can specify the option `--train_path`, `--test_path` like this to ask the script to train on file at specific directory. Please notice that the file extension here is .txt and not json. (*)

`python test_bayes.py --train_path='your_full_path_to_trainset --test_path='your_full_path_to_testset`

You can also specify the two options `--is_lemmatized`, `--is_stemmed` for pre-processing the inputs with lemmatization or stemming. Please notice that default value is only tokenization (--is_stemmed=False) (--is_lemmatized=False) and only one of the two option can set to True at a time. To use this try

`python test_bayes.py --is_lemmatized=True`

`python test_bayes.py --is_stemmed=True`

You can also combined option in (*) with the lemmatizaion and stemming. Please make sure there is a space between each option.

`python test_bayes.py --train_path='your_full_path_to_trainset --test_path='your_full_path_to_testset --is_stemmed=True`

To run LSTM, use this command line:

`python test_rnn.py`

LSTM model can also be called with exactly the same options available to naive Bayes (`--is_lemmatized`, `--is_stemmed`, `--train_path`, `--test_path`). Please notice that LSTM takes up to a day to train with 50 epochs, so you want to do something fun during such time.