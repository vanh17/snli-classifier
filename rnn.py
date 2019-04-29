#Importing libraries
import numpy as np
import h5py
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Merge
from keras.utils import to_categorical
from typing import Iterator, Tuple, Text, Sequence
from sklearn import preprocessing
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

class RNN:
    #Converting into pandas dataframe and filtering only text and ratings given by the users
    #Will need to handle reading data here somehow
    def __init__(self):
        self.embed_dim = 128
        self.lstm_out = 300
        self.batch_size= 64
        #tokenizer for premise to maximum word is 2500, cannot have more than this
        self.tokenizerPremise = Tokenizer(nb_words=3500, split=' ')
        #tokenizer for hypothesis to maximum word is 2500, cannot have more than this
        self.tokenizerHypo = Tokenizer(nb_words=3500, split=' ')
        #initial the model with Sequenctial class from Keras
        self.model = Sequential()
        #initial the premise model with Sequenctial class from Keras
        self.modelPremise = Sequential()
        #initial the hypothesis model with Sequenctial class from Keras
        self.modelHypo = Sequential()
        #initialize label encoder
        self.lbEncoder = preprocessing.LabelEncoder()

    def to_labels(self, labels: Sequence[Text]):
        return self.lbEncoder.transform(labels)

    def to_text(self, i):
        return self.feature_names[i] 

    def train(self, train_premise: Sequence[Text], train_hypo: Sequence[Text], train_labels: Sequence[Text]):
        #this will help us keep track of the words that is frequent
        self.tokenizerPremise.fit_on_texts(train_premise)
        #this will help us keep track of the words that is frequent
        self.tokenizerHypo.fit_on_texts(train_hypo)
        # assing the lbEncnder classes to the feature name
        # so that we can access it in the index
        self.lbEncoder.fit(train_labels)
        self.feature_names = list(self.lbEncoder.classes_)

        #print(tokenizer.word_index)  # To see the dicstionary
        #this will give us the sequence of interger represent for those index create
        #with the fit_on_texts
        doc_feat_matrixPremise = self.tokenizerPremise.texts_to_sequences(train_premise)
        doc_feat_matrixHypo = self.tokenizerHypo.texts_to_sequences(train_hypo)

        #pad_sentence will simply make sure that all the representation has the same length
        #of the longest sentence because not all the sentence have the same length
        #without this this can mess up our embedding
        doc_feat_matrixPremise = pad_sequences(doc_feat_matrixPremise)
        self.maxlenPremise = doc_feat_matrixPremise.shape[1]
        doc_feat_matrixHypo = pad_sequences(doc_feat_matrixHypo)
        self.maxlenHypo = doc_feat_matrixHypo.shape[1]

        # Buidling the LSTM network
        # Keras 2.0 does not support dropout anymore
        # Add spatial dropout instead
        self.modelPremise.add(Embedding(3500, self.embed_dim, input_length = doc_feat_matrixPremise.shape[1], dropout=0.1))
        self.modelPremise.add(LSTM(self.lstm_out, dropout_U=0.1, dropout_W=0.1))
        # structure for Hypo sentences with LSTM
        self.modelHypo.add(Embedding(3500, self.embed_dim, input_length = doc_feat_matrixHypo.shape[1], dropout=0.1))
        self.modelHypo.add(LSTM(self.lstm_out, dropout_U=0.1, dropout_W=0.1))
        #combined structures of two LSTM
        self.model.add(Merge([self.modelPremise, self.modelHypo],  mode='concat'))
        self.model.add(Dense(3, activation='softmax', input_shape=(5,)))
        self.model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

        # # do early stopping
        es = EarlyStopping(monitor='acc', mode='max', min_delta=0.0001)

        # #save the best model
        # filepath="best.hd5"
        # checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [es]

        # #start the training here
        self.model.fit([doc_feat_matrixPremise, doc_feat_matrixHypo], to_categorical(self.lbEncoder.transform(train_labels)), batch_size = self.batch_size, epochs = 5,  callbacks = callbacks_list, verbose = 1)

    def predict(self, test_premise: Sequence[Text], test_hypo: Sequence[Text]):
        # self.model = load_model("best.hd5")
        test_feat_matrixPremise = pad_sequences(self.tokenizerPremise.texts_to_sequences(test_premise), maxlen=self.maxlenPremise)
        test_feat_matrixHypo = pad_sequences(self.tokenizerHypo.texts_to_sequences(test_hypo), maxlen=self.maxlenHypo)
        return np.argmax(self.model.predict([test_feat_matrixPremise, test_feat_matrixHypo], batch_size=64, verbose=0), axis=1)
