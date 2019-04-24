#Importing libraries
import numpy as np
import h5py
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
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

        # ##Buidling the LSTM network
        # # Keras 2.0 does not support dropout anymore
        # # Add spatial dropout instead
        # self.model.add(Embedding(3500, self.embed_dim,input_length = doc_feat_matrix.shape[1], dropout=0.1))
        # self.model.add(LSTM(self.lstm_out, dropout_U=0.1, dropout_W=0.1))
        # self.model.add(Dense(20,activation='softmax'))
        # self.model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

        # # do early stopping
        # es = EarlyStopping(monitor='acc', mode='max', min_delta=0.0001)

        # #save the best model
        # filepath="models/best.hd5"
        # checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        # callbacks_list = [checkpoint, es]

        # #start the training here
        # self.model.fit(doc_feat_matrix, to_categorical(self.lbEncoder.transform(train_labels)), batch_size = self.batch_size, epochs = 10,  callbacks = callbacks_list, verbose = 0)

    def predict(self, test_texts: Sequence[Text]):
        self.model = load_model("models/best.hd5")
        test_feat_matrix = pad_sequences(self.tokenizer.texts_to_sequences(test_texts), maxlen=self.maxlen)
        return np.argmax(self.model.predict(test_feat_matrix, batch_size=64, verbose=0), axis=1)