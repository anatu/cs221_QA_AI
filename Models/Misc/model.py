from __future__ import print_function
import preprocessing as pp
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model

class AttentionModel():

    def __init__(self):
        self.EMBED_HIDDEN_SIZE = 50
        self.SENT_HIDDEN_SIZE = 100
        self.QUERY_HIDDEN_SIZE = 100
        self.BATCH_SIZE = 32
        self.EPOCHS = 5
        
        prep = pp.Preprocessor()    
        self.x, self.tx, self.xq, 
        self.txq, self.y, self.ty = prep.prepare_cmu_data("../../Data/Question_Answer_Dataset_v1.2/S08")
        self.embedding_matrix = prep.generate_embedding_matrix(prep.word_idx, r"C:\Users\Anand Natu\Desktop\glove.6B")
        self.story_maxlen = prep.story_maxlen
        self.query_maxlen = prep.query_maxlen 

    def build_model(self):
        embedding_layer = Embedding(prep.vocab_size, prep.EMBEDDING_DIM, weights = self.embedding_matrix)
        
        sentence = layers.Input(shape = (self.story_maxlen,), dtype="int32")
        encoded_sentence = embedding_layer(sentence)
        encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

        question = layers.Input(shape = (self.query_maxlen,), dtype="int32")
        encoded_question = embedding_layer(question)
        encoded_question = layers.Dropout(0.3)(encoded_question)
        encoded_question = RNN(self.EMBED_HIDDEN_SIZE)(encoded_question)
        encoded_question = layers.RepeatVector(self.story_maxlen)(encoded_question)

        merged = layers.add([encoded_sentence, encoded_question])
        merged = RNN(EMBED_HIDDEN_SIZE)(merged)
        merged = layers.Dropout(0.3)(merged)
        preds_beg = layers.Dense(story_maxlen+3, activation='softmax')(merged)

        merged = layers.add([encoded_sentence, encoded_question])
        merged = RNN(EMBED_HIDDEN_SIZE)(merged)
        merged = layers.Dropout(0.3)(merged)
        preds_end = layers.Dense(story_maxlen+3, activation='softmax')(merged)

        model = Model([sentence, question], [preds_beg, preds_end])
        model = Model([sentence, question], [preds_beg, preds_end])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train_model(self):
        




if __name__ == "__main__":
    am = AttentionModel()
