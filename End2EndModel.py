# -*- encoding:utf-8 -*-


__author__ = 'Suncong Zheng'

import pickle
import os.path
import numpy as np
from PrecessEEdata import get_data_e2e, make_idx_data_index_EE_LSTM2, make_idx_data_index_EE_LSTM, \
    make_idx_data_index_EE_LSTM3
from Evaluate import evaluavtion_triple, evaluavtion_rel, predict_rel
# from keras.models import Sequential
# from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.layers.core import Dropout, Activation

from keras.layers.merge import concatenate, Concatenate,multiply
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, MaxPooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

def get_training_batch_xy_bias(inputsX, entlabel_train, inputsY, max_s, max_t,
                               batchsize, vocabsize, target_idex_word, lossnum, shuffle=False):
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsX) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        x_word = np.zeros((batchsize, max_s)).astype('int32')
        x_entl = np.zeros((batchsize, max_s)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize)).astype('int32')

        for idx, s in enumerate(excerpt):
            x_word[idx,] = inputsX[s]
            x_entl[idx,] = entlabel_train[s]
            for idx2, word in enumerate(inputsY[s]):
                targetvec = np.zeros(vocabsize + 1)
                targetvec = np.zeros(vocabsize)

                wordstr = ''

                if word != 0:
                    wordstr = target_idex_word[word]
                if wordstr.__contains__("E"):
                    targetvec[word] = lossnum
                else:
                    targetvec[word] = 1
                y[idx, idx2,] = targetvec
        if x_word is None:
            print("x is none !!!!!!!!!!!!!!")
        yield x_word, x_entl, y


def get_training_xy_otherset(inputsX, poslabel_train, entlabel_train, inputsY, inputsX_O, poslabel_train_O,
                             entlabel_train_O, inputsY_O, max_s, max_t,
                             vocabsize, target_idex_word, shuffle=False):
    # get any other set as validtest set
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)

    inputsX = inputsX[indices]
    inputsY = inputsY[indices]
    entlabel_train = entlabel_train[indices]
    poslabel_train = poslabel_train[indices]

    x_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_entl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_posl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    y_train = np.zeros((len(inputsX), max_t, vocabsize + 1)).astype('int32')

    for idx, s in enumerate(indices):
        x_train[idx,] = inputsX[s]
        x_entl_train[idx,] = entlabel_train[s]
        x_posl_train[idx,] = poslabel_train[s]
        for idx2, word in enumerate(inputsY[s]):
            targetvec = np.zeros(vocabsize + 1)
            # targetvec = np.zeros(vocabsize)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                else:
                    targetvec[word] = 1
            else:
                targetvec[word] = 1

            # print('targetvec',targetvec)
            y_train[idx, idx2,] = targetvec

    x_word = x_train[:]
    y = y_train[:]
    x_entl = x_entl_train[:]
    x_posl = x_posl_train[:]

    assert len(inputsX_O) == len(inputsY_O)
    indices_O = np.arange(len(inputsX_O))
    x_train_O = np.zeros((len(inputsX_O), max_s)).astype('int32')
    x_entl_train_O = np.zeros((len(inputsX_O), max_s)).astype('int32')
    x_posl_train_O = np.zeros((len(inputsX_O), max_s)).astype('int32')
    y_train_O = np.zeros((len(inputsX_O), max_t, vocabsize + 1)).astype('int32')

    for idx, s in enumerate(indices_O):
        x_train_O[idx,] = inputsX_O[s]
        x_entl_train_O[idx,] = entlabel_train_O[s]
        x_posl_train_O[idx,] = poslabel_train_O[s]
        for idx2, word in enumerate(inputsY_O[s]):
            targetvec = np.zeros(vocabsize + 1)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                else:
                    targetvec[word] = 1
            else:
                targetvec[word] = 1

            # print('targetvec',targetvec)
            y_train_O[idx, idx2,] = targetvec

    x_word_val = x_train_O[:]
    y_val = y_train_O[:]
    x_entl_val = x_entl_train_O[:]
    x_posl_val = x_posl_train_O[:]

    return x_word, x_posl, x_entl, y, x_word_val, x_posl_val, x_entl_val, y_val


def get_training_xy(inputsX, poslabel_train, entlabel_train, inputsY, max_s, max_t, vocabsize, target_idex_word,
                    shuffle=False):
    # get 0.2 of trainset as validtest set
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)

    inputsX = inputsX[indices]
    inputsY = inputsY[indices]
    entlabel_train = entlabel_train[indices]
    poslabel_train = poslabel_train[indices]

    x_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_entl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_posl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    y_train = np.zeros((len(inputsX), max_t, vocabsize + 1)).astype('int32')

    for idx, s in enumerate(indices):
        x_train[idx,] = inputsX[s]
        x_entl_train[idx,] = entlabel_train[s]
        x_posl_train[idx,] = poslabel_train[s]
        for idx2, word in enumerate(inputsY[s]):
            targetvec = np.zeros(vocabsize + 1)
            # targetvec = np.zeros(vocabsize)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                else:
                    targetvec[word] = 1
            else:
                targetvec[word] = 1

            # print('targetvec',targetvec)
            y_train[idx, idx2,] = targetvec

    num_validation_samples = int(0.2 * len(inputsX))
    x_word = x_train[:-num_validation_samples]
    y = y_train[:-num_validation_samples]
    x_entl = x_entl_train[:-num_validation_samples]
    x_posl = x_posl_train[:-num_validation_samples]

    x_word_val = x_train[-num_validation_samples:]
    y_val = y_train[-num_validation_samples:]
    x_entl_val = x_entl_train[-num_validation_samples:]
    x_posl_val = x_posl_train[-num_validation_samples:]

    return x_word, x_posl, x_entl, y, x_word_val, x_posl_val, x_entl_val, y_val


def save_model(nn_model, NN_MODEL_PATH):
    nn_model.save_weights(NN_MODEL_PATH, overwrite=True)


def creat_Model_LSTM_LSTM(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                            entlabel_W, input_seq_lenth,
                            output_seq_lenth,
                            hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)
    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1,
                                  output_dim=poslabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[poslabel_W])(posl_input)
    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1,
                                  output_dim=entlabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)
    LSTM1 = LSTM(hidden_dim, return_sequences=True, dropout=0.1)(concat_input)
    LSTM2 = LSTM(hidden_dim, return_sequences=True, dropout=0.2)(LSTM1)
    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(LSTM2)
    model = Activation('softmax')(TimeD)
    Models = Model([word_input, posl_input, entl_input], model)
    Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    return Models


def creat_Model_CNN_BiLSTM_P(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                             entlabel_W, input_seq_lenth,
                             output_seq_lenth,
                             hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1,
                                  output_dim=poslabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1,
                                  output_dim=entlabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)

    cnn = Conv1D(200, 3, activation='relu', strides=1)(concat_input)
    cnn_maxpool = MaxPooling1D(pool_size=4)(cnn)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.2))(cnn_maxpool)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(BiLSTM)

    model = Activation('softmax')(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return Models


def creat_Model_BiLSTM_CNN_P(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                             entlabel_W, input_seq_lenth,
                             output_seq_lenth,
                             hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1,
                                  output_dim=poslabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=False,
                                  trainable=True,
                                  weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1,
                                  output_dim=entlabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=False,
                                  trainable=True,
                                  weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(BiLSTM)
    cnn = Dropout(0.2)(cnn)
    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(cnn)
    model = Activation('softmax')(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return Models

def creat_Model_BiLSTM_CNN_hybrid(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM multiply CNN--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)

    concat_input_CNN1 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN1 = Dropout(0.3)(concat_input_CNN1)

    # concat_input_CNN2 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    # concat_input_CNN2 = Dropout(0.3)(concat_input_CNN2)
    #
    # concat_input_CNN3 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    # concat_input_CNN3 = Dropout(0.3)(concat_input_CNN3)
    #
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1), merge_mode='ave')(concat_input )
    # cnn1 = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    #
    # concat_LC1 = multiply([BiLSTM, cnn1])
    # concat_LC1 = Dropout(0.2)(concat_LC1)
    #
    # cnn2 = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    # maxpool =  GlobalMaxPooling1D()(cnn2)
    # repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)
    #
    #
    # concat_LC2 = concatenate([concat_LC1, repeat_maxpool], axis=-1)
    # concat_LC2 = Dropout(0.2)(concat_LC2)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1), merge_mode='concat')(concat_input)

    cnn_f2 = Conv1D(100, 2, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    maxpool_f2 =  GlobalMaxPooling1D()(cnn_f2)
    repeat_maxpool_f2 = RepeatVector(input_seq_lenth)(maxpool_f2)
    # repeat_maxpool_f2 = Dropout(0.1)(repeat_maxpool_f2)

    cnn_f3 = Conv1D(200, 3, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    maxpool_f3 =  GlobalMaxPooling1D()(cnn_f3)
    repeat_maxpool_f3 = RepeatVector(input_seq_lenth)(maxpool_f3)
    # repeat_maxpool_f3 = Dropout(0.1)(repeat_maxpool_f3)

    # cnn_f4 = Conv1D(100, 4, activation='relu', strides=1, padding='same')(concat_input_CNN3)
    # maxpool_f4 =  GlobalMaxPooling1D()(cnn_f4)
    # repeat_maxpool_f4 = RepeatVector(input_seq_lenth)(maxpool_f4)
    # repeat_maxpool_f4 = Dropout(0.1)(repeat_maxpool_f4)
    concat_cnns =concatenate([repeat_maxpool_f2,repeat_maxpool_f3], axis=-1)
    concat_cnns = Dropout(0.3)(concat_cnns)

    concat_LC2 = concatenate([BiLSTM, concat_cnns], axis=-1)
    concat_LC2 = Dropout(0.2)(concat_LC2)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC2)

    # model = Activation('softmax')(TimeD)

    crf = CRF(targetvocabsize + 1, sparse_target=False)
    model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    # Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.1), metrics=[crf.accuracy])
    Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CNN_H3_7(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM multiply CNN--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)

    concat_input_CNN1 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN1 = Dropout(0.3)(concat_input_CNN1)

    concat_input_CNN2 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN2 = Dropout(0.3)(concat_input_CNN2)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1), merge_mode='ave')(concat_input)
    cnn_3 = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    concat_LC1 = multiply([BiLSTM, cnn_3])
    concat_LC1 = Dropout(0.2)(concat_LC1)

    cnn_f2 = Conv1D(100, 2, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    maxpool_f2 =  GlobalMaxPooling1D()(cnn_f2)
    repeat_maxpool_f2 = RepeatVector(input_seq_lenth)(maxpool_f2)
    repeat_maxpool_f2 = Dropout(0.5)(repeat_maxpool_f2)

    cnn_f3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    maxpool_f3 =  GlobalMaxPooling1D()(cnn_f3)
    repeat_maxpool_f3 = RepeatVector(input_seq_lenth)(maxpool_f3)
    repeat_maxpool_f3 = Dropout(0.5)(repeat_maxpool_f3)

    cnn_f4 = Conv1D(100, 4, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    maxpool_f4 =  GlobalMaxPooling1D()(cnn_f4)
    repeat_maxpool_f4 = RepeatVector(input_seq_lenth)(maxpool_f4)
    repeat_maxpool_f4 = Dropout(0.5)(repeat_maxpool_f4)

    cnn_f5 = Conv1D(100, 5, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    # cnn_2 = Dropout(0.2)(cnn_2)
    maxpool_f5 =  GlobalMaxPooling1D()(cnn_f5)
    repeat_maxpool_f5 = RepeatVector(input_seq_lenth)(maxpool_f5)
    repeat_maxpool_f5 = Dropout(0.5)(repeat_maxpool_f5)

    concat_LC2 = concatenate([concat_LC1, repeat_maxpool_f2,repeat_maxpool_f3,repeat_maxpool_f4,repeat_maxpool_f5], axis=-1)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC2)

    model = Activation('softmax')(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return Models


def creat_Model_BiLSTM_CNN_H3_fei(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM multiply CNN--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)

    concat_input_CNN1 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN1 = Dropout(0.3)(concat_input_CNN1)

    concat_input_CNN2 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN2 = Dropout(0.3)(concat_input_CNN2)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1), merge_mode='ave')(concat_input)
    cnn_3 = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    concat_LC1 = multiply([BiLSTM, cnn_3])
    concat_LC1 = Dropout(0.2)(concat_LC1)

    cnn_f2 = Conv1D(100, 2, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    maxpool_f2 =  GlobalMaxPooling1D()(cnn_f2)
    repeat_maxpool_f2 = RepeatVector(input_seq_lenth)(maxpool_f2)
    repeat_maxpool_f2 = Dropout(0.5)(repeat_maxpool_f2)

    cnn_f3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    maxpool_f3 =  GlobalMaxPooling1D()(cnn_f3)
    repeat_maxpool_f3 = RepeatVector(input_seq_lenth)(maxpool_f3)
    repeat_maxpool_f3 = Dropout(0.5)(repeat_maxpool_f3)

    cnn_f4 = Conv1D(100, 4, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    maxpool_f4 =  GlobalMaxPooling1D()(cnn_f4)
    repeat_maxpool_f4 = RepeatVector(input_seq_lenth)(maxpool_f4)
    repeat_maxpool_f4 = Dropout(0.5)(repeat_maxpool_f4)


    concat_LC2 = concatenate([concat_LC1, repeat_maxpool_f2, repeat_maxpool_f3, repeat_maxpool_f4], axis=-1)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC2)

    model = Activation('softmax')(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return Models

def creat_Model_BiLSTM_CNN_multiply(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM multiply CNN--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)
    concat_input_CNN = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN = Dropout(0.3)(concat_input_CNN)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1), merge_mode='ave')(concat_input)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN)

    concat_LC = multiply([BiLSTM, cnn])
    concat_LC = Dropout(0.2)(concat_LC)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC)

    # model = Activation('softmax')(TimeD)

    crf = CRF(targetvocabsize + 1, sparse_target=False)
    model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    # Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    # Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.01), metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.005), metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CNN_concat(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)
    concat_input_CNN = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN = Dropout(0.3)(concat_input_CNN)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN)
    maxpool =  GlobalMaxPooling1D()(cnn)
    repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)

    concat_LC = concatenate([BiLSTM, repeat_maxpool], axis=-1)
    concat_LC = Dropout(0.2)(concat_LC)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC)

    # model = Activation('softmax')(TimeD)

    crf = CRF(targetvocabsize+1, sparse_target=False)
    model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CNN_concat_onlyw2v(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.2)(concat_input)
    concat_input_CNN = concatenate([l_A_embedding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN = Dropout(0.2)(concat_input_CNN)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN)
    maxpool =  GlobalMaxPooling1D()(cnn)
    repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)

    concat_LC = concatenate([BiLSTM, repeat_maxpool], axis=-1)
    concat_LC = Dropout(0.2)(concat_LC)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC)

    # model = Activation('softmax')(TimeD)

    crf = CRF(targetvocabsize+1, sparse_target=False)
    model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models

def creat_Model_BiLSTM_CNN_concat_embednotrain(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=False, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=False, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=False, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=False, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=False, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=False, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)
    concat_input_CNN = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN = Dropout(0.3)(concat_input_CNN)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN)
    maxpool =  GlobalMaxPooling1D()(cnn)
    repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)

    concat_LC = concatenate([BiLSTM, repeat_maxpool], axis=-1)
    concat_LC = Dropout(0.2)(concat_LC)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC)

    # model = Activation('softmax')(TimeD)

    crf = CRF(targetvocabsize+1, sparse_target=False)
    model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CRF(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W, entlabel_W, input_seq_lenth,
                          output_seq_lenth,
                          hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1,
                                  output_dim=poslabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1,
                                  output_dim=entlabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)

    concat_input = Dropout(0.3)(concat_input)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(BiLSTM)

    # model = Activation('softmax')(TimeD)

    crf = CRF(targetvocabsize+1, sparse_target=False)
    model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.01), metrics=[crf.accuracy])

    return Models


def creat_Model_CNN(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                       entlabel_W, input_seq_lenth,
                       output_seq_lenth,
                       hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1,
                                      input_length=input_seq_lenth,
                                      mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1,
                                      input_length=input_seq_lenth,
                                      mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)
    concat_input_CNN = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN = Dropout(0.3)(concat_input_CNN)



    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN)
    # maxpool = GlobalMaxPooling1D()(cnn)
    # repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)


    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(cnn)

    model = Activation('softmax')(TimeD)

    # crf = CRF(targetvocabsize+1, sparse_target=False)
    # model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models


def creat_binary_tag_LSTM(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                          entlabel_W, input_seq_lenth,
                          output_seq_lenth,
                          hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    # encoder_a = Sequential()
    # encoder_b = Sequential()
    # # encoder_c = Sequential()
    # encoder_e1 = Sequential()
    # encoder_e2 = Sequential()

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1,
                                  output_dim=poslabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1,
                                  output_dim=entlabelvobsize + 1,
                                  input_length=input_seq_lenth,
                                  mask_zero=True,
                                  trainable=True,
                                  weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    # encoder_a.add(concatenate([l_A_embedding, entlable_embeding], axis=-1))
    # encoder_a.add(Merge([encoder_e1, encoder_e2],mode='concat'))
    # encoder_a.add(l_A_embedding)
    concat_input = Dropout(0.3)(concat_input)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)
    # encoder_b.add(concatenate([encoder_e1, encoder_e2], axis=-1))
    # encoder_b.add(concatenate([l_A_embedding, entlable_embeding], axis=-1))
    # encoder_a.add(Merge([encoder_e1, encoder_e2], mode='concat'))
    # encoder_b.add(l_A_embedding)
    # encoder_b.add(Dropout(0.3))
    # encoder_c.add(l_A_embedding)
    #
    # Model = Sequential()
    #
    # encoder_a.add(LSTM(hidden_dim,return_sequences=True))
    # encoder_b.add(LSTM(hidden_dim,return_sequences=True,go_backwards=True))
    # encoder_rb = Sequential()
    # encoder_rb.add(ReverseLayer2(encoder_b))
    # encoder_ab= Merge(( encoder_a,encoder_rb),mode='concat')
    # Model.add(encoder_ab)

    # decode = LSTMDecoder_tag(hidden_dim=hidden_dim, output_dim=hidden_dim, input_length=input_seq_lenth,
    #                                     output_length=output_seq_lenth,
    #                                     state_input=False,
    #                                      return_sequences=True)
    # decodelayer = decode(BiLSTM)
    # TimeD = TimeDistributed(Dense(targetvocabsize))(decodelayer)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(BiLSTM)

    # -----------1
    model = Activation('softmax')(TimeD)
    # -----------2
    # crf = CRF(targetvocabsize+1, sparse_target=False)
    # model = crf(TimeD)
    # -----------3

    Models = Model([word_input, posl_input, entl_input], model)

    # Model.add(decodelayer)
    # Model.add(TimeDistributed(targetvocabsize+1))
    # Model.add(Activation('softmax'))

    Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    Models.summary()
    return Models


def test_model(nn_model, testdata, poslabel_testdata, entlabel_testdata, index2word, resultfile=''):
    index2word[0] = ''
    testx = np.asarray(testdata[0], dtype="int32")
    testy = np.asarray(testdata[1], dtype="int32")
    entlabel_test = np.asarray(entlabel_testdata, dtype="int32")
    poslabel_test = np.asarray(poslabel_testdata, dtype="int32")

    resultfile = resultfile + str(len(testx))

    batch_size = 50
    testlen = len(testx)
    testlinecount = 0
    if len(testx) % batch_size == 0:
        testnum = len(testx) / batch_size
    else:
        extra_test_num = batch_size - len(testx) % batch_size

        extra_data = testx[:extra_test_num]
        testx = np.append(testx, extra_data, axis=0)

        extra_data = testy[:extra_test_num]
        testy = np.append(testy, extra_data, axis=0)

        extra_data = poslabel_test[:extra_test_num]
        poslabel_test = np.append(poslabel_test, extra_data, axis=0)

        extra_data = entlabel_test[:extra_test_num]
        entlabel_test = np.append(entlabel_test, extra_data, axis=0)

        testnum = len(testx) / batch_size

    testresult = []
    # print('testnum-----',int(testnum))
    count = 0
    for n in range(0, int(testnum)):
        xbatch = testx[n * batch_size:(n + 1) * batch_size]
        plbatch = poslabel_test[n * batch_size:(n + 1) * batch_size]
        elbatch = entlabel_test[n * batch_size:(n + 1) * batch_size]
        ybatch = testy[n * batch_size:(n + 1) * batch_size]

        predictions = nn_model.predict([xbatch, plbatch, elbatch])

        for si in range(0, len(predictions)):
            if testlinecount < testlen:
                sent = predictions[si]
                ptag = []

                for word in sent:

                    next_index = np.argmax(word)
                    # if next_index != 0:
                    next_token = index2word[next_index]
                    ptag.append(next_token)

                # print('next_token--ptag--',str(ptag))
                senty = ybatch[si]
                ttag = []
                # flag =0
                for word in senty:
                    next_token = index2word[word]
                    ttag.append(next_token)
                # print('next_token--ttag--', str(ttag))
                result = []
                result.append(ptag)
                result.append(ttag)
                testlinecount += 1
                testresult.append(result)
                f = open(resultfile + '-2.txt', 'a+')
                f.write(str(ptag) + '\n')
                f.close()
                # print(result.shape)
    # print('count-----------',count)

    # pickle.dump(testresult, open(resultfile+ '2.txt', 'w'))
    #  P, R, F = evaluavtion_triple(testresult)
    P, R, F, PR_count, P_count, TR_count = evaluavtion_rel(testresult, resultfile)
    # print (P, R, F)
    return P, R, F, PR_count, P_count, TR_count


def test_model2(nn_model, testdata, poslabel_testdata, entlabel_testdata, sourc_index2word, target_index2word,
                entl_index2word, resultfile=''):
    target_index2word[0] = ''
    testx = np.asarray(testdata, dtype="int32")
    entlabel_test = np.asarray(entlabel_testdata, dtype="int32")
    poslabel_test = np.asarray(poslabel_testdata, dtype="int32")

    batch_size = 50
    testlen = len(testx)
    testlinecount = 0
    if len(testx) % batch_size == 0:
        testnum = len(testx) / batch_size
    else:
        extra_test_num = batch_size - len(testx) % batch_size

        extra_data = testx[:extra_test_num]
        testx = np.append(testx, extra_data, axis=0)

        extra_data = poslabel_test[:extra_test_num]
        poslabel_test = np.append(poslabel_test, extra_data, axis=0)

        extra_data = entlabel_test[:extra_test_num]
        entlabel_test = np.append(entlabel_test, extra_data, axis=0)

        testnum = len(testx) / batch_size

    result = []
    # print('testnum-----',int(testnum))
    count = 0
    for n in range(0, int(testnum)):
        xbatch = testx[n * batch_size:(n + 1) * batch_size]
        plbatch = poslabel_test[n * batch_size:(n + 1) * batch_size]
        elbatch = entlabel_test[n * batch_size:(n + 1) * batch_size]

        predictions = nn_model.predict([xbatch, plbatch, elbatch])

        for si in range(0, len(predictions)):
            if testlinecount < testlen:
                sent = predictions[si]
                # print('predictions',sent)
                ptag = []
                for word in sent:
                    next_index = np.argmax(word)
                    # if next_index != 0:
                    next_token = target_index2word[next_index]
                    ptag.append(next_token)
                # print('next_token--ptag--',str(ptag))

                result.append(ptag)
                testlinecount += 1

    # print('count-----------',count)
    # file_object = open(resultfile, 'w')
    # for re in result:
    #     file_object.writelines(str(re)+"\n")
    # file_object.close()
    predict_rel(testdata, entlabel_testdata, result, sourc_index2word, entl_index2word)
    # P, R, F = evaluavtion_rel(result)
    # print (P, R, F)
    # return P, R, F

def SelectModel(modelname, sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                             entlabel_W, input_seq_lenth,
                             output_seq_lenth,
                             hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    nn_model = None
    if modelname is 'creat_binary_tag_LSTM':
        nn_model = creat_binary_tag_LSTM(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                         poslabelvobsize=poslabelvobsize,
                                         entlabelvobsize=entlabelvobsize,
                                         source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                         input_seq_lenth=input_seq_lenth,
                                         output_seq_lenth=output_seq_lenth,
                                         hidden_dim=hidden_dim, emd_dim=emd_dim)
    elif modelname is 'creat_Model_LSTM_LSTM':
        nn_model = creat_Model_LSTM_LSTM(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                         poslabelvobsize=poslabelvobsize,
                                         entlabelvobsize=entlabelvobsize,
                                         source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                         input_seq_lenth=input_seq_lenth,
                                         output_seq_lenth=output_seq_lenth,
                                         hidden_dim=hidden_dim, emd_dim=emd_dim)
    elif modelname is 'creat_Model_CNN_BiLSTM_P':
        nn_model = creat_Model_CNN_BiLSTM_P(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                         poslabelvobsize=poslabelvobsize,
                                         entlabelvobsize=entlabelvobsize,
                                         source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                         input_seq_lenth=input_seq_lenth,
                                         output_seq_lenth=output_seq_lenth,
                                         hidden_dim=hidden_dim, emd_dim=emd_dim)
    elif modelname is 'creat_Model_BiLSTM_CNN_P':
        nn_model = creat_Model_BiLSTM_CNN_P(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                         poslabelvobsize=poslabelvobsize,
                                         entlabelvobsize=entlabelvobsize,
                                         source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                         input_seq_lenth=input_seq_lenth,
                                         output_seq_lenth=output_seq_lenth,
                                         hidden_dim=hidden_dim, emd_dim=emd_dim)
    elif modelname is 'creat_Model_BiLSTM_CNN_multiply':
        nn_model = creat_Model_BiLSTM_CNN_multiply(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                         poslabelvobsize=poslabelvobsize,
                                         entlabelvobsize=entlabelvobsize,
                                         source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                         input_seq_lenth=input_seq_lenth,
                                         output_seq_lenth=output_seq_lenth,
                                         hidden_dim=hidden_dim, emd_dim=emd_dim)
    elif modelname is 'creat_Model_BiLSTM_CNN_concat':
        nn_model = creat_Model_BiLSTM_CNN_concat(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                         poslabelvobsize=poslabelvobsize,
                                         entlabelvobsize=entlabelvobsize,
                                         source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                         input_seq_lenth=input_seq_lenth,
                                         output_seq_lenth=output_seq_lenth,
                                         hidden_dim=hidden_dim, emd_dim=emd_dim)

    elif modelname is 'creat_Model_BiLSTM_CNN_concat_embednotrain':
        nn_model = creat_Model_BiLSTM_CNN_concat_embednotrain(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                 poslabelvobsize=poslabelvobsize,
                                                 entlabelvobsize=entlabelvobsize,
                                                 source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                                 input_seq_lenth=input_seq_lenth,
                                                 output_seq_lenth=output_seq_lenth,
                                                 hidden_dim=hidden_dim, emd_dim=emd_dim)

    elif modelname is 'creat_Model_BiLSTM_CNN_concat_onlyw2v':
        nn_model = creat_Model_BiLSTM_CNN_concat_onlyw2v(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                 poslabelvobsize=poslabelvobsize,
                                                 entlabelvobsize=entlabelvobsize,
                                                 source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                                 input_seq_lenth=input_seq_lenth,
                                                 output_seq_lenth=output_seq_lenth,
                                                 hidden_dim=hidden_dim, emd_dim=emd_dim)


    elif modelname is 'creat_Model_BiLSTM_CNN_hybrid':
        nn_model = creat_Model_BiLSTM_CNN_hybrid(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                             poslabelvobsize=poslabelvobsize,
                                             entlabelvobsize=entlabelvobsize,
                                             source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                             input_seq_lenth=input_seq_lenth,
                                             output_seq_lenth=output_seq_lenth,
                                             hidden_dim=hidden_dim, emd_dim=emd_dim)
    elif modelname is 'creat_Model_BiLSTM_CRF':
        nn_model = creat_Model_BiLSTM_CRF(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                             poslabelvobsize=poslabelvobsize,
                                             entlabelvobsize=entlabelvobsize,
                                             source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                             input_seq_lenth=input_seq_lenth,
                                             output_seq_lenth=output_seq_lenth,
                                             hidden_dim=hidden_dim, emd_dim=emd_dim)

    elif modelname is 'creat_Model_CNN':
        nn_model = creat_Model_CNN(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                   poslabelvobsize=poslabelvobsize,
                                   entlabelvobsize=entlabelvobsize,
                                   source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                   input_seq_lenth=input_seq_lenth,
                                   output_seq_lenth=output_seq_lenth,
                                   hidden_dim=hidden_dim, emd_dim=emd_dim)
    return nn_model


def train_e2e_model(Modelname, eelstmfile, modelfile, resultdir, npochos,
                    lossnum=1, batch_size=50, retrain=False):
    # load training data and test data

    traindata, testdata, source_W, source_vob, sourc_idex_word, \
    target_vob, target_idex_word, max_s, k, \
    entlabel_traindata, entlabel_testdata, entlabel_W, entlabel_vob, entlabel_idex_word, \
    poslabel_traindata, poslabel_testdata, poslabel_W, poslabel_vob, poslabel_idex_word = pickle.load(
        open(eelstmfile, 'rb'))

    # train model
    x_train = np.asarray(traindata[0], dtype="int32")
    y_train = np.asarray(traindata[1], dtype="int32")
    entlabel_train = np.asarray(entlabel_traindata, dtype="int32")
    poslabel_train = np.asarray(poslabel_traindata, dtype="int32")

    x_test = np.asarray(testdata[0], dtype="int32")
    y_test = np.asarray(testdata[1], dtype="int32")
    entlabel_test = np.asarray(entlabel_testdata, dtype="int32")
    poslabel_test = np.asarray(poslabel_testdata, dtype="int32")
    nn_model = SelectModel(Modelname, sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                     poslabelvobsize=len(poslabel_vob),
                                     entlabelvobsize=len(entlabel_vob),
                                     source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                                     input_seq_lenth=max_s,
                                     output_seq_lenth=max_s,
                                     hidden_dim=200, emd_dim=k)

    if retrain:
        nn_model.load_weights(modelfile)

    nn_model.summary()

    epoch = 0
    save_inter = 1
    saveepoch = save_inter
    maxF = 0

    # x_word, x_posl, x_entl, y, x_word_val, x_posl_val, x_entl_val, y_val,= get_training_xy_otherset(x_train, poslabel_train, entlabel_train, y_train,
    #                                                                             x_test, poslabel_test, entlabel_test, y_test,
    #                                                                             max_s, max_s,
    #                                                                             len(target_vob), target_idex_word,
    #                                                                             shuffle=True)
    x_word, x_posl, x_entl, y, x_word_val, x_posl_val, x_entl_val, y_val = get_training_xy(x_train, poslabel_train,
                                                                                           entlabel_train, y_train,
                                                                                           max_s, max_s,
                                                                                           len(target_vob),
                                                                                           target_idex_word,
                                                                                           shuffle=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpointer = ModelCheckpoint(filepath="./data/demo/model/best_e2e_lstmb_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.0001)
    nn_model.fit([x_word, x_posl, x_entl], y,
                 batch_size=batch_size,
                 epochs=100,
                 verbose=1,
                 shuffle=True,
                 # validation_split=0.2,
                 validation_data=([x_word_val, x_posl_val, x_entl_val], y_val),
                 callbacks=[reduce_lr, checkpointer, early_stopping])

    save_model(nn_model, modelfile)
    # nn_model.save(modelfile, overwrite=True)

    # while (epoch < npochos):
    #     epoch = epoch + 1
    #     for x_word,x_entl, y in get_training_batch_xy_bias(x_train,entlabel_train, y_train, max_s, max_s,
    #                                       batch_size, len(target_vob),
    #                                         target_idex_word,lossnum,shuffle=True):
    #         nn_model.fit([x_word, x_entl], y, batch_size=batch_size, epochs=1, verbose=0)
    #         if epoch >= saveepoch:
    #         # if epoch >=0:
    #             saveepoch += save_inter
    #             resultfile = resultdir+"result-"+str(saveepoch)
    #             P, R, F = test_model(nn_model, testdata,entlabel_testdata, target_idex_word,resultfile)
    #
    #             if F > maxF:
    #                 maxF=F
    #                 save_model(nn_model, modelfile)
    #             print(P, R, F,'  maxF=',maxF)
    return nn_model


def infer_e2e_model(modelname, eelstmfile, lstm_modelfile, resultdir):
    # traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, \
    # target_idex_word, max_s, k \
    #     = pickle.load(open(eelstmfile, 'rb'))

    # load training data and test data
    inputs = open(eelstmfile, 'rb')
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k, \
    entlabel_traindata, entlabel_testdata, entlabel_W, entlabel_vob, entlabel_idex_word, \
    poslabel_traindata, poslabel_testdata, poslabel_W, poslabel_vob, poslabel_idex_word = pickle.load(
        inputs)
    inputs.close()

    nnmodel = SelectModel(modelname, sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                           poslabelvobsize=len(poslabel_vob),
                           entlabelvobsize=len(entlabel_vob),
                           source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                           input_seq_lenth=max_s,
                           output_seq_lenth=max_s,
                           hidden_dim=200, emd_dim=k)

    nnmodel.load_weights(lstm_modelfile)
    # nnmodel = load_model(lstm_modelfile)
    resultfile = resultdir + "result-" + 'infer_test'

    P, R, F, PR_count, P_count, TR_count= test_model(nnmodel, testdata, poslabel_testdata, entlabel_testdata, target_idex_word, resultfile)
    print('P= ', P, '  R= ', R, '  F= ', F)


def infer_e2e_model2(modelname, eelstmfile, lstm_modelfile, resultdir, testfile, poslabelfile_test, entlabelfile_test):
    # traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, \
    # target_idex_word, max_s, k \
    #     = pickle.load(open(eelstmfile, 'rb'))

    # load training data and test data
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k, \
    entlabel_traindata, entlabel_testdata, entlabel_W, entlabel_vob, entlabel_idex_word, \
    poslabel_traindata, poslabel_testdata, poslabel_W, poslabel_vob, poslabel_idex_word = pickle.load(
        open(eelstmfile, 'rb'))

    testdata2 = make_idx_data_index_EE_LSTM(testfile, max_s, source_vob, target_vob)

    # entlabel_vob, entlabel_idex_word, entlabel_max_s = get_Feature_index(entlabelfile_test)
    entlabel_testdata2 = make_idx_data_index_EE_LSTM2(entlabelfile_test, max_s, entlabel_vob)

    # poslabel_vob, poslabel_idex_word, poslabel_max_s = get_Feature_index(poslabelfile_test)
    poslabel_testdata2 = make_idx_data_index_EE_LSTM2(poslabelfile_test, max_s, poslabel_vob)

    nnmodel = SelectModel(modelname, sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                           poslabelvobsize=len(poslabel_vob),
                           entlabelvobsize=len(entlabel_vob),
                           source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                           input_seq_lenth=max_s,
                           output_seq_lenth=max_s,
                           hidden_dim=200, emd_dim=k)

    nnmodel.load_weights(lstm_modelfile)
    resultfile = resultdir + "result-" + 'infer_test'

    P, R, F, PR_count, P_count, TR_count = test_model(nnmodel, testdata2, poslabel_testdata2, entlabel_testdata2, target_idex_word, resultfile)
    print('P= ', P, '  R= ', R, '  F= ', F)
    return PR_count, P_count, TR_count

def infer_e2e_model3(modelname, eelstmfile, lstm_modelfile, resultdir, testfile, test2file_pos):
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k, \
    entlabel_traindata, entlabel_testdata, entlabel_W, entlabel_vob, entlabel_idex_word, \
    poslabel_traindata, poslabel_testdata, poslabel_W, poslabel_vob, poslabel_idex_word = pickle.load(
        open(eelstmfile, 'rb'))

    testdata = make_idx_data_index_EE_LSTM3(testfile, max_s, source_vob)

    entlabel_testdata = make_idx_data_index_EE_LSTM2(testfile, max_s, entlabel_vob)

    poslabel_testdata = make_idx_data_index_EE_LSTM2(test2file_pos, max_s, poslabel_vob)

    nnmodel = SelectModel(modelname, sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                           poslabelvobsize=len(poslabel_vob),
                           entlabelvobsize=len(entlabel_vob),
                           source_W=source_W, poslabel_W=poslabel_W, entlabel_W=entlabel_W,
                           input_seq_lenth=max_s,
                           output_seq_lenth=max_s,
                           hidden_dim=200, emd_dim=k)

    nnmodel.load_weights(lstm_modelfile)
    resultfile = resultdir + "result-" + 'label_test.txt'

    test_model2(nnmodel, testdata, poslabel_testdata, entlabel_testdata, sourc_idex_word, target_idex_word,
                entlabel_idex_word, resultfile)


if __name__ == "__main__":

    alpha = 1
    maxlen = 50

    # modelname = 'creat_binary_tag_LSTM'
    # modelname = 'creat_Model_LSTM_LSTM'
    # modelname = 'creat_Model_BiLSTM_CNN_P'
    # modelname = 'creat_Model_CNN_BiLSTM_P'
    # modelname = 'creat_Model_BiLSTM_CRF'
    modelname = 'creat_Model_BiLSTM_CNN_concat'
    # modelname = 'creat_Model_BiLSTM_CNN_concat_embednotrain'
    # modelname = 'creat_Model_BiLSTM_CNN_concat_onlyw2v'
    # modelname = 'creat_Model_BiLSTM_CNN_multiply'
    # modelname = 'creat_Model_BiLSTM_CNN_hybrid'
    # modelname = 'creat_Model_CNN'
    # modelname = 'creat_Model_CNN'
    print(modelname)

    w2v_file = "./data/demo/1billion_news_andother_en_reverb_w2v.txt"
    e2edatafile = "./data/demo/model/e2edata.pkl"
    modelfile = "./data/demo/model/e2e_lstmb_model.h5"
    resultdir = "./data/demo/result/"

    trainfile = "./data/demo/Seq2SeqSet-train-rellabel.json"
    entlabelfile_train = "./data/demo/Seq2SeqSet-train-entlabel.json"
    poslabelfile_train = "./data/demo/Seq2SeqSet-train-poslabel.json"

    testfile = "./data/demo/Seq2SeqSet-test-rellabel.json"
    entlabelfile_test = "./data/demo/Seq2SeqSet-test-entlabel.json"
    poslabelfile_test = "./data/demo/Seq2SeqSet-test-poslabel.json"

    testfile_wiki_ollie = "./data/demo/Seq2SeqSet-test-wikipedia-ollie-rellabel.json"
    entlabelfile_test_wiki_ollie = "./data/demo/Seq2SeqSet-test-wikipedia-ollie-entlabel.json"
    poslabelfile_test_wiki_ollie = "./data/demo/Seq2SeqSet-test-wikipedia-ollie-poslabel.json"

    testfile_wiki_clausie = "./data/demo/Seq2SeqSet-test-wikipedia-clausie-rellabel.json"
    entlabelfile_test_wiki_clausie = "./data/demo/Seq2SeqSet-test-wikipedia-clausie-entlabel.json"
    poslabelfile_test_wiki_clausie = "./data/demo/Seq2SeqSet-test-wikipedia-clausie-poslabel.json"

    testfile_nyt_clausie = "./data/demo/Seq2SeqSet-test-nyt-clausie-rellabel.json"
    entlabelfile_test_nyt_clausie = "./data/demo/Seq2SeqSet-test-nyt-clausie-entlabel.json"
    poslabelfile_test_nyt_clausie = "./data/demo/Seq2SeqSet-test-nyt-clausie-poslabel.json"

    testfile_nyt_ollie = "./data/demo/Seq2SeqSet-test-nyt-ollie-rellabel.json"
    entlabelfile_test_nyt_ollie = "./data/demo/Seq2SeqSet-test-nyt-ollie-entlabel.json"
    poslabelfile_test_nyt_ollie = "./data/demo/Seq2SeqSet-test-nyt-ollie-poslabel.json"

    testfile_reverb_ollie = "./data/demo/Seq2SeqSet-test-reverb-ollie-rellabel.json"
    entlabelfile_test_reverb_ollie = "./data/demo/Seq2SeqSet-test-reverb-ollie-entlabel.json"
    poslabelfile_test_reverb_ollie = "./data/demo/Seq2SeqSet-test-reverb-ollie-poslabel.json"

    testfile_reverb_clausie = "./data/demo/Seq2SeqSet-test-reverb-clausie-rellabel.json"
    entlabelfile_test_reverb_clausie = "./data/demo/Seq2SeqSet-test-reverb-clausie-entlabel.json"
    poslabelfile_test_reverb_clausie = "./data/demo/Seq2SeqSet-test-reverb-clausie-poslabel.json"

    test2file_reverb = "./data/demo/Seq2SeqSet-test2-reverb-NPlabel.json"
    test2file_reverb_pos = "./data/demo/Seq2SeqSet-test2-reverb-poslabel.json"

    testfile_nyt_onlyright = "./data/demo/Seq2SeqSet-test-nyt-onlyright-rellabel.json"
    entlabelfile_test_nyt_onlyright = "./data/demo/Seq2SeqSet-test-nyt-onlyright-entlabel.json"
    poslabelfile_test_nyt_onlyright = "./data/demo/Seq2SeqSet-test-nyt-onlyright-poslabel.json"

    testfile_wiki_onlyright = "./data/demo/Seq2SeqSet-test-wikipedia-onlyright-rellabel.json"
    entlabelfile_test_wiki_onlyright = "./data/demo/Seq2SeqSet-test-wikipedia-onlyright-entlabel.json"
    poslabelfile_test_wiki_onlyright = "./data/demo/Seq2SeqSet-test-wikipedia-onlyright-poslabel.json"

    testfile_reverb_onlyright = "./data/demo/Seq2SeqSet-test-reverb-onlyright-rellabel.json"
    entlabelfile_test_reverb_onlyright = "./data/demo/Seq2SeqSet-test-reverb-onlyright-entlabel.json"
    poslabelfile_test_reverb_onlyright = "./data/demo/Seq2SeqSet-test-reverb-onlyright-poslabel.json"

    testfile_OIE_onlyright = "./data/demo/Seq2SeqSet-test-OIE-onlyright-rellabel.json"
    entlabelfile_test_OIE_onlyright = "./data/demo/Seq2SeqSet-test-OIE-onlyright-entlabel.json"
    poslabelfile_test_OIE_onlyright = "./data/demo/Seq2SeqSet-test-OIE-onlyright-poslabel.json"

    testfile_OIE_ClausIE = "./data/demo/Seq2SeqSet-test-OIE-ClausIE-rellabel.json"
    entlabelfile_test_OIE_ClausIE = "./data/demo/Seq2SeqSet-test-OIE-ClausIE-entlabel.json"
    poslabelfile_test_OIE_ClausIE = "./data/demo/Seq2SeqSet-test-OIE-ClausIE-poslabel.json"

    testfile_OIE_OpenIE4 = "./data/demo/Seq2SeqSet-test-OIE-OpenIE4-rellabel.json"
    entlabelfile_test_OIE_OpenIE4 = "./data/demo/Seq2SeqSet-test-OIE-OpenIE4-entlabel.json"
    poslabelfile_test_OIE_OpenIE4 = "./data/demo/Seq2SeqSet-test-OIE-OpenIE4-poslabel.json"

    testfile_OIE_OLLIE = "./data/demo/Seq2SeqSet-test-OIE-OLLIE-rellabel.json"
    entlabelfile_test_OIE_OLLIE = "./data/demo/Seq2SeqSet-test-OIE-OLLIE-entlabel.json"
    poslabelfile_test_OIE_OLLIE = "./data/demo/Seq2SeqSet-test-OIE-OLLIE-poslabel.json"

    retrain = False
    Test = True
    valid = False
    Label = False
    if not os.path.exists(e2edatafile):
        print("Precess lstm data....")
        get_data_e2e(trainfile, testfile, w2v_file, e2edatafile,
                     poslabelfile_train, poslabelfile_test,
                     entlabelfile_train, entlabelfile_test, maxlen=maxlen)
    if not os.path.exists(modelfile):
        print("Lstm data has extisted: " + e2edatafile)
        print("Training EE model....")
        train_e2e_model(modelname, e2edatafile, modelfile, resultdir,
                        npochos=100, lossnum=alpha, retrain=False)
    else:
        if retrain:
            print("ReTraining EE model....")
            train_e2e_model(modelname, e2edatafile, modelfile, resultdir,
                            npochos=100, lossnum=alpha, retrain=retrain)
    if Label:
        print("label EE model....")
        infer_e2e_model3(modelname, e2edatafile, modelfile, resultdir, test2file_reverb, test2file_reverb_pos)

    if Test:
        print("test EE model....")
        # infer_e2e_model(modelname, e2edatafile, modelfile, resultdir)

        print("。。。。。。。。only right wiki - nyt - reverb ...")
        PR_count_wiki, P_count_wiki, TR_count_wiki = infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_wiki_onlyright, poslabelfile_test_wiki_onlyright,
                         entlabelfile_test_wiki_onlyright)
        PR_count_nyt, P_count_nyt, TR_count_nyt = infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_nyt_onlyright, poslabelfile_test_nyt_onlyright,
                         entlabelfile_test_nyt_onlyright)
        PR_count_reverb, P_count_reverb, TR_count_reverb = infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_reverb_onlyright,
                         poslabelfile_test_reverb_onlyright,
                         entlabelfile_test_reverb_onlyright)

        P = float(PR_count_wiki+PR_count_nyt+PR_count_reverb)/(P_count_wiki+P_count_nyt+P_count_reverb)
        R = float(PR_count_wiki + PR_count_nyt + PR_count_reverb) / (TR_count_wiki + TR_count_nyt + TR_count_reverb)
        F = (2 * P * R) / float(P + R) if P != 0 else 0
        print("!!!!only right---Total precision is ", P)
        print("!!!!only right---Total recall is ", R)
        print("!!!!only right---Total Fscore is ", F)

        PR_count_OIE, P_count_OIE, TR_count_OIE = infer_e2e_model2(modelname, e2edatafile, modelfile,
                                                                            resultdir, testfile_OIE_onlyright,
                                                                            poslabelfile_test_OIE_onlyright,
                                                                            entlabelfile_test_OIE_onlyright)

        # PR_count_OIE_ClausIE, P_count_OIE_ClausIE, TR_count_OIE_ClausIE = infer_e2e_model2(modelname, e2edatafile, modelfile,
        #                                                            resultdir, testfile_OIE_ClausIE,
        #                                                            poslabelfile_test_OIE_ClausIE,
        #                                                            entlabelfile_test_OIE_ClausIE)
        #
        # PR_count_OIE_OpenIE4, P_count_OIE_OpenIE4, TR_count_OIE_OpenIE4 = infer_e2e_model2(modelname, e2edatafile, modelfile,
        #                                                            resultdir, testfile_OIE_OpenIE4,
        #                                                            poslabelfile_test_OIE_OpenIE4,
        #                                                            entlabelfile_test_OIE_OpenIE4)
        #
        # PR_count_OIE_OLLIE, P_count_OIE_OLLIE, TR_count_OIE_OLLIE = infer_e2e_model2(modelname, e2edatafile, modelfile,
        #                                                            resultdir, testfile_OIE_OLLIE,
        #                                                            poslabelfile_test_OIE_OLLIE,
        #                                                            entlabelfile_test_OIE_OLLIE)


        # print("。。。。。。。。addition wiki-ollie-clausie test ...")
        # PR_count_wiki_o, P_count_wiki_o, TR_count_wiki_o = infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_wiki_ollie, poslabelfile_test_wiki_ollie,
        #                  entlabelfile_test_wiki_ollie)
        # PR_count_wiki_c, P_count_wiki_c, TR_count_wiki_c = infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_wiki_clausie, poslabelfile_test_wiki_clausie,
        #                  entlabelfile_test_wiki_clausie)
        #
        # P = float(PR_count_wiki_o + PR_count_wiki_c) / (P_count_wiki_o + P_count_wiki_c)
        # R = float(PR_count_wiki_o + PR_count_wiki_c) / (TR_count_wiki_o + TR_count_wiki_c)
        # F = (2 * P * R) / float(P + R) if P != 0 else 0
        # print("!!!!wiki precision is ", P)
        # print("!!!!wiki recall is ", R)
        # print("!!!!wiki Fscore is ", F)


        # print("。。。。。。。。addition nyt-ollie-clausie test ...")
        # PR_count_nyt_o, P_count_nyt_o, TR_count_nyt_o= infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_nyt_ollie, poslabelfile_test_nyt_ollie,
        #                  entlabelfile_test_nyt_ollie)
        # PR_count_nyt_c, P_count_nyt_c, TR_count_nyt_c= infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_nyt_clausie, poslabelfile_test_nyt_clausie,
        #                  entlabelfile_test_nyt_clausie)
        #
        # P = float(PR_count_nyt_o + PR_count_nyt_c) / (P_count_nyt_o + P_count_nyt_c)
        # R = float(PR_count_nyt_o + PR_count_nyt_c) / (TR_count_nyt_o + TR_count_nyt_c)
        # F = (2 * P * R) / float(P + R) if P != 0 else 0
        # print("!!!!nyt precision is ", P)
        # print("!!!!nyt recall is ", R)
        # print("!!!!nyt Fscore is ", F)
        #
        # print("。。。。。。。。addition reverb-ollie-clausie test ...")
        # PR_count_reverb_o, P_count_reverb_o, TR_count_reverb_o= infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_reverb_ollie, poslabelfile_test_reverb_ollie,
        #                  entlabelfile_test_reverb_ollie)
        # PR_count_reverb_c, P_count_reverb_c, TR_count_reverb_c= infer_e2e_model2(modelname, e2edatafile, modelfile, resultdir, testfile_reverb_clausie, poslabelfile_test_reverb_clausie,
        #                  entlabelfile_test_reverb_clausie)
        #
        # P = float(PR_count_reverb_o + PR_count_reverb_c) / (P_count_reverb_o + P_count_reverb_c)
        # R = float(PR_count_reverb_o + PR_count_reverb_c) / (TR_count_reverb_o + TR_count_reverb_c)
        # F = F = (2 * P * R) / float(P + R) if P != 0 else 0
        # print("!!!!reverb precision is ", P)
        # print("!!!!reverb recall is ", R)
        # print("!!!!reverb Fscore is ", F)
        #
        # P = float(PR_count_wiki_o + PR_count_wiki_c + PR_count_nyt_o + PR_count_nyt_c + PR_count_reverb_o + PR_count_reverb_c) / \
        #     (P_count_wiki_o + P_count_wiki_c + P_count_nyt_o + P_count_nyt_c + P_count_reverb_o + P_count_reverb_c)
        # R = float(
        #     PR_count_wiki_o + PR_count_wiki_c + PR_count_nyt_o + PR_count_nyt_c + PR_count_reverb_o + PR_count_reverb_c) / \
        #     (TR_count_wiki_o + TR_count_wiki_c + TR_count_nyt_o + TR_count_nyt_c + TR_count_reverb_o + TR_count_reverb_c)
        # F = F = (2 * P * R) / float(P + R) if P != 0 else 0
        #
        # print("!!!!Total precision is ", P)
        # print("!!!!Total recall is ", R)
        # print("!!!!Total Fscore is ", F)




    '''
    lstm hidenlayer,
    bash size,
    epoach
    '''
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES="" python End2EndModel.py



