# Copyright 2020 Vraj Shah, Arun Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPool1D,
    Input,
    concatenate,
)
from keras.models import Model, load_model
from keras.preprocessing import sequence as keras_seq
from keras.preprocessing import text as keras_text
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# define network parameters
max_features = 256
maxlen = 256


def abs_limit(x):
    if abs(x) > 10000:
        return 10000 * np.sign(x)
    return x


def prepare_data(data, y):

    df = data[
        [
            'total_vals',
            'num_nans',
            '%_nans',
            'num_of_dist_val',
            '%_dist_val',
            'mean',
            'std_dev',
            'min_val',
            'max_val',
            'mean_word_count',
            'std_dev_word_count',
            'mean_stopword_total',
            'mean_whitespace_count',
            'mean_char_count',
            'mean_delim_count',
            'stdev_stopword_total',
            'stdev_whitespace_count',
            'stdev_char_count',
            'stdev_delim_count',
        ]
    ]

    df = df.reset_index(drop=True)
    df = df.fillna(0)

    df = df.rename(
        columns={
            'mean': 'scaled_mean',
            'std_dev': 'scaled_std_dev',
            'min_val': 'scaled_min',
            'max_val': 'scaled_max',
            'mean_word_count': 'scaled_mean_token_count',
            'std_dev_word_count': 'scaled_std_dev_token_count',
            '%_nans': 'scaled_perc_nans',
            'mean_stopword_total': 'scaled_mean_stopword_total',
            'mean_whitespace_count': 'scaled_mean_whitespace_count',
            'mean_char_count': 'scaled_mean_char_count',
            'mean_delim_count': 'scaled_mean_delim_count',
            'stdev_stopword_total': 'scaled_stdev_stopword_total',
            'stdev_whitespace_count': 'scaled_stdev_whitespace_count',
            'stdev_char_count': 'scaled_stdev_char_count',
            'stdev_delim_count': 'scaled_stdev_delim_count',
        }
    )

    df['scaled_mean'] = df['scaled_mean'].apply(abs_limit)
    df['scaled_std_dev'] = df['scaled_std_dev'].apply(abs_limit)
    df['scaled_min'] = df['scaled_min'].apply(abs_limit)
    df['scaled_max'] = df['scaled_max'].apply(abs_limit)
    df['total_vals'] = df['total_vals'].apply(abs_limit)
    df['num_nans'] = df['num_nans'].apply(abs_limit)
    df['num_of_dist_val'] = df['num_of_dist_val'].apply(abs_limit)

    column_names_to_normalize = [
        'total_vals',
        'num_nans',
        'num_of_dist_val',
        'scaled_mean',
        'scaled_std_dev',
        'scaled_min',
        'scaled_max',
    ]
    x = df[column_names_to_normalize].values
    x = np.nan_to_num(x)
    x_scaled = StandardScaler().fit_transform(x)
    df_temp = pd.DataFrame(
        x_scaled, columns=column_names_to_normalize, index=df.index
    )
    df[column_names_to_normalize] = df_temp

    y.y_act = y.y_act.astype(float)
    return df


# %%
X_train = pd.read_csv('../../Benchmark-Labeled-Data/data_train.csv')
X_test = pd.read_csv('../../Benchmark-Labeled-Data/data_test.csv')

# for i in range(0,1000,10):
X_train = X_train.sample(frac=1, random_state=100).reset_index(drop=True)
# print(len(xtrain))

atr_train = X_train.loc[:, ['Attribute_name']]
atr_test = X_test.loc[:, ['Attribute_name']]
# print(atr_train)

samp_train = X_train.loc[:, ['sample_1']]
samp_test = X_test.loc[:, ['sample_1']]

y_train = X_train.loc[:, ['y_act']]
y_test = X_test.loc[:, ['y_act']]


dict_label = {
    'numeric': 0,
    'categorical': 1,
    'datetime': 2,
    'sentence': 3,
    'url': 4,
    'embedded-number': 5,
    'list': 6,
    'not-generalizable': 7,
    'context-specific': 8,
}

y_train['y_act'] = [dict_label[i] for i in y_train['y_act']]
y_test['y_act'] = [dict_label[i] for i in y_test['y_act']]
y_train


X_train = prepare_data(X_train, y_train)
X_test = prepare_data(X_test, y_test)


# X_train = func2(xtrain,xtrain1,1)
# X_test = func2(xtest,xtest1,0)
# print(atr_train)
print(atr_train['Attribute_name'].values)

X_train.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
# atr_train.reset_index(inplace=True,drop=True)
# atr_test.reset_index(inplace=True,drop=True)


X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values


# atr_train = atr_train.values
# atr_test = atr_test.values


structured_data_train = X_train
structured_data_test = X_test


list_sentences_train = atr_train['Attribute_name'].values
list_sentences_test = atr_test['Attribute_name'].values

list_sentences_train1 = samp_train['sample_1'].values
list_sentences_test1 = samp_test['sample_1'].values


print(list_sentences_train)

for i in range(len(list_sentences_train)):
    list_sentences_train[i] = str(list_sentences_train[i])
for i in range(len(list_sentences_test)):
    list_sentences_test[i] = str(list_sentences_test[i])


for i in range(len(list_sentences_train1)):
    list_sentences_train1[i] = str(list_sentences_train1[i])
for i in range(len(list_sentences_test1)):
    list_sentences_test1[i] = str(list_sentences_test1[i])

print(list_sentences_train)


tokenizer = keras_text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(list(list_sentences_train))


tokenizer1 = keras_text.Tokenizer(char_level=True)
tokenizer1.fit_on_texts(list(list_sentences_train1))

# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = keras_seq.pad_sequences(list_tokenized_train, maxlen=maxlen)

list_tokenized_train1 = tokenizer.texts_to_sequences(list_sentences_train1)
X_t1 = keras_seq.pad_sequences(list_tokenized_train1, maxlen=maxlen)

# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = keras_seq.pad_sequences(list_tokenized_test, maxlen=maxlen)

list_tokenized_test1 = tokenizer.texts_to_sequences(list_sentences_test1)
X_te1 = keras_seq.pad_sequences(list_tokenized_test1, maxlen=maxlen)


# %%
def build_model(neurons, numfilters, embed_size):
    inp = Input(shape=(None,))
    x = Embedding(
        input_dim=len(tokenizer.word_counts) + 1, output_dim=embed_size
    )(inp)
    #     prefilt_x = Dropout(0.5)(x)
    out_conv = []

    #     x = prefilt_x
    for i in range(2):
        x = Conv1D(
            numfilters,
            kernel_size=3,
            activation='tanh',
            kernel_initializer='glorot_normal',
        )(x)
        numfilters = numfilters * 2

    #     out_conv += [Dropout(0.5)(GlobalMaxPool1D()(x))]
    out_conv += [(GlobalMaxPool1D()(x))]
    #     xy = Flatten()(out_conv)
    out_conv += [GlobalMaxPool1D()(x)]
    x += [GlobalMaxPool1D()(x)]
    xy = concatenate(out_conv, axis=-1)

    inp1 = Input(shape=(None,))
    x = Embedding(
        input_dim=len(tokenizer.word_counts) + 1, output_dim=embed_size
    )(inp1)
    out_conv = []

    for i in range(2):
        x = Conv1D(
            numfilters,
            kernel_size=3,
            activation='tanh',
            kernel_initializer='glorot_normal',
        )(x)
        numfilters = numfilters * 2

    out_conv += [(GlobalMaxPool1D()(x))]
    out_conv += [GlobalMaxPool1D()(x)]
    x += [GlobalMaxPool1D()(x)]
    xy1 = concatenate(out_conv, axis=-1)

    Str_input = Input(shape=(19,))
    layersfin = keras.layers.concatenate([xy, xy1, Str_input])
    x = BatchNormalization()(layersfin)
    #     x = Dense(1000, activation='tanh',kernel_initializer='glorot_uniform')(Str_input)

    x = Dense(neurons, activation='tanh')(x)
    x = Dropout(0.5)(x)
    #     x = Dense(500, activation='tanh')(x)
    #     x = Dense(neurons, activation='relu')(x)
    x = Dense(neurons, activation='relu')(x)
    x = Dropout(0.5)(x)
    #     x = Dense(1000, activation='relu',kernel_initializer='random_normal')(x)
    #     x = Dense(1000, activation='tanh')(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(inputs=[inp, inp1, Str_input], outputs=[x])
    opt = keras.optimizers.Adam(learning_rate=3e-3)
    opt = keras.optimizers.RMSprop(learning_rate=1e-2)
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
    )
    return model


model = build_model(100, 100, 100)
model.summary()


# %%
print(X_t)
print(X_t.shape)
# print(y_train[1851:])
print(len(y_train))
print(structured_data_train)

y_train = y_train.values
structured_data_train = structured_data_train.values

# %%
batch_size = 128
epochs = 25

k = 5
kf = KFold(n_splits=k)

neurons = [100, 500, 1000]
n_filters_grid = [32, 64, 128]
embed_size = [64, 128, 256]


models = []

avgsc_lst, avgsc_val_lst, avgsc_train_lst = [], [], []
avgsc, avgsc_val, avgsc_train = 0, 0, 0
i = 0
for train_index, test_index in kf.split(X_t):
    #     if i==1: break
    file_path = 'CNN_best_model' + str(i) + '.h5'

    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
    )

    callbacks_list = [checkpoint]  # early

    #     print('\n')
    X_train_cur, X_test_cur = X_t[train_index], X_t[test_index]
    X_train_cur1, X_test_cur1 = X_t1[train_index], X_t1[test_index]
    y_train_cur, y_test_cur = y_train[train_index], y_train[test_index]
    structured_data_train_cur, structured_data_test_cur = (
        structured_data_train[train_index],
        structured_data_train[test_index],
    )

    X_train_train, X_val, y_train_train, y_val = train_test_split(
        X_train_cur, y_train_cur, test_size=0.25, random_state=100
    )
    structured_data_train_train, structured_data_val = train_test_split(
        structured_data_train_cur, test_size=0.25, random_state=100
    )

    bestscore = 0
    for neuro in neurons:
        for ne in n_filters_grid:
            for md in embed_size:
                print('\n-------------\n')
                print('Neurons:' + str(neuro))
                print(
                    'Num Filters:' + str(ne) + '   ' + 'Embed Size:' + str(md)
                )
                clf = build_model(neuro, ne, md)
                history = clf.fit(
                    [X_train_cur, X_train_cur1, structured_data_train_cur],
                    to_categorical(y_train_cur),
                    validation_data=(
                        [X_test_cur, X_test_cur1, structured_data_test_cur],
                        to_categorical(y_test_cur),
                    ),
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    callbacks=callbacks_list,
                )

                bestPerformingModel = load_model(
                    'CNN_best_model' + str(i) + '.h5'
                )

                loss, bscr_train = bestPerformingModel.evaluate(
                    [X_train_cur, X_train_cur1, structured_data_train_cur],
                    to_categorical(y_train_cur),
                )
                print(loss, bscr_train)
                loss, bscr_val = bestPerformingModel.evaluate(
                    [X_test_cur, X_test_cur1, structured_data_test_cur],
                    to_categorical(y_test_cur),
                )
                print(loss, bscr_val)
                loss, bscr = bestPerformingModel.evaluate(
                    [X_te, X_te1, structured_data_test], to_categorical(y_test)
                )
                print(loss, bscr)
                print('\n-------------\n')

    bestPerformingModel = load_model('CNN_best_model' + str(i) + '.h5')

    loss, bscr_train = bestPerformingModel.evaluate(
        [X_train_cur, X_train_cur1, structured_data_train_cur],
        to_categorical(y_train_cur),
    )
    print(loss, bscr_train)
    loss, bscr_val = bestPerformingModel.evaluate(
        [X_test_cur, X_test_cur1, structured_data_test_cur],
        to_categorical(y_test_cur),
    )
    print(loss, bscr_val)
    loss, bscr = bestPerformingModel.evaluate(
        [X_te, X_te1, structured_data_test], to_categorical(y_test)
    )
    print(loss, bscr)

    models.append(clf)

    avgsc_train = avgsc_train + bscr_train
    avgsc_val = avgsc_val + bscr_val
    avgsc = avgsc + bscr

    avgsc_train_lst.append(bscr_train)
    avgsc_val_lst.append(bscr_val)
    avgsc_lst.append(bscr)

    print('The training accuracy is:')
    print(bscr_train)
    print('The validation accuracy is:')
    print(bscr_val)
    print('The test accuracy is:')
    print(bscr)
    print('\n')
    i = i + 1


# %%


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %%
kf = KFold(n_splits=5)
avgsc_lst, avgsc_val_lst, avgsc_train_lst = [], [], []

i = 0
for train_index, test_index in kf.split(X_t):
    X_train_cur, X_test_cur = X_t[train_index], X_t[test_index]
    X_train_cur1, X_test_cur1 = X_t1[train_index], X_t1[test_index]
    y_train_cur, y_test_cur = y_train[train_index], y_train[test_index]
    print(len(X_train_cur), len(X_test_cur))
    print(len(y_train_cur), len(y_test_cur))
    structured_data_train_cur, structured_data_test_cur = (
        structured_data_train[train_index],
        structured_data_train[test_index],
    )
    #     print(len(structured_data_train_cur),len(structured_data_test_cur))
    print(len(X_te), len(y_test))

    bestPerformingModel = load_model('CNN_best_model' + str(i) + '.h5')

    loss, bscr_train = bestPerformingModel.evaluate(
        [X_train_cur, X_train_cur1, structured_data_train_cur],
        to_categorical(y_train_cur),
    )
    print(loss, bscr_train)
    loss, bscr_val = bestPerformingModel.evaluate(
        [X_test_cur, X_test_cur1, structured_data_test_cur],
        to_categorical(y_test_cur),
    )
    print(loss, bscr_val)
    loss, bscr = bestPerformingModel.evaluate(
        [X_te, X_te1, structured_data_test], to_categorical(y_test)
    )
    print(loss, bscr)

    avgsc_train_lst.append(bscr_train)
    avgsc_val_lst.append(bscr_val)
    avgsc_lst.append(bscr)
    print('\n')
    i = i + 1
print(avgsc_train_lst)
print(avgsc_val_lst)
print(avgsc_lst)


# %%
print(avgsc_train_lst)
print(avgsc_val_lst)
print(avgsc_lst)
print(np.mean(avgsc_train_lst))
print(np.mean(avgsc_val_lst))
print(np.mean(avgsc_lst))

y_pred = bestPerformingModel.predict([X_te, X_te1, structured_data_test])
y_pred1 = [np.argmax(i) for i in y_pred]
cm = confusion_matrix(y_test, y_pred1)
print('Confusion Matrix: Actual (Row) vs Predicted (Column)')
print(cm)
