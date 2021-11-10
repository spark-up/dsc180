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

import editdistance
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler

# %%
xtrain = pd.read_csv('../../Benchmark-Labeled-Data/data_train.csv')
xtest = pd.read_csv('../../Benchmark-Labeled-Data/data_test.csv')


xtrain = xtrain.sample(frac=1, random_state=100).reset_index(drop=True)
print(len(xtrain))

y_train = xtrain.loc[:, ['y_act']]
y_test = xtest.loc[:, ['y_act']]

atr_train = xtrain.loc[:, ['Attribute_name']]
atr_test = xtest.loc[:, ['Attribute_name']]


# %%
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

# %%
def func1(data, y):

    data1 = data[
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
            'has_delimiters',
            'has_url',
            'has_email',
            'has_date',
            'mean_word_count',
            'std_dev_word_count',
            'mean_stopword_total',
            'stdev_stopword_total',
            'mean_char_count',
            'stdev_char_count',
            'mean_whitespace_count',
            'stdev_whitespace_count',
            'mean_delim_count',
            'stdev_delim_count',
            'is_list',
            'is_long_sentence',
        ]
    ]
    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)

    def abs_limit(x):
        if abs(x) > 1000:
            return 1000 * np.sign(x)
        return x

    column_names_to_normalize = [
        'total_vals',
        'num_nans',
        '%_nans',
        'num_of_dist_val',
        '%_dist_val',
        'mean',
        'std_dev',
        'min_val',
        'max_val',
        'has_delimiters',
        'has_url',
        'has_email',
        'has_date',
        'mean_word_count',
        'std_dev_word_count',
        'mean_stopword_total',
        'stdev_stopword_total',
        'mean_char_count',
        'stdev_char_count',
        'mean_whitespace_count',
        'stdev_whitespace_count',
        'mean_delim_count',
        'stdev_delim_count',
        'is_list',
        'is_long_sentence',
    ]

    for col in column_names_to_normalize:
        data1[col] = data1[col].apply(abs_limit)

    print(column_names_to_normalize)
    x = data1[column_names_to_normalize].values
    x = np.nan_to_num(x)
    x_scaled = StandardScaler().fit_transform(x)
    df_temp = pd.DataFrame(
        x_scaled, columns=column_names_to_normalize, index=data1.index
    )
    data1[column_names_to_normalize] = df_temp

    y.y_act = y.y_act.astype(float)

    print(f"> Data mean: {data1.mean()}\n")
    print(f"> Data median: {data1.median()}\n")
    print(f"> Data stdev: {data1.std()}")

    return data1


X_train = func1(xtrain, y_train)
X_test = func1(xtest, y_test)

# %%
X_train.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
atr_train.reset_index(inplace=True, drop=True)
atr_test.reset_index(inplace=True, drop=True)


X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values

atr_train = atr_train.values
atr_test = atr_test.values

# %%
k = 5
kf = KFold(n_splits=k)
avg_train_acc, avg_test_acc = 0, 0

avgsc_lst, avgsc_train_lst, avgsc_hld_lst = [], [], []
avgsc, avgsc_train, avgsc_hld = 0, 0, 0

acc_val_lst, acc_test_lst = [], []

for train_index, test_index in kf.split(X_train):

    print(train_index)
    print()
    print(test_index)
    X_train_cur, X_test_cur = X_train[train_index], X_train[test_index]
    y_train_cur, y_test_cur = y_train[train_index], y_train[test_index]

    print(y_train_cur)
    atr_train_train, atr_val = atr_train[train_index], atr_train[test_index]

    X_train_train = X_train_cur
    X_val = X_test_cur

    y_train_train = y_train_cur
    y_val = y_test_cur

    Matrix = [[0 for x in range(len(X_train_train))] for y in range(len(X_val))]
    dist_euc = DistanceMetric.get_metric('euclidean')

    np_X_train = np.asmatrix(X_train_train)
    np_X_test = np.asmatrix(X_val)

    for i in range(len(X_val)):
        if i % 100 == 0:
            print(i)
        a = np_X_test[i]
        for j in range(len(X_train_train)):
            b = np_X_train[j]
            dist = np.linalg.norm(a - b)
            Matrix[i][j] = dist

    Matrix_ed = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_val))
    ]

    for i in range(len(X_val)):
        if i % 100 == 0:
            print(i)
        a = atr_val[i]
        #         print(a)
        for j in range(len(X_train_train)):
            b = atr_train_train[j]
            #         print(b)
            dist = editdistance.eval(str(a), str(b))
            Matrix_ed[i][j] = dist

    Matrix_net = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_val))
    ]
    alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    best_sc, best_alpha, best_neighbr = 0, 0, 0
    for alp in alpha:
        for i in range(len(Matrix)):
            for j in range(len(Matrix[i])):
                Matrix_net[i][j] = alp * Matrix[i][j] + Matrix_ed[i][j]

        for neighbr in range(1, 11):
            y_pred = []
            for i in range(len(X_val)):
                #     print('---')
                #         print(Matrix_net[i])
                dist = np.argsort(Matrix_net[i])[:neighbr]
                ys = []
                for x in dist:
                    ys.append(y_train_train[x])
                ho = stats.mode(ys)
                pred = ho[0][0]
                y_pred.append(pred)
            acc = accuracy_score(y_val, y_pred)
            print(str(neighbr) + '--->' + str(alp) + '--->' + str(acc))
            if acc > best_sc:
                best_sc = acc
                best_alpha = alp
                best_neighbr = neighbr

    print(best_sc, best_alpha, best_neighbr)

    ##################################
    X_train_train = X_train
    y_train_train = y_train
    atr_train_train = atr_train

    Matrix = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_test))
    ]
    dist_euc = DistanceMetric.get_metric('euclidean')

    np_X_train = np.asmatrix(X_train_train)
    np_X_test = np.asmatrix(X_test)

    for i in range(len(X_test)):
        if i % 100 == 0:
            print(i)
        a = np_X_test[i]
        for j in range(len(X_train_train)):
            b = np_X_train[j]
            dist = np.linalg.norm(a - b)
            Matrix[i][j] = dist

    Matrix_ed = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_test))
    ]

    for i in range(len(X_test)):
        if i % 100 == 0:
            print(i)
        a = atr_test[i]
        #         print(a)
        for j in range(len(X_train_train)):
            b = atr_train_train[j]
            #         print(b)
            dist = editdistance.eval(str(a), str(b))
            Matrix_ed[i][j] = dist

    #################################

    Matrix_net = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_test))
    ]
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            Matrix_net[i][j] = best_alpha * Matrix[i][j] + Matrix_ed[i][j]

    y_pred = []
    for i in range(len(X_test)):
        dist = np.argsort(Matrix_net[i])[:best_neighbr]
        ys = []
        for x in dist:
            ys.append(y_train_train[x])
        ho = stats.mode(ys)
        pred = ho[0][0]
        y_pred.append(pred)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

    acc_val_lst.append(best_sc)
    acc_test_lst.append(acc)

    print(acc_val_lst)
    print(acc_test_lst)

    print('\n\n\n')


# %%
print(acc_val_lst)
print(acc_test_lst)
print(np.mean(acc_val_lst))
# print(np.mean(acc_test_lst))
bestValId = np.argmax(acc_val_lst)
print(acc_test_lst[bestValId])


# %%
