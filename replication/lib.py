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


import re

import numpy as np
import pandas as pd
from keras.preprocessing import sequence as keras_seq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from .common import abs_limit_10000 as abs_limit
from .resources import (
    load_cnn,
    load_keras_name_tokenizer,
    load_keras_sample_tokenizer,
    load_logistic_regression,
    load_random_forest,
    load_sklearn_name_vectorizer,
    load_sklearn_sample_vectorizer,
    load_svm,
    load_test,
    load_train,
)

DEL_RE = re.compile(r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}')

DELIMITER_RE = re.compile(r'(,|;|\|)')

URL_RE = re.compile(
    r'(http|ftp|https):\/\/'
    r'([\w_-]+(?:(?:\.[\w_-]+)+))'
    r'([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
)

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b')

STOPWORDS = set(stopwords.words('english'))


def _summary_stats(col):
    pass


def summary_stats(df, keys):
    b_data = []
    for col in keys:
        nans = np.count_nonzero(pd.isnull(df[col]))
        dist_val = len(pd.unique(df[col].dropna()))
        total_val = len(df[col])
        mean = 0
        std_dev = 0
        var = 0
        min_val = 0
        max_val = 0
        if is_numeric_dtype(df[col]):
            mean = np.mean(df[col])

            if not pd.isnull(mean):
                std_dev = np.std(df[col])
                var = np.var(df[col])
                min_val = float(np.min(df[col]))
                max_val = float(np.max(df[col]))
        b_data.append(
            [total_val, nans, dist_val, mean, std_dev, min_val, max_val]
        )
    return b_data


def get_sample(df, keys):
    rand = []
    for name in keys:
        rand_sample = list(pd.unique(df[name]))
        rand_sample = rand_sample[:5]
        while len(rand_sample) < 5:
            rand_sample.append(
                list(pd.unique(df[name]))[
                    np.random.randint(len(list(pd.unique(df[name]))))
                ]
            )
        rand.append(rand_sample[:5])
    return rand


# summary_stat_result has a structure like [[Total_val, nans, dist_va, ...], ...].
def get_ratio_dist_val(summary_stat_result):
    ratio_dist_val = []
    for r in summary_stat_result:
        ratio_dist_val.append(r[2] * 100.0 / r[0])
    return ratio_dist_val


def get_ratio_nans(summary_stat_result):
    ratio_nans = []
    for r in summary_stat_result:
        ratio_nans.append(r[1] * 100.0 / r[0])
    return ratio_nans


def featurize_file(df):
    stats = []
    attribute_name = []
    sample = []
    i = 0

    ratio_dist_val = []
    ratio_nans = []

    keys = list(df.keys())

    attribute_name.extend(keys)
    summary_stat_result = summary_stats(df, keys)
    stats.extend(summary_stat_result)
    samples = get_sample(df, keys)
    sample.extend(samples)

    ratio_dist_val.extend(get_ratio_dist_val(summary_stat_result))
    ratio_nans.extend(get_ratio_nans(summary_stat_result))

    cols = [
        'Attribute_name',
        'total_vals',
        'num_nans',
        'num_of_dist_val',
        'mean',
        'std_dev',
        'min_val',
        'max_val',
        '%_dist_val',
        '%_nans',
        'sample_1',
        'sample_2',
        'sample_3',
        'sample_4',
        'sample_5',
    ]
    golden_data = pd.DataFrame(columns=cols)

    for i in range(len(attribute_name)):
        # print(attribute_name[i])
        val_append = []
        val_append.append(attribute_name[i])
        val_append.extend(stats[i])

        val_append.append(ratio_dist_val[i])
        val_append.append(ratio_nans[i])
        val_append.extend(sample[i])

        golden_data.loc[i] = val_append
    #     print(golden_data)

    curdf = golden_data

    for row in curdf.itertuples():

        # print(row[11])
        is_list = False
        curlst = [row[11], row[12], row[13], row[14], row[15]]

        delim_cnt, url_cnt, email_cnt, date_cnt = 0, 0, 0, 0
        chars_totals, word_totals, stopwords, whitespaces, delims_count = (
            [],
            [],
            [],
            [],
            [],
        )

        for value in curlst:
            word_totals.append(len(str(value).split(' ')))
            chars_totals.append(len(str(value)))
            whitespaces.append(str(value).count(' '))

            if DEL_RE.match(str(value)):
                delim_cnt += 1
            if URL_RE.match(str(value)):
                url_cnt += 1
            if EMAIL_RE.match(str(value)):
                email_cnt += 1

            delims_count.append(len(DELIMITER_RE.findall(str(value))))

            tokenized = word_tokenize(str(value))
            # print(tokenized)
            stopwords.append(len([w for w in tokenized if w in STOPWORDS]))

            try:
                _ = pd.Timestamp(value)
                date_cnt += 1
            except (ValueError, TypeError):
                date_cnt += 0

        # print(delim_cnt,url_cnt,email_cnt)
        if delim_cnt > 2:
            curdf.at[row.Index, 'has_delimiters'] = True
        else:
            curdf.at[row.Index, 'has_delimiters'] = False

        if url_cnt > 2:
            curdf.at[row.Index, 'has_url'] = True
        else:
            curdf.at[row.Index, 'has_url'] = False

        if email_cnt > 2:
            curdf.at[row.Index, 'has_email'] = True
        else:
            curdf.at[row.Index, 'has_email'] = False

        if date_cnt > 2:
            curdf.at[row.Index, 'has_date'] = True
        else:
            curdf.at[row.Index, 'has_date'] = False

        curdf.at[row.Index, 'mean_word_count'] = np.mean(word_totals)
        curdf.at[row.Index, 'std_dev_word_count'] = np.std(word_totals)

        curdf.at[row.Index, 'mean_stopword_total'] = np.mean(stopwords)
        curdf.at[row.Index, 'stdev_stopword_total'] = np.std(stopwords)

        curdf.at[row.Index, 'mean_char_count'] = np.mean(chars_totals)
        curdf.at[row.Index, 'stdev_char_count'] = np.std(chars_totals)

        curdf.at[row.Index, 'mean_whitespace_count'] = np.mean(whitespaces)
        curdf.at[row.Index, 'stdev_whitespace_count'] = np.std(whitespaces)

        curdf.at[row.Index, 'mean_delim_count'] = np.mean(whitespaces)
        curdf.at[row.Index, 'stdev_delim_count'] = np.std(whitespaces)

        if (
            curdf.at[row.Index, 'has_delimiters']
            and curdf.at[row.Index, 'mean_char_count'] < 100
        ):
            curdf.at[row.Index, 'is_list'] = True
        else:
            curdf.at[row.Index, 'is_list'] = False

        if curdf.at[row.Index, 'mean_word_count'] > 10:
            curdf.at[row.Index, 'is_long_sentence'] = True
        else:
            curdf.at[row.Index, 'is_long_sentence'] = False

        # print(np.mean(stopwords))

        # print('\n\n\n')

    golden_data = curdf

    return golden_data


def extract_features(df, use_samples=False):
    df_orig = df.copy()
    df = df_orig[
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
    df = df.reset_index(drop=True).fillna(0)

    arr = df_orig['Attribute_name'].values.astype(str)

    X = load_sklearn_name_vectorizer().transform(arr)
    df_attr = pd.DataFrame(X.toarray())

    if not use_samples:
        return pd.concat([df, df_attr], axis=1, sort=False)

    sample_1_values = df_orig['sample_1'].values.astype(str)
    sample_2_values = df_orig['sample_2'].values.astype(str)
    X1 = load_sklearn_sample_vectorizer().transform(sample_1_values)
    X2 = load_sklearn_sample_vectorizer().transform(sample_2_values)

    sample_1_df = pd.DataFrame(X1.toarray())
    sample_2_df = pd.DataFrame(X2.toarray())
    return pd.concat(
        [df, df_attr, sample_1_df, sample_2_df], axis=1, sort=False
    )


def predict_rf(df: pd.DataFrame):
    pred = load_random_forest().predict(df).tolist()
    return pred


def process_statistics(df: pd.DataFrame):
    df = df[
        [
            '%_dist_val',
            '%_nans',
            'max_val',
            'mean',
            'mean_char_count',
            'mean_delim_count',
            'mean_stopword_total',
            'mean_whitespace_count',
            'mean_word_count',
            'min_val',
            'num_nans',
            'num_of_dist_val',
            'std_dev',
            'std_dev_word_count',
            'stdev_char_count',
            'stdev_delim_count',
            'stdev_stopword_total',
            'stdev_whitespace_count',
            'total_vals',
        ]
    ].rename(
        columns={
            '%_nans': 'scaled_perc_nans',
            'max_val': 'scaled_max',
            'mean': 'scaled_mean',
            'mean_char_count': 'scaled_mean_char_count',
            'mean_delim_count': 'scaled_mean_delim_count',
            'mean_stopword_total': 'scaled_mean_stopword_total',
            'mean_whitespace_count': 'scaled_mean_whitespace_count',
            'mean_word_count': 'scaled_mean_token_count',
            'min_val': 'scaled_min',
            'std_dev': 'scaled_std_dev',
            'std_dev_word_count': 'scaled_std_dev_token_count',
            'stdev_char_count': 'scaled_stdev_char_count',
            'stdev_delim_count': 'scaled_stdev_delim_count',
            'stdev_stopword_total': 'scaled_stdev_stopword_total',
            'stdev_whitespace_count': 'scaled_stdev_whitespace_count',
        }
    )

    df = df.reset_index(drop=True)
    df = df.fillna(0)

    cols_to_abs_limit = [
        'num_nans',
        'num_of_dist_val',
        'scaled_max',
        'scaled_mean',
        'scaled_min',
        'scaled_std_dev',
        'total_vals',
    ]
    for col in cols_to_abs_limit:
        df[col] = df[col].apply(abs_limit)

    cols_to_normalize = [
        'total_vals',
        'num_nans',
        'num_of_dist_val',
        'scaled_mean',
        'scaled_std_dev',
        'scaled_min',
        'scaled_max',
    ]
    X = df[cols_to_normalize].values
    X = np.nan_to_num(X)
    X_scaled = StandardScaler().fit_transform(X)
    df[cols_to_normalize] = pd.DataFrame(
        X_scaled,
        columns=cols_to_normalize,
        index=df.index,
    )

    return df


def predict_cnn(df):
    cnn = load_cnn()

    featurized = featurize_file(df)
    structured_data_test = process_statistics(featurized)

    tokenizer = load_keras_name_tokenizer()
    tokenizer_sample = load_keras_sample_tokenizer()

    names = featurized['Attribute_name'].values.astype(str)
    samples = featurized['sample_1'].values.astype(str)

    X_names = keras_seq.pad_sequences(
        tokenizer.texts_to_sequences(names),
        maxlen=256,
    )
    X_samples = keras_seq.pad_sequences(
        tokenizer_sample.texts_to_sequences(samples),
        maxlen=256,
    )

    y_pred = cnn.predict([X_names, X_samples, structured_data_test])
    y_CNN = [np.argmax(i) for i in y_pred]
    return y_CNN
