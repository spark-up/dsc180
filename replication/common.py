from pathlib import Path
from typing import Any, Callable, List

import numba
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
ZOO_RESOURCES = ROOT / 'data' / 'zoo-resources'
ZOO_BENCHMARK_DATA = ROOT / 'data' / 'zoo-benchmark'

DICT_LABELS = {
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


def load_train():
    return pd.read_csv(ZOO_BENCHMARK_DATA / 'data_train.csv')


def load_test():
    return pd.read_csv(ZOO_BENCHMARK_DATA / 'data_test.csv')


@numba.vectorize
def abs_limit_1000(x):
    if x > 1000:
        return 1000 * np.sign(x)
    return x


@numba.vectorize
def abs_limit_10000(x):
    if x > 10000:
        return 10000 * np.sign(x)
    return x


def to_string_list(it: Any) -> List[str]:
    return list(map(str, it))


def prepare_data(
    df: pd.DataFrame,
    y: pd.DataFrame,
    *,
    normalize,
    abs_limit: Callable = lambda x: x,
):
    df = df[
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
    if normalize:
        df = normalize_data(df, abs_limit=abs_limit)

    y['y_act'] = y['y_act'].map(DICT_LABELS).astype(float)

    # print(f"> Data mean: {df.mean()}\n")
    # print(f"> Data median: {df.median()}\n")
    # print(f"> Data stdev: {df.std()}")

    return df, y


def normalize_data(df: pd.DataFrame, abs_limit: Callable):
    df = df.copy()

    cols_to_normalize = [
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

    for col in cols_to_normalize:
        df[col] = df[col].apply(abs_limit)

    X = df[cols_to_normalize].values
    X = np.nan_to_num(X)
    X_scaled = StandardScaler().fit_transform(X)
    df_tmp = pd.DataFrame(X_scaled, columns=cols_to_normalize, index=df.index)
    df[cols_to_normalize] = df_tmp
    return df
