from typing import Any, Callable, List, TypeVar, cast

import numba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

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

_Numeric = TypeVar('_Numeric', np.ndarray, pd.Series, int, float)


def abs_limit_1000(x: _Numeric) -> _Numeric:
    if isinstance(x, (pd.Series, np.ndarray)):
        x = x.copy()
        x[x > 1000] = 1000 * np.sign(x[x > 1000])  # type: ignore
    elif x > 1000:
        x = 1000 * np.sign(x)  # type: ignore
    return x


def abs_limit_10000(x: _Numeric) -> _Numeric:
    if isinstance(x, (pd.Series, np.ndarray)):
        x = x.copy()
        x[x > 10000] = 10000 * np.sign(x[x > 10000])  # type: ignore
    elif x > 10000:
        x = 10000 * np.sign(x)  # type: ignore
    return x


def to_string_list(it: Any) -> List[str]:
    return list(map(str, it))


def process_stats(
    df: pd.DataFrame,
    *,
    normalize,
    abs_limit: Callable = lambda x: x,
) -> pd.DataFrame:
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

    return df


def process_targets(y: pd.DataFrame) -> pd.DataFrame:
    y['y_act'] = y['y_act'].map(DICT_LABELS).astype(float)
    return y


def normalize_data(df: pd.DataFrame, abs_limit: Callable):
    df = cast(pd.DataFrame, df.apply(abs_limit))

    X = np.nan_to_num(df.values)
    X_scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(X_scaled, columns=df.columns, index=df.index)


def create_vectorizer() -> CountVectorizer:
    return CountVectorizer(ngram_range=(2, 2), analyzer='char')


def extract_features(
    df: pd.DataFrame,
    df_stats: pd.DataFrame,
    /,
    *,
    name_vectorizer: CountVectorizer,
    fit=False,
) -> pd.DataFrame:
    """
    Create a final featurized DataFrame with statistics and vectorized strings.

    Args:
        df: The dataframe to vectorize
        df_stats: The statistical features to use
        vectorizer_name:
    """
    df = df.copy()

    names = to_string_list(df['Attribute_name'].values)

    # list_sample_1 = to_string_list(df['sample_1'].values)
    # list_sample_2 = to_string_list(df['sample_2'].values)
    # list_sample_3 = to_string_list(df['sample_3'].values)

    if fit:
        X = name_vectorizer.fit_transform(names)
        # X1 = vectorizer_sample.fit_transform(list_sample_1)
        # X2 = vectorizer_sample.transform(list_sample_2)

    else:
        X = name_vectorizer.transform(names)
        # X1 = vectorizer_sample.transform(list_sample_1)
        # X2 = vectorizer_sample.transform(list_sample_2)

    attr_df = pd.DataFrame(X.toarray())
    # sample1_df = pd.DataFrame(X1.toarray())
    # sample2_df = pd.DataFrame(X2.toarray())

    # if use_sample_1:
    #     out = sample1_df
    # if use_sample_2:
    #     out = sample2_df

    return pd.concat([df_stats, attr_df], axis=1, sort=False)
