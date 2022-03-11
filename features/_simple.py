from typing import Final

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from .util import ColumnFn, count_nan, is_struct_field_numeric

SIMPLE_FEATURES: dict[str, ColumnFn] = {
    'count': F.count,
    'distinct': F.count_distinct,
    'distinct_percent': lambda c: 100 * F.count_distinct(c) / F.count(c),
}

SIMPLE_NUMERIC_FEATURES: dict[str, ColumnFn] = {
    'nans': count_nan,
    'nans_percent': lambda c: 100 * count_nan(c) / F.count(c),
    'mean': F.mean,
    'std': F.stddev,
    'min': F.min,
    'max': F.max,
}

_LONG_FEATURES = [
    'count',
    'distinct',
    'nans',
]
_DOUBLE_FEATURES = [
    'distinct_percent',
    'nans_percent',
    'mean',
    'std',
    'min',
    'max',
]

_FEATURES = list(SIMPLE_FEATURES.keys()) + list(SIMPLE_NUMERIC_FEATURES.keys())

# This map's iteration order determines output order as per spec
_LEGACY_NAME_MAP: dict[str, str] = {
    'name': 'Attribute_name',
    'count': 'total_vals',
    'nans': 'num_nans',
    'nans_percent': '%_nans',
    'distinct': 'num_of_dist_val',
    'distinct_percent': '%_dist_val',
    'mean': 'mean',
    'std': 'std_dev',
    'min': 'min_val',
    'max': 'max_val',
}

N_SAMPLES: Final = 5


def _create_schema() -> StructType:
    fields = [StructField('name', StringType(), False)]
    fields.extend(StructField(k, LongType(), False) for k in _LONG_FEATURES)
    fields.extend(StructField(k, DoubleType(), False) for k in _DOUBLE_FEATURES)
    return StructType(fields)


def _get_dtypes() -> dict:
    dtypes = {'name': 'string'}
    dtypes.update({k: 'Int64' for k in _LONG_FEATURES})
    dtypes.update({k: 'float64' for k in _DOUBLE_FEATURES})
    return dtypes


def simple_features_impl(
    df: SparkDataFrame,
    /,
    *,
    use_legacy_names=False,
    explain=False,
) -> SparkDataFrame:
    cols = df.columns
    s_features = SIMPLE_FEATURES
    sn_features = SIMPLE_NUMERIC_FEATURES

    if use_legacy_names:
        s_features = {_LEGACY_NAME_MAP[k]: v for k, v in s_features.items()}
        sn_features = {_LEGACY_NAME_MAP[k]: v for k, v in sn_features.items()}

    simple_aggs = [
        fn(col(c)).alias(f'{c}::{name}')
        for c in cols
        for name, fn in s_features.items()
    ]

    numeric_aggs = [
        (fn(col(c)) if is_struct_field_numeric(df.schema[c]) else lit(0)).alias(
            f'{c}::{name}'
        )
        for c in cols
        for name, fn in sn_features.items()
    ]

    agg_df = df.agg(*simple_aggs, *numeric_aggs)

    if explain:
        print('-' * 20)
        print('[EXPLAIN] simple_features_impl')
        print('-' * 20)
        agg_df.explain(mode='cost')
        agg_df.explain(mode='formatted')

    return agg_df


def simple_features_melt(
    df: SparkDataFrame,
    /,
    *,
    _cols: list[str] | None = None,
    use_legacy_names=False,
    explain=False,
) -> SparkDataFrame:
    if not _cols:
        _cols = list({c.rsplit('::')[0] for c in df.columns})

    features = _FEATURES
    if use_legacy_names:
        features = [_LEGACY_NAME_MAP.get(f, f) for f in features]

    result = df.sql_ctx.createDataFrame([], schema=_create_schema())
    for c in _cols:
        exprs = [lit(c)]
        exprs.extend(col(f'{c}::{f}') for f in features)
        tmp = df.select(exprs)
        result = result.union(tmp)

    if use_legacy_names:
        expr = [col(k).alias(v) for k, v in _LEGACY_NAME_MAP.items()]
        result = result.select(expr)

    if explain:
        print('-' * 20)
        print('[EXPLAIN] simple_features_melt')
        print('-' * 20)
        result.explain(mode='cost')
        result.explain(mode='formatted')
    return result


def _split(s: str) -> tuple[str, str]:
    return tuple(s.rsplit('::', 1))


def simple_features_melt_in_pandas(
    df: pd.DataFrame,
    /,
    *,
    use_legacy_names=False,
) -> pd.DataFrame:
    dtypes = _get_dtypes()
    features = _FEATURES
    name = 'name'
    if use_legacy_names:
        features = [_LEGACY_NAME_MAP.get(f, f) for f in features]
        dtypes = {_LEGACY_NAME_MAP.get(k, k): v for k, v in dtypes.items()}
        name = _LEGACY_NAME_MAP['name']

    df = df.copy()
    df.columns = df.columns.map(_split)
    result = (
        df.stack()
        .reset_index(0, drop=True)
        .rename_axis(columns=name)
        .T.reset_index()
    )
    if use_legacy_names:
        result = result[list(_LEGACY_NAME_MAP.values())]

    result = result.astype(dtypes)

    return result
