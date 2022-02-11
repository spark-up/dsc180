from typing import cast

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit

from ._sample import sample_features_from_values, sample_with_select_distinct
from ._simple import simple_features_impl, simple_features_melt


def extract_features(df: SparkDataFrame) -> pd.DataFrame:
    persist_df = df.persist()
    simple_sdf = simple_features_melt(simple_features_impl(persist_df))
    sample_values_sdf = sample_with_select_distinct(persist_df, 5)
    simple_df = cast(pd.DataFrame, simple_sdf.toPandas())
    sample_values_df = cast(pd.DataFrame, sample_values_sdf.toPandas())
    sample_df = sample_features_from_values(sample_values_df)
    persist_df.unpersist(False)
    return pd.concat([simple_df, sample_df], axis='columns')
