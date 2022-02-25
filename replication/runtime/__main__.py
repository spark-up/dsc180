import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import argparse
from textwrap import indent

import pandas as pd

from ..lazy_resources import force_load
from .experiments import spark_scale
from .measure import ExperimentLab

from ..lazy_resources import load_scale

from pyspark.sql import SparkSession, DataFrame
from pyspark.context import SparkContext


spark = SparkSession.builder.appName('experiment').getOrCreate() 
sc = spark.sparkContext
sc.setCheckpointDir('checkpoint')

EXPERIMENTS = {
  #  'cnn': Cnn,
  #  'logistic': Logistic,
    'Spark-Scale': spark_scale,
   # 'noop': Experiment,
   # 'random-forest': RandomForest,
   # 'svm': Svm,
}
EXPERIMENT_ALIASES = {
  #  'logistic': 'lr',
    'Spark-Scale': 'SS',
#    'random-forest': 'rf',
}
COLUMNS = {
    'prepare': float,
    'run': float,
    'iterations': int,
    'total': float,
}

CONSOLE_FORMAT = '''
---
results:
  prepare:  {prepare:.3e}
  run:      {run:.3e}
  total:    {total:.3e}
'''


def non_negative_int(s: str) -> int:
    i = int(s)
    if i < 0:
        raise ValueError('Input {} must be non-negative!'.format(i))
    return i


ap = argparse.ArgumentParser(__package__)
ap.add_argument(
    '-n',
    '--trials',
    type=non_negative_int,
    default=1,
    help='The number of independent trials to measure.',
)
ap.add_argument(
    '-t',
    '--tests',
    type=non_negative_int,
    default=1,
    help='The number of test executions per trial.',
)
ap.add_argument(
    '-f', '--format', choices=('console', 'latex'), default='console'
)

args = ap.parse_args()

force_load()

# results = {}
# raw_results = {}

sdf = load_scale()

# for i in range(10):
#     sdf = sdf.union(sdf)

for name, Klass in EXPERIMENTS.items():
    for i in range(10):
        results = {}
        raw_results = {}

        x = ExperimentLab(Klass(spark, sdf), trials=args.trials, tests=args.tests)
        profiles = x.measure()
        df = pd.DataFrame(
            columns=tuple(COLUMNS.keys()),
        ).astype(COLUMNS)
        if profiles:
            df[['prepare', 'run', 'iterations']] = pd.DataFrame.from_records(profiles)  # type: ignore
        df['total'] = df.prepare + df.run

        raw_results[name] = df
        for name, df in raw_results.items():
            iterations = df.iterations.sum()
            prepare = df.prepare.sum() / iterations
            run = df.run.sum() / iterations
            total = prepare + run

            results[name] = dict(
                # name=name,
                prepare=prepare,
                run=run,
                total=total,
            )
        if args.format == 'console':
            for result in results.values():
                print(CONSOLE_FORMAT.format_map(result))
                print('  table: |')
                print(indent(df.to_string(), ' ' * 4))  # type: ignore
        elif args.format == 'latex':
            df = pd.DataFrame.from_records(results.values()).set_index('name')
        sdf = sdf.union(sdf)
        sdf = sdf.checkpoint(True)

#print(result)