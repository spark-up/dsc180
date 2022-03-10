import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import sys
from textwrap import indent
from time import time

import pandas as pd
from pyspark.sql import SparkSession

from runtime.experiments import spark_scale as SparkScalingExperiment
from runtime.measure import ExperimentLab


EXPERIMENTS = {'Spark-Scale': SparkScalingExperiment}
EXPERIMENT_ALIASES = {'Spark-Scale': 'SS'}
COLUMNS = {
    'prepare': float,
    'run': float,
    'iterations': int,
    'total': float,
}

CONSOLE_FORMAT = '''
---
results:
  name:     {name}
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
ap.add_argument(
    '--scale-factor',
    type=non_negative_int,
    default=6,
    help='The maximum scale factor.',
)

args = ap.parse_args()

spark = SparkSession.builder.appName('experiment').getOrCreate()
sc = spark.sparkContext
sc.setCheckpointDir('hdfs://10.11.12.207:9000/data/checkpoints')

sdf = spark.read.csv(
    'hdfs://10.11.12.207:9000/data/members.csv', header=True, inferSchema=True
)

name = 'spark-scaling'
output_file = f'/tmp/results-{int(time())}.yml'

for i in range(args.scale_factor):
    x = ExperimentLab(
        SparkScalingExperiment(spark, sdf), trials=args.trials, tests=args.tests
    )
    profiles = x.measure()
    df = pd.DataFrame(
        columns=tuple(COLUMNS.keys()),
    ).astype(COLUMNS)
    if profiles:
        df[['prepare', 'run', 'iterations']] = pd.DataFrame.from_records(profiles)  # type: ignore
    df['total'] = df.prepare + df.run

    iterations = df.iterations.sum()
    prepare = df.prepare.sum() / iterations
    run = df.run.sum() / iterations
    total = prepare + run

    result = dict(
        name=f'{name}-{args.scale_factor}',
        prepare=prepare,
        run=run,
        total=total,
    )
    with open(output_file, 'a') as output:
        for f in (output, sys.stdout):
            print(CONSOLE_FORMAT.format_map(result), file=f)
            print('  table: |', file=f)
            print(indent(df.to_string(), ' ' * 4), file=f)  # type: ignore

    if i < args.scale_factor - 1:
        sdf = sdf.union(sdf)
        sdf = sdf.checkpoint(True)
    else:
        # Last iteration
        pass
