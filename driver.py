import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from textwrap import indent

import pandas as pd

from runtime.experiments import spark_scale
from runtime.measure import ExperimentLab

from pyspark.sql import SparkSession, DataFrame
from pyspark.context import SparkContext
import seaborn as sns 

spark = SparkSession.builder.appName('experiment').getOrCreate() 
sc = spark.sparkContext
sc.setCheckpointDir('hdfs://10.11.11.35:9000/data/checkpoints')

def load_scale():
    return spark.read.csv('hdfs://10.11.11.35:9000/data/members.csv', header = True, inferSchema = True)

EXPERIMENTS = {
    'Spark-Scale': spark_scale
}
EXPERIMENT_ALIASES = {
    'Spark-Scale': 'SS'
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

sdf = load_scale()


run_time = []
for name, Klass in EXPERIMENTS.items():
    for i in range(8):
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
            run_time.append(run)
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


print(run_time)

x_val = [1.25, 2.51, 5.03, 10.07, 20.14, 40.29, 80.59, 161.19]
graph = sns.lineplot(x=x_val, y =run_time)
graph.set_xlabel("Size (gbs)", fontsize = 20)
graph.set_ylabel("time (seconds)", fontsize = 20)
graph.figure.savefig('results.png')
