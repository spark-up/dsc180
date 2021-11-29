import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from ..resources import force_load
from .experiments import Cnn, Experiment, Logistic, RandomForest, Svm
from .measure import ExperimentLab

EXPERIMENTS = {
    'cnn': Cnn,
    'logistic': Logistic,
    'noop': Experiment,
    'random-forest': RandomForest,
    'svm': Svm,
}


force_load()

for Klass in EXPERIMENTS.values():
    x = ExperimentLab(Klass(), iterations=1)
    print(x.experiment.name, x.measure())
