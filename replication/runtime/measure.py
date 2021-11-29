from __future__ import annotations

import sys
from cProfile import Profile
from dataclasses import dataclass, field
from time import perf_counter
from typing import NamedTuple

from .experiments import Experiment


class TrialResults(NamedTuple):
    prepare: float
    run: float

    @property
    def total(self) -> float:
        return self.prepare + self.run  # type: ignore

    def __str__(self):
        return f'TrialResults(prepare={self.prepare}, run={self.run})'


_SENTINEL = object()


@dataclass
class ExperimentLab:
    experiment: Experiment
    iterations: int = 10
    trials: int = 1
    profile: Profile = field(default_factory=lambda: Profile(perf_counter))

    def measure_trial(self):
        experiment = self.experiment
        iterations = self.iterations

        experiment.setup()
        with self.profile as pr:
            for _ in range(iterations):
                experiment.prepare()
                experiment.run()

        # HACK: Time to use undocumented hacks
        raw_stats = pr.getstats()  # type: ignore
        # Stats(pr).print_stats(10)
        # breakpoint()
        prepare_src = sys.modules[self.experiment.prepare.__module__].__file__
        run_src = sys.modules[self.experiment.run.__module__].__file__
        stats = {
            entry[0].co_name: entry.totaltime
            for entry in raw_stats
            if getattr(entry[0], 'co_filename', _SENTINEL)
            in (prepare_src, run_src)
        }
        return TrialResults(stats['prepare'], stats['run'])  # type: ignore

    def measure(self):
        results = [self.measure_trial() for _ in range(self.iterations)]
        return results
