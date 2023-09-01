import logging
import pprint
import random
import shutil
import sys
from functools import partial
from pathlib import Path
from random import Random
from typing import Iterable, Optional, Sequence, Any
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdpy import tools, containers, algorithms
from qdpy.algorithms import Evolution, QDAlgorithmLike
from qdpy.containers import Container
from qdpy.phenotype import Individual as QDPyIndividual, IndividualLike, Fitness as QDPyFitness, \
    Features as QDPyFeatures
from qdpy.plots import plot_evals, plot_iterations
import dataclasses
import json
import logging
import os
import pprint
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, NamedTuple, Callable, Optional

from colorama import Fore, Style
from abrain.core.genome import GIDManager
from robot.genome import RVGenome


def normalize_run_parameters(options: NamedTuple):
    if options.id is None:
        options.id = int(time.strftime('%m%d%H%M'))
        logging.info(f"Generated run id: {options.id}")

    if options.seed is None:
        try:
            options.seed = int(options.id)
        except ValueError:
            options.seed = round(1000 * time.time())
        logging.info(f"Deduced seed: {options.seed}")

    # Define the run folder
    folder_name = options.id
    if not isinstance(folder_name, str):
        folder_name = f"run{options.id}"
    options.run_folder = os.path.normpath(f"{options.base_folder}/{folder_name}/")
    logging.info(f"Run folder: {options.run_folder}")

    # Check the thread parameter
    import platform
    if platform.system() == "Linux":
        options.threads = max(1, min(options.threads, len(os.sched_getaffinity(0))))
    elif platform.system() == "Windows":
        import psutil
        options.threads = max(1, min(options.threads, psutil.cpu_count(logical=False)))
    else:
        print("Unknown operating system")
   
    logging.info(f"Parallel: {options.threads}")

    if options.verbosity >= 0:
        raw_dict = {k: v for k, v in options.__dict__.items() if not k.startswith('_')}
        logging.info(f"Post-processed command line arguments:\n{pprint.pformat(raw_dict)}")

@dataclass
class EvaluationResult:
    DataCollection = Dict[str, float]
    fitnesses: DataCollection = field(default_factory=dict)
    descriptors: DataCollection = field(default_factory=dict)
    #stats: DataCollection = field(default_factory=dict)

@dataclass
class Individual:
    DataCollection = EvaluationResult.DataCollection
    genome: RVGenome
    fitnesses: DataCollection = field(default_factory=dict)
    descriptors: DataCollection = field(default_factory=dict)
    #stats: DataCollection = field(default_factory=dict)
    eval_time: float = 0

    def id(self):
        return self.genome.id()

    def __eq__(self, other):
        assert self.id() is not None
        assert other.id() is not None
        return self.id() == other.id()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"{{id={self.id()}, fitness={self.fitness}, features={self.descriptors}}}"

    def update(self, r: EvaluationResult):
        self.fitnesses = r.fitnesses
        self.descriptors = r.descriptors
        #self.stats = r.stats

    def evaluation_result(self) -> EvaluationResult:
        return EvaluationResult(
            fitnesses=self.fitnesses,
            descriptors=self.descriptors,
            #stats=self.stats,
        )

    def to_json(self):
        dct = dataclasses.asdict(self)
        dct["genome"] = self.genome.to_json()
        return dct

    def to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f)

    @classmethod
    def from_json(cls, data):
        data.pop('id', None)
        data.pop('parents', None)
        ind = cls(**data)
        ind.genome = RVGenome.from_json(data['genome'])
        return ind

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            return cls.from_json(json.load(f))

class QDIndividual(Individual, QDPyIndividual):
    def __init__(self, genome: RVGenome, **kwargs):
        Individual.__init__(self, genome=genome, **kwargs)
        QDPyIndividual.__init__(self)
        assert self.id() is not None
        self.name = str(self.id())

    @property
    def fitness(self):
        return QDPyFitness(list(self.fitnesses.values()),
                           [1 for _ in self.fitnesses])

    @fitness.setter
    def fitness(self, _): pass

    @property
    def features(self):
        return QDPyFeatures(list(self.descriptors.values()))

    @features.setter
    def features(self, _): pass


class Algorithm(Evolution):
    def __init__(self, container: Container, options, labels, **kwargs):
        # Manage run id, seed, data folder...
        normalize_run_parameters(options)
        name = options.id

        self.rng = Random(options.seed)
        random.seed(options.seed)
        np.random.seed(options.seed % (2**32-1))

        self.id_manager = GIDManager()
        self.genealogical_info=[]
        self.stats = pd.DataFrame(columns=['Avg', 'Std', 'Max', 'QDs'])
        self.init_pop=[]
        

        def select(grid):
            #return self.rng.choice(grid)
            k = min(len(grid), options.tournament)
            candidates = self.rng.sample(grid.items, k)
            candidate_cells = [grid.index_grid(c.features) for c in candidates]
            curiosity = [grid.curiosity[c] for c in candidate_cells]
            if all([c == 0 for c in curiosity]):
                cell = self.rng.choice(candidate_cells)
            else:
                cell = candidate_cells[np.argmax(curiosity)]
            selection = candidates[candidate_cells.index(cell)]
            return selection

        def init(_):
            genome = RVGenome.random(self.rng, self.id_manager)
            for _ in range(options.initial_mutations):
                genome.mutate(self.rng)
                
            #print("genome.id()",genome.id())
            self.init_pop.append(QDIndividual(genome))
            return QDIndividual(genome)

        def vary(parent):
            child = QDIndividual(parent.genome.mutated(self.rng, self.id_manager))
            self.genealogical_info.append((child.id(), parent.id()))
            self._curiosity_lut[child.id()] = self.container.index_grid(parent.features)
            return child

        sel_or_init = partial(tools.sel_or_init, init_fn=init, sel_fn=select, sel_pb=1)

        run_folder = Path(options.run_folder)
        if options.overwrite and run_folder.exists():
            shutil.rmtree(options.run_folder, ignore_errors=True)
            logging.warning(f"Purging contents of {options.run_folder}, as requested")

        run_folder.mkdir(parents=True, exist_ok=False)

        self.labels = labels
        self._curiosity_lut = {}

        Evolution.__init__(self, container=container, name=name,
                           budget=options.budget, batch_size=options.batch_size,
                           select_or_initialise=sel_or_init, vary=vary,
                           optimisation_task="maximisation",
                           **kwargs)

    def tell(self, individual: IndividualLike, *args, **kwargs) -> bool:
        grid: Grid = self.container
        added = super().tell(individual, *args, **kwargs)
        parent = self._curiosity_lut.pop(individual.id(), None)
        if parent is not None:
            grid.curiosity[parent] += {True: 1, False: -.5}[added]
        
        self.save_grid_stats()
        return added
    
    def update_grid(self, updates):
        #Remove old
        self.genealogical_info=[]
        self.container.empty()
        #Add new
        for ind, r in updates:
            ind.update(r)
            self.tell(ind)

    def save_grid_stats(self):
        fitness_values = np.array([ind.fitness.values for ind in self.container])
        avg = float(np.mean(fitness_values, axis=0) if len(fitness_values) else np.nan)
        std = float(np.std(fitness_values, axis=0))
        max_f = float(np.max(fitness_values, axis=0))
        qd_score = float(self.container.qd_score(normalized=False))
        self.stats.loc[len(self.stats)] = [ avg, std, max_f, qd_score]



class Grid(containers.Grid):
    def __init__(self, **kwargs):
        containers.Grid.__init__(self, **kwargs)
        self.curiosity = np.zeros(self._shape, dtype=float)

    def update(self, iterable: Iterable,
               ignore_exceptions: bool = True, issue_warning: bool = True) -> int:
        added = containers.Grid.update(self, iterable, ignore_exceptions, issue_warning)
        return added

    def add(self, individual: IndividualLike,
            raise_if_not_added_to_depot: bool = False) -> Optional[int]:
        r = containers.Grid.add(self, individual, raise_if_not_added_to_depot)
        return r
    
    def discard(self, individual: IndividualLike, also_from_depot: bool = True)-> None:
        containers.Grid.discard(self, individual, also_from_depot)
    
    def empty(self):
        to_remove=[]
        for _, element in enumerate(self):
            to_remove.append(element)
        for i in to_remove:  
            self.discard(i, True)

    

        




    
        


class Logger(algorithms.TQDMAlgorithmLogger):

    final_filename = "iteration-final.p"
    iteration_filenames = "iteration-%03i.p"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         final_filename = Logger.final_filename,
                         iteration_filenames=self.iteration_filenames)

    def _started_optimisation(self, algo: QDAlgorithmLike) -> None:
        """Do a mery dance so that tqdm uses stdout instead of stderr"""
        sys.stderr = sys.__stdout__
        super()._started_optimisation(algo)
        self._tqdm_pbar.file = sys.stdout

    def _vals_to_cols_title(self, content: Sequence[Any]) -> str:
        header = algorithms.AlgorithmLogger._vals_to_cols_title(self, content)
        mid_rule = "-" * len(header)
        return header + "\n" + mid_rule

    def _tell(self, algo: QDAlgorithmLike, ind: IndividualLike) -> None:
        super()._tell(algo, ind)

    def summary_plots(self,**kwargs):
        summary_plots(evals=self.evals, iterations=self.iterations,
                      grid=self.algorithms[0].container,
                      labels=self.algorithms[0].labels,
                      output_dir=self.log_base_path,
                      name=Path(self.final_filename).stem,
                      **kwargs)
        
    def level_summary(self,output_dir,**kwargs):
        level_summary(evals=self.evals, iterations=self.iterations,
                      grid=self.algorithms[0].container,
                      labels=self.algorithms[0].labels,
                      output_dir=output_dir,
                      **kwargs)
        


def plot_grid(data, filename, xy_range, cb_range, labels, fig_size, cmap="inferno",
              fontsize=12, nb_ticks=5):
    fig, ax = plt.subplots(figsize=fig_size)

    if cb_range in [None, "equal"]:
        cb_range_arg = cb_range
        cb_range = np.quantile(data, [0, 1])

        if isinstance(cb_range_arg, str):
            if cb_range_arg == "equal":
                extrema = max(abs(cb_range[0]), abs(cb_range[1]))
                cb_range = (-extrema, extrema)
            else:
                raise ValueError(f"Unkown cb_range type '{cb_range}'")

    g_shape = data.shape
    cax = ax.imshow(data.T, interpolation="none", cmap=plt.get_cmap(cmap),
                    vmin=cb_range[0], vmax=cb_range[1],
                    aspect="equal",
                    origin='lower', extent=(-.5, g_shape[0]+.5, -.5, g_shape[1]+.5))

    # Set labels
    def ticks(i):
        return np.linspace(-.5, g_shape[i]+.5, nb_ticks), [
            f"{(xy_range[i][1] - xy_range[i][0]) * x / g_shape[i] + xy_range[i][0]:3.3g}"
            for x in np.linspace(0, g_shape[i], nb_ticks)
        ]

    ax.set_xlabel(labels[1], fontsize=fontsize)
    ax.set_xticks(*ticks(0))
    ax.set_yticks(*ticks(1))
    ax.set_ylabel(labels[2], fontsize=fontsize)
    ax.autoscale_view()

    ax.xaxis.set_tick_params(which='minor', direction="in", left=False, bottom=False, top=False, right=False)
    ax.yaxis.set_tick_params(which='minor', direction="in", left=False, bottom=False, top=False, right=False)
    ax.set_xticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[1], 1), minor=True)

    # Place the colorbar with same size as the image
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size=0.5, pad=0.15)
    cbar = fig.colorbar(cax, cax=cax2, format="%g")
    cbar.ax.tick_params(labelsize=fontsize-2)
    cbar.ax.set_ylabel(labels[0], fontsize=fontsize)

    # Write
    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if filename.exists():
        logging.info(f"Generated {filename}")
    else:
        logging.warning(f"Failed to generate {filename}")
    plt.close()


def summary_plots(evals: pd.DataFrame, iterations: pd.DataFrame, grid: Grid,
                  output_dir: Path, name: str,
                  labels, ext="png", fig_size=(4, 4), ticks=5):

    output_path = Path(output_dir).joinpath("plots")
    def path(filename): return output_path.joinpath(f"{name}_{filename}.{ext}")
    output_path.mkdir(exist_ok=True)
    assert len(str(name)) > 0

    if name.endswith("final"):
        plot_evals(evals["max0"], path("fitness_max"), ylabel="Fitness", figsize=fig_size)
        #plot_evals(ecal)
        ylim_contsize = (0, len(grid)) if np.isinf(grid.capacity) else (0, grid.capacity)
        plot_evals(evals["cont_size"], path("container_size"), ylim=ylim_contsize, ylabel="Container size",
                   figsize=fig_size)
        plot_iterations(iterations["nb_updated"], path("container_updates"), ylabel="Number of updated bins",
                        figsize=fig_size)
    
    for filename, cb_label, data, bounds in [
        ("grid_fitness", labels[0], grid.quality_array[..., 0], grid.fitness_domain[0]),
        #("grid_activity", "activity", grid.activity_per_bin, (0, np.max(grid.activity_per_bin))),
        #("grid_curiosity", "curiosity", grid.curiosity, "equal")
    ]:
        plot_path = path(filename)
        plot_grid(data=data, filename=plot_path,
                  xy_range=grid.features_domain, cb_range=bounds, labels=[cb_label, *labels[1:]],
                  fig_size=fig_size, nb_ticks=ticks)
                  
def level_summary(evals: pd.DataFrame, iterations: pd.DataFrame, grid: Grid,
                  output_dir: Path, labels, ext="png", fig_size=(4, 4), ticks=5):

    output_path = Path(output_dir)
    def path(filename): return output_path.joinpath(f"{filename}.{ext}")
    output_path.mkdir(exist_ok=True)

    plot_evals(evals["max0"], path("fitness_max"), ylabel="Fitness", figsize=fig_size)
    ylim_contsize = (0, len(grid)) if np.isinf(grid.capacity) else (0, grid.capacity)
    plot_evals(evals["cont_size"], path("container_size"), ylim=ylim_contsize, ylabel="Container size",
                figsize=fig_size)
    plot_iterations(iterations["nb_updated"], path("container_updates"), ylabel="Number of updated bins",
                    figsize=fig_size)
    
    for filename, cb_label, data, bounds in [
        ("grid_fitness", labels[0], grid.quality_array[..., 0], grid.fitness_domain[0]),
        #("grid_activity", "activity", grid.activity_per_bin, (0, np.max(grid.activity_per_bin))),
        #("grid_curiosity", "curiosity", grid.curiosity, "equal")
    ]:
        plot_path = path(filename)
        plot_grid(data=data, filename=plot_path,
                  xy_range=grid.features_domain, cb_range=bounds, labels=[cb_label, *labels[1:]],
                  fig_size=fig_size, nb_ticks=ticks)


class Tee:
    """Ensure that everything that's printed is also saved
    """

    class PassthroughStream:
        """Forwards received messages to log/file"""
        def __init__(self, parent: 'Tee'):
            self.tee = parent

        def write(self, msg):
            self.tee.write(msg)

        def flush(self):
            self.tee.flush()

        def isatty(self): return self.tee.out.isatty()

        def close(self): pass

    class FormattedStream(PassthroughStream):
        """Forwards received messages to log/file """
        def __init__(self, parent: 'Tee', formatter: str):
            super().__init__(parent)
            self.formatter = formatter

        def write(self, msg):
            super().write(self.formatter.format(msg))

    def __init__(self, filter_out: Optional[Callable[[str], bool]] = lambda _: False):
        self.out = sys.stdout
        self.log = None
        self.msg_queue = []    # Collect until log file is available
        self.registered = False
        self.filter = filter_out

    def register(self):
        if not self.registered:
            sys.stdout = self.PassthroughStream(self)
            sys.stderr = self.FormattedStream(self, Fore.RED + "{}" + Style.RESET_ALL)
            self.registered = True

    def teardown(self):
        if self.registered:
            self.flush()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.registered = False

    def set_log_path(self, path: Path):
        self.register()
        self.log = open(path, 'w')
        for msg in self.msg_queue:
            self._write(msg)

    def _write(self, msg: str):
        if not self.filter(msg):
            self.log.write(msg)

    def write(self, msg: str):
        if self.log is None:
            self.msg_queue.append(msg)
        else:
            self._write(msg)
        self.out.write(msg)

    def flush(self):
        self.out.flush()
        if self.log is not None:
            self.log.flush()