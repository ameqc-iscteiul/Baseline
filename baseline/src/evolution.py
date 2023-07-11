import json
import logging
from functools import partial
import pprint
import sys
import time
from typing import Optional, List
from datetime import timedelta
from pathlib import Path
import humanize
from abrain.core.genome import logger as genome_logger
import pandas as pd

from evo_alg.my_map_elite import Tee,QDIndividual,Algorithm, Grid, EvaluationResult, Logger
from robot.genome import RVGenome
from simulation.config import Config
from simulation.evaluator import Evaluator
from simulation.my_runner import RunnerOptions

class Options:
    def __init__(self):
        self.id: Optional[int] = None
        self.base_folder: str = "baseline/tmp/qdpy/toy-revolve"
        self.run_folder: str = None  # Automatically filled in
        self.snapshots: int = 10
        self.overwrite: bool = False
        self.verbosity: int = 1

        self.seed: int = 100
        self.batch_size: int = 20
        self.budget: int = 100
        self.tournament: int = 3
        self.threads: int = 1
        # number of initial mutations for abrain's genome
        self.initial_mutations: int = 3

        #-----------------
        #Evaluator Options:
        #-----------------
        #Scenario:
        self.scenario_level : int =0 
        #-Target
        self.target_position=[2,0,0]
        #-Robot Vision
        self.vision_w: int = 2
        self.vision_h: int = 2
        #fitness name
        self.fitness_name: str
        #descriptor names
        self.descriptor_names: List

        
def eval_mujoco(ind:QDIndividual, evaluator : Optional[Evaluator]):
    assert isinstance(ind, QDIndividual)
    assert isinstance(ind.genome, RVGenome)
    assert ind.id() is not None, "ID-less individual"
    r: EvaluationResult = evaluator.evaluate_evo(ind.genome)
    ind.update(r)
    # print(ind, r)
    return ind


def evolution (args : Options()):
    
    start = time.perf_counter()
    # Log everything to file except for the progress bar
    tee = Tee(filter_out=lambda msg: "\r" in msg or "\x1b" in msg)
    tee.register()  # Start capturing now (including logging's reference to stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]|[%(levelname)s]|[%(module)s] %(message)s",
        stream=sys.stdout
    )
    genome_logger.setLevel(logging.INFO)

    for m in ['matplotlib', 'OpenGL.arrays.arraydatatype', 'OpenGL.acceleratesupport']:
        logger = logging.getLogger(m)
        logger.setLevel(logging.WARNING)
        logging.info(f"Muting {logger}")

    logging.captureWarnings(True)

    ########################################################################################
    evaluator =  Evaluator()
    evaluator.set_target_options(args.vision_w, args.vision_h)
    r = RunnerOptions()
    r.level = args.scenario_level
    evaluator.set_runner_options(r)
    evaluator.set_descriptors(args.descriptor_names)

    grid = Grid(shape=(16, 16),
                max_items_per_bin=1,
                fitness_domain=evaluator.fitness_bounds(),
                features_domain=evaluator.descriptor_bounds())
    
    logging.info(f"Grid size: {grid.shape}")
    logging.info(f"   bounds: {grid.features_domain}")
    logging.info(f"     bins: "
                 f"{[(d[1]-d[0]) / s for d, s in zip(grid.features_domain, grid.shape)]}")
    
    
    algo = Algorithm(grid, args, labels=[evaluator.fitness_name(), *evaluator.descriptor_names()])
    run_folder = Path(args.run_folder)

    
    # Prepare (and store) configuration
    Config.argparse_process(args)
    Config.evolution = args.__dict__
    Config._evolving = True

    config_path = run_folder.joinpath("config.json")
    Config.write_json(config_path)
    logging.info(f"Stored configuration in {config_path.absolute()}")
    
    # Create a logger to pretty-print everything and generate output data files
    save_every = round(args.budget / (args.batch_size * args.snapshots))
    logger = Logger(algo,
                    save_period=save_every,
                    log_base_path=args.run_folder)
    tee.set_log_path(run_folder.joinpath("log"))

    '''with ParallelismManager(parallelism_type = "sequential", max_workers=20) as mgr:
        #mgr.executor._mp_context = multiprocessing.get_context("fork")  # TODO: Very brittle
        #executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)'''
    
    logging.info("Starting illumination!")
    new_eval =  partial(eval_mujoco, evaluator=evaluator)
    best = algo.optimise(evaluate=new_eval, batch_mode=True)



    #Collect Best
    i=0
    for _, element in enumerate(grid):
        if element.fitness[0]>80: 
            i+=1
            with open(Path(args.run_folder).joinpath(f"best{i}.json"), 'w') as f:
                data = {
                    "id": element.id(), "parents": element.genome.parents(),
                    "fitnesses": element.fitnesses,
                    "descriptors": element.descriptors,
                    #"stats": best.stats,
                    "genome": element.genome.to_json()
                }
                logging.info(f"best:\n{pprint.pformat(data)}")
                json.dump(data, f)

    #Store all Map Elites final Solutions
    grid_pop = []
    for _, element in enumerate(grid):
        ind = { "id": element.id(), "parents": element.genome.parents(),
                "fitnesses": element.fitnesses,
                "descriptors": element.descriptors,
                "genome": element.genome.to_json()}
        
        grid_pop.append(ind)
    df = pd.DataFrame(grid_pop)
    df.to_csv(Path(args.run_folder).joinpath("final_grid.csv"), index=False)
    
        
    # Print results info
    logging.info(algo.summary())

    # Plot the results
    logger.summary_plots()

    logging.info(f"All results are available under {logger.log_base_path}")
    logging.info(f"Unified storage file is {logger.log_base_path}/{logger.final_filename}")

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    logging.info(f"Completed evolution in {duration}")



