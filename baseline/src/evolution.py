import json
import logging
from functools import partial
import math
import pprint
import sys
import time
from typing import Optional, List
from datetime import timedelta
from pathlib import Path
import humanize
from abrain.core.genome import logger as genome_logger
import pandas as pd
from utils import create_genealogical_tree
from evo_alg.my_map_elite import Tee,QDIndividual,Algorithm, Grid, EvaluationResult, Logger
from robot.genome import RVGenome
from simulation.config import Config
from simulation.evaluator import Evaluator
from simulation.runner import RunnerOptions

class Options:
    def __init__(self):
        self.id: Optional[int] = None
        self.base_folder: str 
        self.run_folder: str = None  # Automatically filled in
        self.snapshots: int = 10
        self.overwrite: bool = False
        self.verbosity: int = 1


        #Algorithm parameters
        self.seed: int = 100
        self.threads: int = 1

        self.batch_size: int = 20
        self.budget: int = 100

        self.tournament: int = 3
        #-Grid options
        self.grid_size : int = 12
        #-number of initial mutations for abrain's genome
        self.initial_mutations: int = 3

        #-----------------
        #Evaluator Options:
        #-----------------
        #Scenario:
        self.initial_level : int = 0
        self.numb_levels : int = 0 
        #-Robot Vision
        self.vision_w: int = 2
        self.vision_h: int = 2
        #fitness name
        self.fitness_name: str
        #descriptor names
        self.descriptor_names = []

        
def eval_mujoco(ind:QDIndividual, evaluator : Optional[Evaluator], grid : Optional[Grid], revaluate : bool = False):
    assert isinstance(ind, QDIndividual)
    assert isinstance(ind.genome, RVGenome)
    assert ind.id() is not None, "ID-less individual"
    # Check if it's time to change the level
    change_scenario_at = (evaluator.eval_budget // evaluator.numb_levels)
    if evaluator.counter == change_scenario_at and evaluator.evaluation != evaluator.eval_budget-1:
        evaluator.counter = 0
        print("!!!!!!!!!!!!!!!!!Next level!!!!!!!!!!!!!!!")
        print("evaluator.evaluation", evaluator.evaluation)
        # Calculate the next level
        next_level = evaluator.runner_options.level + 1
        evaluator.runner_options.level = min(next_level, 6)
        # Reset the evaluation count for the next level
        evaluator.runner_options.level +=1

        for _, element in enumerate(grid):
            eval_mujoco(element, evaluator, grid, revaluate=True)
            
    r: EvaluationResult = evaluator.evaluate_evo(ind.genome)
    ind.update(r)

    if revaluate == False:
        evaluator.counter += 1
        evaluator.evaluation +=1
        #print("ind",ind)
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
    
    evaluator = Evaluator()
    '''r = RunnerOptions()
    r.view=RunnerOptions.View()
    evaluator.set_runner_options(r)'''
    evaluator.set_options(args.descriptor_names, args.vision_w, args.vision_h, args.budget, args.numb_levels,args.initial_level)
    
    grid = Grid(shape=(args.grid_size, args.grid_size),
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

    
    '''with ParallelismManager(parallelism_type = "none", max_workers=args.threads) as mgr:
        #mgr.executor._mp_context = multiprocessing.get_context("fork")  # TODO: Very brittle
        logging.info("Starting illumination!")
        best = algo.optimise(evaluate=eval_mujoco, executor=mgr.executor, batch_mode=True)'''

    logging.info("Starting illumination!")
    new_eval =  partial(eval_mujoco, evaluator=evaluator, grid=grid)
    best = algo.optimise(evaluate=new_eval, batch_mode=True)

    
    #Store all Map Elites final Solutions
    grid_pop = []
    for _, element in enumerate(grid):
        #print("element.fitness",element.fitness[0])
        ind = { "id": element.id(), "parents": element.genome.parents(),
                "fitnesses": element.fitness[0],
                "descriptors": element.descriptors,
                "genome": element.genome.to_json()}
        
        grid_pop.append(ind)
    df = pd.DataFrame(grid_pop)
    df = df.sort_values(by="fitnesses", ascending=False)
    df.to_csv(Path(args.run_folder).joinpath("final_grid.csv"), index=False)

    '''history = pd.DataFrame(algo.history)
    history.to_csv(Path(args.run_folder).joinpath("history.csv"), index=False)

    # Save the genealogical tree dictionary as a JSON file
    with open(Path(args.run_folder).joinpath("genealogical_tree.json"), "w") as file:
        g=create_genealogical_tree(algo.genealogical_info)
        json.dump(g, file)'''
    
    
    
    ''' 
    n=10
    n_best_individuals = sorted(grid, key=lambda x: x.fitness[0], reverse=True)[:n]
    #print("best_individuals:", best_individuals)
    #Collect Best
    i=0
    for element in n_best_individuals:
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
    '''
      
    # Print results info
    logging.info(algo.summary())

    # Plot the results
    logger.summary_plots()

    logging.info(f"All results are available under {logger.log_base_path}")
    logging.info(f"Unified storage file is {logger.log_base_path}/{logger.final_filename}")

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    logging.info(f"Completed evolution in {duration}")