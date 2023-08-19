import json
import logging
from functools import partial
import math
import os
import pprint
import sys
import time
from typing import Optional, List
from datetime import timedelta
from pathlib import Path
import humanize
from abrain.core.genome import logger as genome_logger
import pandas as pd
from utils import save_grid, create_genealogy_tree, final_grid_ancestry
from evo_alg.my_map_elite import Tee,QDIndividual,Algorithm, Grid, EvaluationResult, Logger
from robot.genome import RVGenome
from simulation.config import Config
from simulation.evaluator import Evaluator
from simulation.runner import RunnerOptions
from Grid_Analyser import analyse_Experiment


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

        self.batch_size: int = 20
        self.threads: int = 1
        self.budget: int = 100

        self.tournament: int = 3
        #-Grid options
        self.grid_size : int = 16
        self.fitness_name: str
        self.descriptor_names = []
        #-number of initial mutations for abrain's genome
        self.initial_mutations: int = 2

        #-----------------
        #Evaluator Options:
        #-----------------
        #Scenario:
        self.level : int = 0
        self.numb_levels: int = 0 

        #-Robot
        self.vision_w: int 
        self.vision_h: int 
        self.robot_type : int = 0

        self.make_final_videos=False


def eval_mujoco(ind : QDIndividual, evaluator : Evaluator , options : Options):
    '''Happens in Multithreading'''
    assert isinstance(ind, QDIndividual)
    assert isinstance(ind.genome, RVGenome)
    assert ind.id() is not None, "ID-less individual"
    evaluator.set_options(options.descriptor_names,options.fitness_name,options.robot_type, options.vision_w, options.vision_h, options.level)
    r: EvaluationResult = evaluator.evaluate(ind.genome)
    ind.update(r)
    #print("ind",ind)
    return ind

def change_level(algo, evaluator, args, logger):
    args.level+=1
    evaluator.set_level(args.level)
    
    '''Revaluate'''
    updates=[]
    for _, element in enumerate(algo.container):
        individual : QDIndividual = element
        r: EvaluationResult = evaluator.evaluate(individual.genome)
        updates.append((individual, r))
    algo.update_grid(updates)
    '''Plots'''
    #logger.summary_plots(extraname=f'_{args.level}_beginning')
    #save_grid(algo.container, f'{args.run_folder}/{args.level}', 'initial') 
    return algo, evaluator, args



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
    evaluator.set_options(args.descriptor_names,args.fitness_name,args.robot_type, args.vision_w, args.vision_h, args.level)
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

    logging.info("Starting illumination!")

    import platform
    if platform.system() == "Linux":
        from qdpy.base import ParallelismManager
        #import multiprocessing
        with ParallelismManager(parallelism_type = "multiprocessing", max_workers=args.threads) as mgr:
            #mgr.executor._mp_context = multiprocessing.get_context("fork")  # TODO: Very brittle
            budget_per_level = int(args.budget/args.numb_levels)
            for l in range(args.numb_levels):
                if l>0: algo, evaluator, args = change_level(algo, evaluator, args, logger)
                logging.info(f"Level : {args.level}")
                best = algo.optimise(evaluate=partial(eval_mujoco, evaluator = evaluator, options = args), budget=budget_per_level, executor=mgr.executor,batch_mode=True)
                logging.info(f"Level {args.level} finished. Saving Final Grid & Plots ...")
                save_results(algo, args, logger)  
    
    elif platform.system() == "Windows":
        budget_per_level = int(args.budget/args.numb_levels)
        for l in range(args.numb_levels):
            if l>0: algo, evaluator, args = change_level(algo, evaluator, args, logger)
            logging.info(f"Level : {args.level}")
            best = algo.optimise(evaluate=partial(eval_mujoco, evaluator = evaluator, options = args), budget=budget_per_level,batch_mode=True)
            logging.info(f"Level {args.level} finished. Saving Final Grid & Plots ...")
            save_results(algo, args, logger)   
    else:
        print("ERROR: Unknown operating system")
    
    # Print results info
    logging.info(algo.summary())
    # Plot the results
    logging.info(f"All results are available under {logger.log_base_path}")
    logging.info(f"Unified storage file is {logger.log_base_path}/{logger.final_filename}")

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    logging.info(f"Completed evolution in {duration}")
   
    logging.info(f"Generating Videos...")
    if args.make_final_videos: analyse_Experiment(args.run_folder)
    logging.info(f"DONE !")


def save_results(algo, args, logger):
    #Save final grid, plots, and genealogic_list from this level
    save_path = f'{args.run_folder}/{args.level}'

    '''Plots'''
    logger.level_summary(save_path)

    '''Grid'''
    save_grid(algo.container,save_path,'final')

    '''Tree'''
    with open(Path(f"{args.run_folder}/{args.level}/son_father_pairs.json"), "w") as file:
        json.dump(algo.genealogical_info, file)
