#!/usr/bin/env python3
import json
from pathlib import Path
import pickle
import random
from abrain import plotly_render
import abrain
from utils import distribute_points
import pandas as pd

from simulation.evaluator import Evaluator
from simulation.runner import RunnerOptions
from robot.genome import RVGenome
from robot.control import Brain, ANNControl
from abrain.core.genome import GIDManager




def rerun(g : RVGenome, options, run_path=None, view=False, record=False): 
    r = RunnerOptions()
    r.level=options['scenario_level']
    if view:
        r.view=RunnerOptions.View()
    elif record:
        r.record=RunnerOptions.Record(video_file_path=f"{run_path}/{g.id()}video.mp4")
    e = Evaluator()
    e.set_runner_options(r)
    e.set_view_dims(options['vision_w'],options['vision_h'])
    e.set_descriptors(options['descriptor_names'])
    
    result = e.evaluate_rerun(g)
    print("result", result)



def run_random_genome(view=False):
    rng = random.Random()
    rng.seed(1)
    r = RunnerOptions()
    r.level=3
    r.return_ann=True
    if view:
        r.view=RunnerOptions.View()
    e = Evaluator()
    e.set_runner_options(r)
    e.set_view_dims(4,4)
    e.set_descriptors(["trajectory", "white_gazing"])
    g = RVGenome.random(rng, GIDManager())
    #result = e.evaluate_rerun(g)
    result, ann = e.evaluate_rerun(g)
    #plotly_render(ann).write_html("ann.html")
    print("result", result)

def rerun2(g : RVGenome,  view=False , run_path=None, record=False): 
    r = RunnerOptions()
    #for i in range(1,6):
    r.level=2
    if view:
        r.view=RunnerOptions.View()
    elif record:
        r.record=RunnerOptions.Record(video_file_path=f"{run_path}/{g.id()}video.mp4")
    e = Evaluator()
    e.set_runner_options(r)
    e.set_view_dims(4,4)
    e.set_descriptors(["trajectory", "white_gazing"])
    
    result = e.evaluate_rerun(g)
    print("result", result)

def main():
    run_random_genome(view=True)
    exit()
    #run_path = "./new_Experiment_3Results_3X3/run7200026" 
    #final_grid = pd.read_csv(f"{run_path}/final_grid.csv")
    final_grid = pd.read_csv(f"baseline/src/final_grid.csv")
    # Extract 'trajectory' and 'white_gazing' columns from the 'descriptors' dictionary
    final_grid[['trajectory', 'white_gazing']] = final_grid['descriptors'].apply(lambda x: pd.Series(eval(x)))
    # Drop the 'descriptors' column
    final_grid = final_grid.drop('descriptors', axis=1)    

    '''with open(f"{run_path}/config.json", "rb") as f:
        data = json.load(f)
        options = data["evolution"]'''
    
    '''best_g = RVGenome.from_json(eval(final_grid['genome'].head(1)[0]))  
    rerun(best_g, options, run_path)'''
    
    
    successful = final_grid[final_grid['fitnesses'] > 100.0]
    print(len(successful))
    #for i in range(len(successful)):
    g = RVGenome.from_json(eval(successful['genome'].iloc[1]))
    rerun2(g, view=True)


    '''# Sort by 'white_gazing' column in descending order
    sorted_gazing = successful.sort_values(by = 'white_gazing', ascending=False)
    highest_gazer = RVGenome.from_json(eval(sorted_gazing['genome'].iloc[0]))
    lowest_gazer = RVGenome.from_json(eval(sorted_gazing['genome'].iloc[-1]))
    rerun(highest_gazer, options, run_path, view=True)
    rerun(lowest_gazer, options, run_path, view=True)'''

    
    
        


    '''pickle_file_path = "./Experiment_Results/run7071728/iteration-010.p"

    # Unpickle the object
    with open(pickle_file_path, "rb") as f:
        unpickled_object = pickle.load(f)

    print("unpickled_object",unpickled_object)'''

if __name__ == "__main__":
    main()
