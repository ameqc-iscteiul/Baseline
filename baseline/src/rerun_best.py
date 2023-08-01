#!/usr/bin/env python3
import json
import os
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



#A video of the top 10 individuals on all envolved environments:
def make_videos(run_path, top=10):
    final_grid = pd.read_csv(f"{run_path}/final_grid.csv")
    # Extract 'trajectory' and 'white_gazing' columns from the 'descriptors' dictionary
    final_grid[['trajectory', 'white_gazing']] = final_grid['descriptors'].apply(lambda x: pd.Series(eval(x)))
    # Drop the 'descriptors' column
    final_grid = final_grid.drop('descriptors', axis=1)    
    sorted_grid = final_grid.sort_values(by='fitnesses', ascending=False)
    successful = sorted_grid.head(top)

    with open(f"{run_path}/config.json", "rb") as f:
        data = json.load(f)
        options = data["evolution"]    
   
    for i in range(len(successful)):
        g = RVGenome.from_json(eval(successful['genome'].iloc[i]))
        rerun(g, options, run_path=run_path ,record=True, ANN_display = True)


def rerun(g : RVGenome, options, run_path=None, view=False, record=False, ANN_display = False):
    level=options['level']
    i=options['numb_levels']
    while i>0:
        r = RunnerOptions()
        if ANN_display:
            r.return_ann=True    
            os.makedirs(f'{run_path}/anns', exist_ok=True) 
        if view:
            r.view=RunnerOptions.View()
        elif record:
            os.makedirs(f'{run_path}/videos', exist_ok=True) 
            r.record=RunnerOptions.Record(video_file_path=f'{run_path}/videos/{g.id()}_{level}.mp4')
        e = Evaluator()
        e.set_runner_options(r)
        e.set_options(options['descriptor_names'], options['vision_w'],options['vision_h'], level)
        if ANN_display and run_path is not None:
            result, ann = e.evaluate_rerun(g)
            plotly_render(ann).write_html(f'{run_path}/anns/{g.id()}ann.html')
        else: result = e.evaluate_rerun(g)
        print("result", result)
        level+=1
        i-=1



def run_random_genome(view=False):
    rng = random.Random()
    rng.seed(1)
    r = RunnerOptions()
    r.level=0
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
    plotly_render(ann).write_html("Gecko_ann.html")
    print("result", result)


def main():
    #run_random_genome(view=True)
    run_path = f'C:/Users/anton/Desktop/Thesis_Project/Baseline/baseline/Testing_level_change/4X4_0_to_3/run8010009'
    make_videos(run_path)
    exit()
    #C:\Users\anton\Desktop\Thesis_Project\Baseline\baseline\Testing_level_change\4X4_0_to_3\run7312307
    run_path = "C:/Users/anton/Desktop/Thesis_Project/Baseline/baseline/Testing_level_change/4X4_0_to_3/run7312307" 
    final_grid = pd.read_csv(f"{run_path}/final_grid.csv")
    #final_grid = pd.read_csv(f"baseline/src/final_grid.csv")
    # Extract 'trajectory' and 'white_gazing' columns from the 'descriptors' dictionary
    final_grid[['trajectory', 'white_gazing']] = final_grid['descriptors'].apply(lambda x: pd.Series(eval(x)))
    # Drop the 'descriptors' column
    final_grid = final_grid.drop('descriptors', axis=1)    

    with open(f"{run_path}/config.json", "rb") as f:
        data = json.load(f)
        options = data["evolution"]
    
    '''best_g = RVGenome.from_json(eval(final_grid['genome'].head(1)[0]))  
    rerun(best_g, options, run_path)'''
    
    
    successful = final_grid[final_grid['fitnesses'] > 5]
    print(len(successful))
    #for i in range(len(successful)):
    g = RVGenome.from_json(eval(successful['genome'].iloc[1]))
    rerun(g, options, view=True)


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
