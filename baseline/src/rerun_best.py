#!/usr/bin/env python3
import json
import os
import glob
import pandas as pd
from pathlib import Path
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

from abrain import plotly_render
from simulation.evaluator import Evaluator
from simulation.runner import RunnerOptions
from robot.genome import RVGenome
from abrain.core.genome import GIDManager
from utils import save_grid, create_genealogy_tree, final_grid_ancestry




def make_videos(run_path, top=5):
    with open(f'{run_path}/config.json', "rb") as f:
        data = json.load(f)
        options = data["evolution"]
    for folder_name in os.listdir(run_path):
        # Construct the full path to the current subfolder
        folder_path = os.path.join(run_path, folder_name)
        # Check if the item is a directory (subfolder)
        if os.path.isdir(folder_path):
            # Loop through all files in the subfolder
            for file_name in os.listdir(folder_path):
                # Check if the file is a CSV file (you can add additional checks if needed)
                if file_name.endswith('final_grid.csv'):
                    # Construct the full path to the CSV file
                    csv_file_path = os.path.join(folder_path, file_name)
                    subfolder_name = os.path.basename(os.path.dirname(csv_file_path))
                    options['level']=int(subfolder_name)
                    # Process the CSV file
                    grid = pd.read_csv(csv_file_path)  
                    sorted_grid = grid.sort_values(by='fitnesses', ascending=False)
                    successful = sorted_grid.head(top)
                    for i in range(len(successful)):
                        g = RVGenome.from_json(eval(successful['genome'].iloc[i]))
                        rerun(g, options, save_path=folder_path, record=True, ANN_display = True)


def rerun(g : RVGenome, options, save_path=None, view=False, record=False, ANN_display = False, name=''):

    e = Evaluator()
    r = RunnerOptions()   
    
    if view:
        r.view = RunnerOptions.View()
    elif record:
        os.makedirs(f'{save_path}/Videos', exist_ok=True) 
        r.record = RunnerOptions.Record(video_file_path=f'{save_path}/Videos/{g.id()}.mp4')
    e.set_runner_options(r)
    e.set_options(options['descriptor_names'],options['fitness_name'], options['robot_type'], options['vision_w'],options['vision_h'], options['level'])
    result = e.evaluate_rerun(g)
    e.ann_descriptor()
    
    if ANN_display and save_path is not None:
        os.makedirs(f'{save_path}/ANNs', exist_ok=True) 
        plotly_render(e.get_ann()).write_html(f'{save_path}/ANNs/{g.id()}.html')
    #print(result)
        
def get_genealogical_trees(run_path):
    #Get Options
    with open(f'{run_path}/config.json', "rb") as f:
        data = json.load(f)
        options = data["evolution"]
    
    #Get son father pairs
    with open(Path(run_path).joinpath(f"{options['level']}/son_father_pairs.json"), "r") as file:
        fam_list = json.load(file)

    #level = (int(options['level'])+int(options['numb_levels']))-1
    for level in range(int(options['level']),int(options['level'])+int(options['numb_levels'])):
        #Get Final Grid
        grid = pd.read_csv(f"{run_path}/{level}/final_grid.csv")
        final_ids = grid['id'].tolist()
        final_grid_ancestry(fam_list, final_ids, f'{run_path}/{level}/success_tree')
    



def run_random_genome(name, view=False ):
    rng = random.Random()
    rng.seed(100)
    r = RunnerOptions()
    r.level=6
    #r.return_ann=True
    if view:
        r.view=RunnerOptions.View()
    e = Evaluator()
    e.set_runner_options(r)
    #e.set_view_dims(2,1)
    #e.set_descriptors(["trajectory", "white_gazing"])
    e.set_options(["EdgePerNodeRatio", "estimated_mean_z"],'brightness', 1, 4,4, r.level)

    g = RVGenome.random(rng, GIDManager())
    #result = e.evaluate_rerun(g)
    result = e.evaluate_rerun(g)
    #print("result", result)

    

    
def main():
    
    '''run_random_genome('', view=True)
    exit()'''

    run_path = f'C:/Users/anton/Desktop/Thesis_Project/Baseline/baseline/results/run8300544'
    
    with open(f"{run_path}/config.json", "rb") as f:
        data = json.load(f)
        options = data["evolution"]

    final_grid = pd.read_csv(f"{run_path}/6/final_grid.csv")
   
    #print(RVGenome.from_json(eval(final_grid['genome'].iloc[0])))
    '''print('best')
    for i in range(10):
        g = RVGenome.from_json(eval(final_grid['genome'].iloc[i]))
        rerun(g, options, view=False, save_path=None, ANN_display=False)'''
        
    #print('worse')
    for i in range(173,180):    
        g = RVGenome.from_json(eval(final_grid['genome'].iloc[i]))
        rerun(g, options, view=True, save_path=None,ANN_display=False)
    
    #get_genealogical_trees(run_path)
        
if __name__ == "__main__":
    main()
