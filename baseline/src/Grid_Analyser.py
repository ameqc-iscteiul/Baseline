#!/usr/bin/env python3
import ast
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
from utils import final_grid_ancestry


def get_Grid (grid_path):
    final_grid = pd.read_csv(f"{grid_path}/final_grid.csv")

    '''Converting Descriptor Dict, into 2 Descriptor columns'''
    final_grid['descriptors'] = final_grid['descriptors'].apply(ast.literal_eval)
    for key in final_grid['descriptors'].iloc[0].keys():
        final_grid[key] = final_grid['descriptors'].apply(lambda x: x.get(key))
    final_grid = final_grid.drop('descriptors', axis=1)

    return final_grid

def get_genealogical_trees(grid_df, run_path):
    #Get son father pairs
    with open(Path(run_path).joinpath("son_father_pairs.json"), "r") as file:
        fam_list = json.load(file)
    final_ids = grid_df['id'].tolist()
    final_grid_ancestry(fam_list, final_ids, f'{run_path}/gen_tree')    

def extract_from_grid(grid_df, options, run_path):
    for i in range(len(grid_df)):
        g = RVGenome.from_json(eval(grid_df['genome'].iloc[i]))
        save_ann_and_record(g,options,run_path)

def save_ann_and_record(g:RVGenome, options, save_path):
    rerun(g, options, save_path, record=True, ANN_display=True)

def record_genome(g:RVGenome, options, save_path):
    rerun(g, options, save_path, record=True)

def save_ann(g:RVGenome, options, save_path):
    rerun(g, options, save_path, ANN_display=True)

def rerun(g : RVGenome, options, save_path=None, view=False, record=False, ANN_display = False):
    e = Evaluator()
    r = RunnerOptions()   
    if view:
        r.view = RunnerOptions.View()
    elif record:
        os.makedirs(f'{save_path}/Videos', exist_ok=True) 
        r.record = RunnerOptions.Record(video_file_path=f'{save_path}/Videos/{g.id()}.mp4')
    e.set_runner_options(r)
    e.set_options(options['descriptor_names'],options['fitness_name'], options['robot_type'], options['vision_w'],options['vision_h'], options['level'])
    e.evaluate_rerun(g)
    if ANN_display and save_path is not None:
        os.makedirs(f'{save_path}/ANNs', exist_ok=True) 
        plotly_render(e.get_ann()).write_html(f'{save_path}/ANNs/{g.id()}.html')
       
        
'''Main Mth'''
def analyse_Grid(run_path, options, top=5):
    grid_df = get_Grid(run_path)
    
    #Analyse Best Solutions
    sorted_grid = grid_df.sort_values(by='fitnesses', ascending=False)
    champs = sorted_grid.head(top)
    save_path = f'{run_path}/Top{top}'
    os.makedirs(f'{save_path}', exist_ok=True)
    extract_from_grid(champs, options, save_path)
    
    #Analyse the Successful with extreme Descriptor Values
    features=list(grid_df.columns[-2:])
    successful = grid_df[grid_df['fitnesses'] >= 70.0]
    for i in range(2):
        grid_aux = successful.sort_values(by=f'{features[i]}', ascending=False)
        extremes_grid = pd.concat([grid_aux.head(top), grid_aux.tail(top)], ignore_index=True).drop_duplicates()
        save_path = f'{run_path}/Extremes/{features[i]}'
        os.makedirs(f'{save_path}', exist_ok=True)
        extract_from_grid(extremes_grid, options, save_path)
        extremes_grid.drop(['parents','genome'], axis=1,inplace=False).to_csv(f'{save_path}/collectio.csv', index=False)
    
    '''Check where the successful came from'''
    get_genealogical_trees(successful, run_path)
    

def analyse_Experiment(experiment_path):
    '''Collect Configs'''
    with open(f'{experiment_path}/config.json', "rb") as f:
        data = json.load(f)
        options = data["evolution"]
    '''Finding all final grids'''
    for subfolder_name in os.listdir(experiment_path):
        folder_path = os.path.join(experiment_path, subfolder_name)
        if os.path.isdir(folder_path) and subfolder_name != 'plots':
            options['level']=int(subfolder_name)
            analyse_Grid(folder_path, options)


def main():
    run_path = 'C:/Users/anton/Desktop/Thesis_Project/Baseline/baseline/results/run8162309'
    analyse_Experiment(run_path)

if __name__ == "__main__":
    main()
