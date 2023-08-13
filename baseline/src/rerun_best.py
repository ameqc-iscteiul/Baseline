#!/usr/bin/env python3
import json
import os
import glob
import pandas as pd
from pathlib import Path
import random
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

    if ANN_display and save_path is not None:
        os.makedirs(f'{save_path}/ANNs', exist_ok=True) 
        plotly_render(e.get_ann()).write_html(f'{save_path}/ANNs/{g.id()}.html')
       
        
def get_genealogical_trees(run_path):
    #Get Options
    with open(f'{run_path}/config.json', "rb") as f:
        data = json.load(f)
        options = data["evolution"]
    
    #Get son father pairs
    with open(Path(run_path).joinpath("son_father_pairs.json"), "r") as file:
        fam_list = json.load(file)

    final_level = (int(options['level'])+int(options['numb_levels']))-1
    #Get Final Grid
    grid = pd.read_csv(f"{run_path}/{final_level}/final_grid.csv")
    final_ids = grid['id'].tolist()
    
    create_genealogy_tree(fam_list, f'{run_path}/genealogical_trees')
    final_grid_ancestry(fam_list, final_ids, f'{run_path}/success_tree')
    



def run_random_genome(name, view=False ):
    rng = random.Random()
    rng.seed(100)
    r = RunnerOptions()
    r.level=0
    #r.return_ann=True
    if view:
        r.view=RunnerOptions.View()
    e = Evaluator()
    e.set_runner_options(r)
    #e.set_view_dims(2,1)
    #e.set_descriptors(["trajectory", "white_gazing"])
    e.set_options(["trajectory", "white_gazing"],'brightness', 0, 4,4, r.level)

    g = RVGenome.random(rng, GIDManager())
    #result = e.evaluate_rerun(g)
    result = e.evaluate_rerun(g)
    #plotly_render(e.get_ann(g)).write_html(f'ANN_{name}.html')
    print("result", result)


def main():
    
    run_path = 'C:/Users/anton/Desktop/Thesis_Project/Baseline/baseline/results/test-g2/run8131843'
    get_genealogical_trees(run_path)



    #for r in range(10):
    run_random_genome('' )
    exit()
    run = 'run8032023'
    run_path = f'C:/Users/anton/Desktop/Thesis_Project/Baseline/baseline/results/Default/{run}'
    #make_videos_before_after(run_path)
    #make_final_videos(run_path)
    #exit()
    with open(f"{run_path}/config.json", "rb") as f:
        data = json.load(f)
        options = data["evolution"]

    

    level=options['level']
    run_path=f'{run_path}/{level}'
    final_grid = pd.read_csv(f"{run_path}/final_grid.csv")
    #final_grid = pd.read_csv(f"baseline/src/final_grid.csv")
    # Extract 'trajectory' and 'white_gazing' columns from the 'descriptors' dictionary
    final_grid[['trajectory', 'white_gazing']] = final_grid['descriptors'].apply(lambda x: pd.Series(eval(x)))
    # Drop the 'descriptors' column
    final_grid = final_grid.drop('descriptors', axis=1)    

    successful = final_grid[final_grid['fitnesses'] > 5]
    #print(successful)
    
    #for i in range(len(successful)):
    g = RVGenome.from_json(eval(successful['genome'].iloc[0]))
    rerun(g, options, save_path=run_path, view=True)


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
