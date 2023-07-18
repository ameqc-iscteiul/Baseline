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

    #if run_path is not None:
    #    abrain.plotly_render(ann).write_html(f"{run_path}/{g.id()}ann.html")



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
    e.set_view_dims(4,2)
    e.set_descriptors(["trajectory", "white_gazing"])
    g = RVGenome.random(rng, GIDManager)
    result, ann = e.evaluate_rerun(g)
    plotly_render(ann).write_html("./1.2sample_ann.html")
    print("result", result)


def main():
    #run_random_genome()
    #exit()

    run_path = "./new_Experiment_3Results_4X4/run7160115" 
    final_grid = pd.read_csv(f"{run_path}/final_grid.csv")
    #selected_genome = final_grid.iloc[14]['genome']
    best_g = RVGenome.from_json(eval(final_grid['genome'].head(1)[0]))  

    with open(f"{run_path}/config.json", "rb") as f:
        data = json.load(f)
        options = data["evolution"]
    #for i in range(len(final_grid['genome'])):
    g=final_grid.iloc[25]['genome']
    rerun(RVGenome.from_json(eval(g)), options, run_path)
    rerun(best_g, options, run_path, record=True)

    #ann = Brain.make_controller()
    #plotly_render(ann).write_html("./sample_ann.html")

    
    
        


    '''pickle_file_path = "./Experiment_Results/run7071728/iteration-010.p"

    # Unpickle the object
    with open(pickle_file_path, "rb") as f:
        unpickled_object = pickle.load(f)

    print("unpickled_object",unpickled_object)'''

if __name__ == "__main__":
    main()
