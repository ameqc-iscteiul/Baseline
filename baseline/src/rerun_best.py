#!/usr/bin/env python3
import json
from pathlib import Path

from simulation.evaluator import Evaluator
from simulation.my_runner import RunnerOptions
from robot.genome import RVGenome


def record_rerun(run:str):
    options = Evaluator.options()
    options.save_folder = Path("C:/Users/anton/Desktop/Thesis_Project/Baseline/baseline/src/recordings")  # Update the save folder path
    
    # Create the save folder if it doesn't exist
    options.save_folder.mkdir(parents=True, exist_ok=True)
    
    for i in range(1,7):
        with open(f"./Experiment_Results/{run}/best{i}.json", "rb") as f:
            data = json.load(f)
            genome = RVGenome.from_json(data["genome"])
            f=str(round(data['fitnesses']['closeness'],3))
            options.record = Evaluator.options.Record(
                video_file_path=options.save_folder / f"video{f}.mp4"  # Update the video file path
            )   
            result, viewer = Evaluator.evaluate_rerun(genome, options)

def watch_rerun(data, e:Evaluator):
    genome = RVGenome.from_json(data["genome"])
    f = data["fitnesses"]["brightness"]
    if f > 88:
        result = e.evaluate_rerun(genome)
        print("result", result)

def simple_rerun(data, e:Evaluator):    
    genome = RVGenome.from_json(data["genome"]) 
    f = data["fitnesses"]["brightness"]
    if f > 88:
        result = e.evaluate_evo(genome)
        print("result", result)


                



def main():
    print("Start")
    run='run7071728'
    n=9
    r = RunnerOptions()
    r.level = 4
    

    print(r)
    e = Evaluator()
    e.set_target_options(2,2)
    e.set_runner_options(r)
    e.set_descriptors(["distance", "avg_speed"])
    

    for i in range(1,n):
        with open(f"./Experiment_Results/{run}/best{i}.json", "rb") as f:
            data = json.load(f)
            watch_rerun(data,e)
            #simple_rerun(data,e)
    #record_rerun(run)
    

if __name__ == "__main__":
    main()
