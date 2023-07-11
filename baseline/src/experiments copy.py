#!/usr/bin/env python3
from evolution import evolution, Options


def run_experiment(result_folder_name, batch_size, budget_size, w, h, l):
   o=Options()
   o.base_folder=result_folder_name
   o.batch_size=batch_size
   o.budget=budget_size
   o.tournament = 3
   o.initial_mutations = 3

   o.vision_w=w
   o.vision_h=h
   
   o.scenario_level=0
   o.descriptor_names=["distance", "avg_speed"]

   evolution(o)

def main():
   print("Start")
   evals=400
   w=2
   h=2
   run_experiment(f"./Experiment_Results_2X2", 20, evals,w,h,l=0)

   



if __name__ == '__main__':
    main()