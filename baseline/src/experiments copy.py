#!/usr/bin/env python3
from evolution import evolution, Options


def run_experiment(result_folder_name, batch_size, budget_size, features, w, h, l):
   o=Options()
   o.base_folder=result_folder_name
   o.batch_size=batch_size
   o.budget=budget_size
   o.tournament = 3
   o.initial_mutations = 3
   o.vision_w=w
   o.vision_h=h
   o.scenario_level=l
   o.descriptor_names=features
   evolution(o)

def main():
   print("Start")
   evals=80
   w=3
   h=3
   run_experiment(f"./new_Experiment_2_2Results_3X3", 20, evals, ["white_gazing", "distance"], w,h,l)

   



if __name__ == '__main__':
    main()