#!/usr/bin/env python3
from evolution import evolution, Options


def run_experiment(result_folder_name, batch_size, budget_size, features, w, h, numb_levels, init_level):
   o=Options()
   o.base_folder=result_folder_name
   o.batch_size=batch_size
   o.budget=budget_size
   o.tournament = 3
   o.initial_mutations = 2
   o.vision_w=w
   o.vision_h=h
   o.numb_levels=numb_levels
   o.level=init_level
   o.descriptor_names=features
   o.grid_size = 16
   evolution(o)

def main():
   print("Start")

   w,h = 4,4
   #Only on harder
   run_experiment(f"baseline/Experiment_Traj_Gaze/{w}X{h}_only_level_3", 20, 40000, ["trajectory", "white_gazing"], w,h, 1, 3)
   #Increase from 0 to 3
   run_experiment(f"baseline/Experiment_Traj_Gaze/{w}X{h}_0_to_3", 20, 40000, ["trajectory", "white_gazing"], w,h, 4, 0)

   
if __name__ == '__main__':
    main()