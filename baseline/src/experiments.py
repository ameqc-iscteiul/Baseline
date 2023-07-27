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
   o.initial_level=init_level
   o.descriptor_names=features

   evolution(o)

def main():
   print("Start")

   w,h = 4,4
   #Only on harder
   run_experiment(f"baseline/Experiment_Dist_Gaze/{w}X{h}_only_level_3", 20, 4000, ["distance", "white_gazing"], w,h, 1, 3)
   run_experiment(f"baseline/Experiment_Traj_Gaze/{w}X{h}_only_level_3", 20, 4000, ["trajectory", "white_gazing"], w,h, 1, 3)
   #Increase from 0 to 3
   run_experiment(f"baseline/Experiment_Dist_Gaze/{w}X{h}_0_to_3", 20, 4000, ["distance", "white_gazing"], w,h, 4, 0)
   run_experiment(f"baseline/Experiment_Dist_Gaze/{w}X{h}_0_to_3", 20, 4000, ["trajectory", "white_gazing"], w,h, 4, 0)

   run_experiment(f"baseline/Experiment_Dist_Gaze/{w}X{h}_climb_ramp", 20, 6000, ["distance", "white_gazing"], w,h, 3, 4)

   
  

   '''w=2
   h=2
   run_experiment(f"./new_Experiment_1Results_{w}X{h}", 20, evals, ["white_gazing", "avg_speed"], w,h,l)
   run_experiment(f"./new_Experiment_2Results_{w}X{h}", 20, evals, ["white_gazing", "distance"], w,h,l)
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)

   w=3
   h=2
   run_experiment(f"./new_Experiment_1Results_{w}X{h}", 20, evals, ["white_gazing", "avg_speed"], w,h,l)
   run_experiment(f"./new_Experiment_2Results_{w}X{h}", 20, evals, ["white_gazing", "distance"], w,h,l)
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)


   w=3
   h=3
   run_experiment(f"./new_Experiment_1Results_{w}X{h}", 20, evals, ["white_gazing", "avg_speed"], w,h,l)
   run_experiment(f"./new_Experiment_2Results_{w}X{h}", 20, evals, ["white_gazing", "distance"], w,h,l)

   w=4
   h=3
   run_experiment(f"./new_Experiment_1Results_{w}X{h}", 20, evals, ["white_gazing", "avg_speed"], w,h,l)
   run_experiment(f"./new_Experiment_2Results_{w}X{h}", 20, evals, ["white_gazing", "distance"], w,h,l)
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)

   w=4
   h=4
   run_experiment(f"./new_Experiment_1Results_{w}X{h}", 20, evals, ["white_gazing", "avg_speed"], w,h,l=0)
   run_experiment(f"./new_Experiment_2Results_{w}X{h}", 20, evals, ["white_gazing", "distance"], w,h,l=0)
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)'''
if __name__ == '__main__':
    main()