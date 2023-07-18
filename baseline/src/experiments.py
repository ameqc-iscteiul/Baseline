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
   
   evals=8000
   l=1
   w=4
   h=4
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)
   exit()

   w=2
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
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)

   w=4
   h=3
   run_experiment(f"./new_Experiment_1Results_{w}X{h}", 20, evals, ["white_gazing", "avg_speed"], w,h,l)
   run_experiment(f"./new_Experiment_2Results_{w}X{h}", 20, evals, ["white_gazing", "distance"], w,h,l)
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)

   w=4
   h=4
   run_experiment(f"./new_Experiment_1Results_{w}X{h}", 20, evals, ["white_gazing", "avg_speed"], w,h,l=0)
   run_experiment(f"./new_Experiment_2Results_{w}X{h}", 20, evals, ["white_gazing", "distance"], w,h,l=0)
   run_experiment(f"./new_Experiment_3Results_{w}X{h}", 20, evals, ["trajectory", "white_gazing"], w,h,l)
if __name__ == '__main__':
    main()