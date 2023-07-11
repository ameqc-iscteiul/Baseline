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
   
   o.scenario_level=l

   evolution(o)

def main():
   print("Start")
   evals=5000

   w=2
   h=2
   for i in range(1,5):
      run_experiment(f"./Experiment_Results_2X2", 20, evals,w,h,l=i)

   w=3
   h=2
   for i in range(1,5):
      run_experiment(f"./Experiment_Results_3X2", 20, evals,w,h,l=i)

   w=3
   h=3
   for i in range(1,5):
      run_experiment(f"./Experiment_Results_3X3", 20, evals,w,h,l=i)

   w=4
   h=3
   for i in range(1,5):
      run_experiment(f"./Experiment_Results_4X3", 20, evals,w,h,l=i)

   w=4
   h=4
   for i in range(1,5):
      run_experiment(f"./Experiment_Results_4X4", 20, evals,w,h,l=i)

   









if __name__ == '__main__':
    main()