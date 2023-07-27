#!/usr/bin/env python3
from evolution import evolution, Options


def run_experiment(result_folder_name, batch_size, budget_size, features, w, h, l):
   o=Options()
   o.base_folder=result_folder_name
   o.batch_size=batch_size
   o.budget=budget_size
   o.tournament = 3
   o.initial_mutations = 2
   o.vision_w=w
   o.vision_h=h
   o.numb_levels=l
   o.descriptor_names=features
   evolution(o)

def main():
   print("Start")
   #Testing 3 dims. To in both grid scenarios
   evals=300
   vision_dims=[(4,4)]
   #for l in range(1,3):
   for w,h in vision_dims:
      run_experiment(f"./Experiment_Test/{w}X{h}", 10, evals, ["distance", "white_gazing"], w,h,4)
   
  

if __name__ == '__main__':
    main()