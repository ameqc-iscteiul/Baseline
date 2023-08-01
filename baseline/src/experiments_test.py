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
   o.grid_size=16
   evolution(o)

def main():
   print("Start")
   vision_dims=[(4,4)]
   for w,h in vision_dims: 
      run_experiment(f"baseline/Testing_level_change/{w}X{h}_0_to_3", 20, 4000, ["distance", "white_gazing"], w,h, 2, 0)


if __name__ == '__main__':
    main()