#!/usr/bin/env python3
import os
import pickle
from evolution import evolution, Options
import matplotlib.pyplot as plt

import pandas as pd

   
def experiment(name, seed, scaff_type):
      o = Options()
      o.base_folder = name
      o.seed=seed
      if scaff_type==1:
         o.level = 6
      else: 
         o.level = 0
      o.numb_levels = scaff_type
   
      
      o.descriptor_names = ['edges','z_peaks_avg']
      o.fitness_name = "new"
     
      o.threads=25
      o.batch_size = 25
      o.budget = 200000
      o.robot_type = 1
      o.grid_size = 15
      o.tournament = 3
      o.initial_mutations = 3
      o.vision_w = 4
      o.vision_h = 4
      o.make_final_videos=True

      return evolution(o)

def plot_seed_results(values_list, seeds, save_path, y_label):
   plt.figure(figsize=(12, 8))
   for i, values in enumerate(values_list):
      plt.plot(values, label=f"Replicate {seeds[i]}")
   plt.xlabel('Evaluations')
   plt.ylabel(y_label)
   plt.title(f'{y_label} Across Replicates')
   plt.grid(True)
   plt.legend()
   
   plt.savefig(f'{save_path}/{y_label}_plot.png')

def main():
   path = f'baseline/results'
   seeds=[5,6,7,8]
   scaffolding_degree = [1, 3, 7]
   for scaff_type in scaffolding_degree:
      save_path=f'{path}/Scaffolding degree_{scaff_type}'
      os.makedirs(save_path, exist_ok=True) 
      seed_avgs=[]
      seed_maxs=[]
      seed_QDs=[]
      for seed in seeds:
         stats = experiment(f'{save_path}/seed_{seed}', seed, scaff_type)
         seed_avgs.append(stats['Avg'].tolist())
         seed_maxs.append(stats['Max'].tolist())
         seed_QDs.append(stats['QDs'].tolist())

      
      plot_seed_results(seed_avgs, seeds, save_path, 'Average_Fitness')
      plot_seed_results(seed_maxs, seeds, save_path, 'Max_Fitness')
      plot_seed_results(seed_QDs, seeds, save_path, 'QD_Score')

      with open(f'{save_path}/avgs.pkl', 'wb') as f:
         pickle.dump(seed_avgs, f)
      with open(f'{save_path}/maxs.pkl', 'wb') as f:
         pickle.dump(seed_maxs, f)
      with open(f'{save_path}/QDs.pkl', 'wb') as f:
         pickle.dump(seed_QDs, f)
         

if __name__ == '__main__':
    main()