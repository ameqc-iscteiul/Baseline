#!/usr/bin/env python3
import pickle
from evolution import evolution, Options
from replicates_experiment import plot_seed_results
import matplotlib.pyplot as plt

def run_experiment(result_folder_name, batch_size, budget_size, features, fitness, numb_levels, init_level, robot_type, grid_size, make_final_videos, init_mut, threads):
    o = Options()

    o.base_folder = result_folder_name

    o.threads = threads
    o.batch_size = batch_size
    o.budget = budget_size

    o.numb_levels = numb_levels
    o.level = init_level

    o.grid_size = grid_size
    o.tournament = 3
    o.descriptor_names = features
    o.fitness_name = fitness
    
    o.robot_type = robot_type
    o.initial_mutations = init_mut
    o.vision_w = 4
    o.vision_h = 4

    o.make_final_videos = make_final_videos

    return evolution(o)

def plot_comparative_results(values_list, names, save_path, y_label):
   plt.figure(figsize=(10, 6))
   for i, values in enumerate(values_list):
      plt.plot(values, label=f"Experiment {names[i]}")
   plt.xlabel('Evaluations')
   plt.ylabel(y_label)
   plt.title(f'{y_label} Across Replicates')
   plt.grid(True)
   plt.legend()

   plt.savefig(f'{save_path}/{y_label}_plot.png')

def main():
   result_folder_name = f'baseline/results/Comparing_Experiments4/'
   threads=25
   batch_size=25
   budget_size=200000
   numb_levels = 1
   init_level = 6
   robot_type = 1
   grid_size =  16

   avgs=[]
   maxs=[]
   QDs=[]
   stats = run_experiment(f'{result_folder_name}/E_Z_peaks_Avg-G-0-6mut', batch_size, budget_size, ['edges','z_peaks_avg'], 'new', numb_levels, init_level, robot_type, grid_size, True, 3, threads)
   avgs.append(stats['Avg'].tolist())
   maxs.append(stats['Max'].tolist())
   QDs.append(stats['QDs'].tolist())

   '''stats = run_experiment(f'{result_folder_name}/E_Z_peaks_Avg-G-0-6mut', batch_size, budget_size, ['edges','z_descriptor'], 'new', numb_levels, init_level, robot_type, grid_size, True, 3, threads)
   avgs.append(stats['Avg'].tolist())
   maxs.append(stats['Max'].tolist())
   QDs.append(stats['QDs'].tolist())'''

   

   

   '''stats = run_experiment(f'{result_folder_name}/C_Z_peaks_Avg-G-0-6mut', batch_size, budget_size, ['complexity','z_peaks_avg'], 'new', numb_levels, init_level, robot_type, grid_size, True, 6, threads)
   avgs.append(stats['Avg'].tolist())
   maxs.append(stats['Max'].tolist())
   QDs.append(stats['QDs'].tolist())   

   stats = run_experiment(f'{result_folder_name}/C_Z-G-0-6mut', batch_size, budget_size, ['complexity','z_descriptor'], 'new', numb_levels, init_level, robot_type, grid_size, True, 6, threads)
   avgs.append(stats['Avg'].tolist())
   maxs.append(stats['Max'].tolist())
   QDs.append(stats['QDs'].tolist())'''


   names=['E_Z_peaks_Avg']
          #,'C_Z_peaks_Avg']
   # #, 'E_Z', 'C_Z']

   plot_comparative_results(avgs, names, result_folder_name, 'Average_Fitness')
   plot_comparative_results(maxs, names, result_folder_name, 'Max_Fitness')
   plot_comparative_results(QDs, names, result_folder_name, 'QD_Score')   


   with open(f'baseline/results/Comparing_Experiments4/avgs.pkl', 'wb') as f:
      pickle.dump(avgs, f)
   with open(f'baseline/results/Comparing_Experiments4/maxs.pkl', 'wb') as f:
      pickle.dump(maxs, f)
   with open(f'baseline/results/Comparing_Experiments4/QDs.pkl', 'wb') as f:
      pickle.dump(QDs, f)
   
if __name__ == '__main__':
    main()