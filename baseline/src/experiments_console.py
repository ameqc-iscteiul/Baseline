#!/usr/bin/env python3
from evolution import evolution, Options


def run_experiment(result_folder_name, batch_size, budget_size, features, fitness, numb_levels, init_level, robot_type, grid_size, make_final_videos, init_mut, threads):
    o = Options()
    o.base_folder = result_folder_name
    o.threads=threads
    o.batch_size = batch_size
    o.budget = budget_size
    o.robot_type = robot_type
    o.numb_levels = numb_levels
    o.level = init_level
    o.descriptor_names = features
    o.fitness_name = fitness
    o.grid_size = grid_size
    o.tournament = 3
    o.vision_w = 4
    o.vision_h = 4
    o.make_final_videos=make_final_videos
    o.initial_mutations=init_mut
    evolution(o)

def get_user_input(prompt, default=None, input_type=str):
    while True:
        user_input = input(prompt)
        if not user_input:
            if default is not None:
                return default
            else:
                print("Default value not allowed. Please provide a value.")
        else:
            try:
                return input_type(user_input)
            except ValueError:
                print("Invalid input. Please try again.")


def main():
    print("Start")
    folder_name = get_user_input("Enter result folder name:   ", "Default")
    result_folder_name = f'baseline/results/{folder_name}'
    threads = get_user_input("Enter number of threaths:   ", 10, int)

    batch_size = get_user_input("Enter batch size:   ", 10, int)
    budget_size = get_user_input("Enter budget size:   ", 100, int)

    feature = get_user_input("BD Pair: 1-White Gazing & Edges | 2-White Gazing & Complexity  | 3- White Gazing & z_descriptor | 4- White Gazing & z_peaks_avg :  ", 1, int)
    if feature == 1:
        features = ['white_gazing','edges']
    elif feature == 2:
        features = ['white_gazing','complexity']
    elif feature == 3:
        features = ['white_gazing','z_descriptor']
    elif feature == 4:
        features = ['white_gazing','z_peaks_avg']


    fitness = get_user_input("Enter fitness number: 1- Official_Fitness || 2-Brightness :  ", 1, int)
    if fitness == 1:
        fitness = 'new'
    elif fitness == 2:
        fitness = 'brightness'

    init_mut= get_user_input("Initial Mutations: ", 2, int)

    numb_levels = get_user_input("Enter number of levels:   ", 1, int)
    init_level = get_user_input("Enter initial level:   ", 0, int)
    robot_type = get_user_input("Enter Robot type: 0-Default or 1-Gecko:   ", 0, int) 
    grid_size = get_user_input("Enter Grid size:   ", 16, int) 
    make_final_videos=get_user_input("Make videos: 0-No : 1-Yes:   ", 0, bool) 
    run_experiment(result_folder_name, batch_size, budget_size, features, fitness,numb_levels, init_level, robot_type, grid_size, make_final_videos, init_mut, threads)


if __name__ == '__main__':
    main()
