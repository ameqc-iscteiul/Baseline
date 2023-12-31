import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from abrain import Point
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from ete3 import Tree, TreeStyle, NodeStyle,TextFace




def create_genealogy_tree(pairs: list, save_path: str, final_grid):

    g=final_grid['genome']
    final_grid.drop(columns=['genome'], inplace=True)
    final_grid['genome'] = g

    # Create a dictionary to store the nodes
    nodes = {}
    # Create the nodes from the pairs
    for child_id, parent_id in pairs:
        # Create the parent node if it doesn't exist
        if parent_id not in nodes:
            nodes[parent_id] = Tree(name=str(parent_id))
        # Create the child node if it doesn't exist
        if child_id not in nodes:
            nodes[child_id] = Tree(name=str(child_id))
        # Add the child to the parent
        nodes[parent_id].add_child(nodes[child_id])
    
    #print('nodes: ',nodes)
    # Find all the root nodes (the ones without a parent)
    roots = [node for node in nodes.values() if not node.up]
        
    print("roots",len(roots))
    
    # Create a TreeStyle object to customize the tree layout
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.scale =  120
    #ts.mode = 'c'
    
    # Create a NodeStyle object to customize the node appearance
    ns = NodeStyle()
    ns['size'] = 10

    # Render each tree and save it as a separate image
    for i, root in enumerate(roots):
        # Apply the NodeStyle to all nodes in the tree
        for n in root.traverse():
            n.set_style(ns)
            n.add_face(TextFace(n.name), column=0, position='branch-right')
        
        # Find individuals descended from the current root that are in Final Grid
        descendants = [int(ind_id) for ind_id in root.get_leaf_names() if int(ind_id) in final_grid['id'].values]
        # Create a new DataFrame containing only the descendants
        root_df = final_grid[final_grid['id'].isin(descendants)].copy()
        root_df.sort_values(by='id', ascending=True).to_csv(f'{save_path}/root_{root.name}.csv', index=False)

        '''# Render the tree and save it as an image
        root.render(f'{save_path}/root_{i}.png', tree_style=ts)'''



def final_grid_ancestry(pairs: list, final_grid_df , save_path):
    final_ids = final_grid_df['id'].tolist()

    # Create a dictionary to store the parent of each child
    parents = {}
    for child_id, parent_id in pairs:
        parents[child_id] = parent_id
    # Create a set to store the IDs to keep
    keep = set(final_ids)
    # Add the ancestors of the final IDs to the keep set
    for final_id in final_ids:
        current_id = final_id
        while current_id in parents:
            current_id = parents[current_id]
            keep.add(current_id)
    
    # Create a new list of pairs that only includes the IDs to keep
    new_pairs = [(child_id, parent_id) for child_id, parent_id in pairs if child_id in keep and parent_id in keep]
    create_genealogy_tree(new_pairs, save_path, final_grid_df)


def contrast_filter(pixel_value, contrast_factor=0.5, bound_value=0.5):
    if pixel_value < bound_value:
        new_value = pixel_value * (1 - contrast_factor)
    else:
        new_value = pixel_value
    return max(0.0, min(1.0, new_value))  # Ensure the result stays within the range [0, 1]

def center_and_normalize(signal):
    # Center the signal by subtracting its mean
    centered_signal = signal - np.mean(signal)
    # Normalize the centered signal by dividing by its standard deviation
    normalized_signal = centered_signal / np.std(centered_signal)
    return normalized_signal

def save_grid(grid, run_folder, name):
    os.makedirs(run_folder, exist_ok=True) 
    grid_pop = []
    for _, element in enumerate(grid):
        #print("element.fitness",element.fitness[0])
        ind = { "id": element.id(), "parents": element.genome.parents(),
                "fitnesses": element.fitness[0],
                "descriptors": element.descriptors,
                "genome": element.genome.to_json()}
        grid_pop.append(ind)
    df = pd.DataFrame(grid_pop)
    df = df.sort_values(by="fitnesses", ascending=False)
    df.to_csv(Path(run_folder).joinpath(f"{name}_grid.csv"), index=False)

    # Create a box plot of the 'fitness' column
    plt.figure(figsize=(8, 6))  # Optional: Set the size of the plot
    plt.boxplot(df['fitnesses'])
    plt.title('Box Plot of Fitness Scores')
    plt.ylabel('Fitness')
    plt.xlabel('Individuals')
    # Save the plot to the specified file
    plt.savefig(Path(run_folder).joinpath(f"{name}_BoxPlot.png"))
    plt.close()

    return df

def target_area_distance(position, target_position, target_side_len):
    n = target_side_len / 2
    min_x,max_x = target_position[0] - n, target_position[0] + n
    min_y, max_y = target_position[1] - n, target_position[1] + n
    '''if (min_x <= position[0] <= max_x and min_y <= position[1] <= max_y):
        return 0.0  # Position is inside the square'''
    distance_x = max(min_x - position[0], position[0] - max_x)
    distance_y = max(min_y - position[1], position[1] - max_y)
    distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
    return distance

'''def create_genealogical_tree(genealogy_list):
    tree = {}
    for son, father in genealogy_list:
        if father not in tree:
            tree[father] = {}
        tree[father][son] = tree.get(son, {})
    return tree'''






def distribute_points(y, w, h):
    points = []
    center_x = (w - 1) / 2
    center_z = (h - 1) / 2
    max_abs_value = max(center_x, center_z)

    for i in range(h):
        for j in range(w):
            x = (j - center_x) / max_abs_value
            z = (center_z - i) / max_abs_value
            points.append(Point(x, y, z))

    return points


