import math

from abrain import Point

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

def create_genealogical_tree(genealogy_list):
    tree = {}
    for son, father in genealogy_list:
        if father not in tree:
            tree[father] = {}
        if son not in tree:
            tree[son] = {}
        tree[father][son] = tree[son]
    return tree


def distribute_points2(y, w, h):
    points = []
    for i in range(h-1, -1, -1):
        for j in range(w):
            points.append(Point(j, y, i))
    return points
    

    
def distribute_points4(y, w, h):
    points = []
    center_x = (w - 1) / 2
    center_z = (h - 1) / 2

    for i in range(h):
        for j in range(w):
            x = j - center_x
            z = center_z - i
            points.append(Point(x, y, z))

    return points

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


def main():
    p = distribute_points(-1, 4, 4) 
    print(p)   

if __name__ == "__main__":
    main()
