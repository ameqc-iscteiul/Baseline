import math

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