import heapq
import math
import time

# Defines a single node on the grid with specific properties
class Node:
    def __init__(self, x, y, g=float('inf'), h=0, parent=None):
        self.x = x  # x-coordinate of the node
        self.y = y  # y-coordinate of the node
        self.g = g  # cost from the start node to the current node
        self.h = h  # heuristic cost from the current node to the target node
        self.f = g + h  # total cost of the node
        self.parent = parent  # parent node of the current node (used to reconstruct the path)

    # Less than operator overload for node comparison based on f cost
    def __lt__(self, other):
        return self.f < other.f or (self.f == other.f and self.h < other.h)

def manhattan_heuristic(current, target):
    dx = abs(current.x - target.x)
    dy = abs(current.y - target.y)
    return dx + dy - min(dx, dy) * (math.sqrt(2) - 2)

def euclidean_distance(current, target):
    dx = abs(current.x - target.x)
    dy = abs(current.y - target.y)
    return math.sqrt(dx * dx + dy * dy)

def diagonal_distance(current, target):
    dx = abs(current.x - target.x)
    dy = abs(current.y - target.y)
    return max(dx, dy)

# Return all nodes that are within the visibility limit
def get_visible_area(node, grid, visibility_limit=10):
    visible_area = []
    # iterate through each cell within the visibility limit
    for dx in range(-visibility_limit, visibility_limit + 1):
        for dy in range(-visibility_limit, visibility_limit + 1):
            new_x, new_y = node.x + dx, node.y + dy
            # check if the cell is within the grid bounds
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[new_x]):
                visible_area.append((new_x, new_y))
    return visible_area

# Get valid neighbors of the current node
def get_neighbors(node, grid, open_set, visible_area):
    neighbors = []
    # define directions for 8-connected grid (4 orthogonal + 4 diagonal)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for dx, dy in directions:
        new_x, new_y = node.x + dx, node.y + dy
        # check if the neighbor is in the visible area and is not a wall
        if (new_x, new_y) in visible_area and grid[new_x][new_y] != 'W':
            # diagonal moves have a cost of sqrt(2), orthogonal moves have a cost of 1
            cost = math.sqrt(2) if dx != 0 and dy != 0 else 1
            # if the neighbor is in the open set, retrieve it, otherwise create a new node
            if (new_x, new_y) in open_set:
                neighbors.append((open_set[(new_x, new_y)], cost))
            else:
                neighbors.append((Node(new_x, new_y), cost))
    return neighbors

# Reconstruct the path from start to target
def reconstruct_path(node):
    path = []
    total_distance = 0
    # iterate from the target node to the start node
    while node:
        path.append((node.x, node.y))
        if node.parent:
            dx = node.x - node.parent.x
            dy = node.y - node.parent.y
            # calculate the total distance using the Euclidean distance formula
            total_distance += math.sqrt(dx*dx + dy*dy)
        node = node.parent
    return list(reversed(path)), len(path), total_distance

# A* search algorithm with limited visibility
def a_star(grid, start_node, target_node, visibility_limit=10):
    computation_counter = 0  # to keep track of how many computations are made
    # initialize the start node
    start_node.g = 0
    start_node.h = euclidean_distance(start_node, target_node)
    start_node.f = start_node.g + start_node.h
    # the open set keeps track of nodes to be evaluated
    open_set = {(start_node.x, start_node.y): start_node}
    open_heap = [start_node]  # a heap version of the open set for efficient minimum f value retrieval
    closed_set = set()  # the closed set keeps track of nodes already evaluated

    while open_set:  # while there are nodes to be evaluated
        current = heapq.heappop(open_heap)  # node in open set with the lowest f(x) value
        if (current.x, current.y) in closed_set:  # skip the current node if it's already evaluated
            continue

        visible_area = get_visible_area(current, grid, visibility_limit)  # determine the visible area
        # time.sleep(0.001)  # Add delay to simulate the robot scanning the area

        # if the current node is the target, then we found a path
        if current.x == target_node.x and current.y == target_node.y:
            return reconstruct_path(current), computation_counter

        closed_set.add((current.x, current.y))  # add the current node to the closed set

        # for each neighbor of the current node
        for neighbor, cost in get_neighbors(current, grid, open_set, visible_area):
            tentative_g = current.g + cost  # calculate the tentative g value for the neighbor
            # if the neighbor is not in the open set or if the tentative g value is less than the previous g value
            if (neighbor.x, neighbor.y) not in open_set or tentative_g < open_set[(neighbor.x, neighbor.y)].g:
                neighbor.g = tentative_g
                neighbor.h = euclidean_distance(neighbor, target_node)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current

                open_set[(neighbor.x, neighbor.y)] = neighbor  # add the neighbor to the open set
                heapq.heappush(open_heap, neighbor)  # add the neighbor to the open heap
                computation_counter += 1  # increment the computation counter

    return None, computation_counter  # if there's no path, return None

# Read the map from a file
def read_map(file_path):
    grid = []
    with open(file_path, 'r') as file:
        for line in file:  # read each line in the file
            grid.append(list(line.strip()))  # add each line to the grid as a list of characters
    return grid

# Find the starting point and the target on the grid
def find_start_and_target(grid):
    start = None
    target = None
    # iterate through the grid
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 'S':  # if the cell contains 'S', it's the start point
                start = Node(i, j)
            elif grid[i][j] == 'T':  # if the cell contains 'T', it's the target
                target = Node(i, j)
    return start, target

# Print the path on the grid
def print_grid(grid, path):
    # mark the path on the grid with '*'
    for x, y in path:
        if grid[x][y] != 'S' and grid[x][y] != 'T':
            grid[x][y] = '*'
    # print the grid
    for row in grid:
        print(''.join(row))

# Write the grid and the path to a file
def write_to_file(file_path, grid, path, results):
    with open(file_path, 'w') as file:
        file.write('\n'.join(results))  # write the results
        file.write('\n')
        # mark the path on the grid with '*'
        for x, y in path:
            if grid[x][y] != 'S' and grid[x][y] != 'T':
                grid[x][y] = '*'
        # write the grid to the file
        for row in grid:
            file.write(''.join(row))
            file.write('\n')

# Main execution
file_path = 'round_3.txt'
grid = read_map(file_path)
start, target = find_start_and_target(grid)

# check if both start and target points are found
if start and target:
    # perform the A* search
    result, computations = a_star(grid, start, target)
    # if a path is found
    if result:
        path, steps, total_distance = result
        results = [
            "Path found!",
            f"Number of computations: {computations}",
            f"Steps: {steps}",
            f"Total distance: {total_distance}"
        ]
        print('\n'.join(results))  # print the results
        print_grid(grid, path)  # print the grid with the path
        write_to_file('round_3_result.txt', grid, path, results)  # write the results and the path to the file
    else:
        print("No path found from start to target")  # if no path is found
else:
    print("No start or target found")  # if no start or target point is found
