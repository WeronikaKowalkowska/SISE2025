import sys
import numpy as np
import time
from collections import deque
from bfs import *

# Celowy stan puzzli 15
GOAL_STATE = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 0)
)

MOVES = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

def main():
    if len(sys.argv) < 6:
        print("Not enough arguments.")
        sys.exit(1)

    strategy = sys.argv[1]
    parameter = sys.argv[2]
    start_file = sys.argv[3]
    end_file = sys.argv[4]
    additional_info_file = sys.argv[5]

    start_time = time.time()

    def read_matrix_from_file(filename):
        with open(filename, 'r') as file:
            rows, cols = map(int, file.readline().strip().split())
            matrix = [list(map(int, file.readline().strip().split())) for _ in range(rows)]
        return matrix

    start_matrix = read_matrix_from_file(start_file)
    start_state = tuple(tuple(row) for row in start_matrix)
    print(start_state)
    #start_state = start_matrix
    result = bfs(start_state)
    print(result)
    with open(end_file, 'w') as f:
        f.write(str(result))

    end_time = time.time()
    runtime = end_time - start_time

    # if path is not None:
    #     with open(end_file, 'w') as f:
    #         f.write(''.join(path) + '\n')
    #     with open(additional_info_file, 'w') as f:
    #         f.write(f"{len(path)}\n")
    #         f.write(f"{visited}\n")
    #         f.write(f"{processed}\n")
    #         f.write(f"{max_depth}\n")
    #         f.write(f"{runtime:.3f}\n")
    # else:
    #     with open(end_file, 'w') as f:
    #         f.write("No solution\n")

# def bfs(start):
#     queue = deque()
#     visited = set()
#     parent = {}
#     move_record = {}
#
#     queue.append((start, 0))  # (state, depth)
#     visited.add(start)
#     parent[start] = None
#     move_record[start] = None
#
#     max_depth = 20
#     processed = 0
#
#     while queue:
#         current, depth = queue.popleft()
#         processed += 1
#         max_depth = max(max_depth, depth)
#
#         if current == GOAL_STATE:
#             return reconstruct_path(parent, move_record, current), len(visited), processed, max_depth
#
#         for move, (dx, dy) in MOVES.items():
#             new_state = make_move(current, dx, dy)
#             if new_state and new_state not in visited:
#                 visited.add(new_state)
#                 parent[new_state] = current
#                 move_record[new_state] = move
#                 queue.append((new_state, depth + 1))
#
#     return None, len(visited), processed, max_depth
#
# def make_move(state, dx, dy):
#     state_list = [list(row) for row in state]
#     x=0
#     y=0
#     for i in range(dx):
#         for j in range(dy):
#             if state_list[i][j] == 0:
#                 x, y = i, j
#                 break
#
#     new_x, new_y = x + dx, y + dy
#     if 0 <= new_x < dx and 0 <= new_y < dy:
#         # Swap 0 with target
#         state_list[x][j], state_list[new_x][new_y] = state_list[new_x][new_y], state_list[x][j]
#         return tuple(tuple(row) for row in state_list)
#     return None
#
# def reconstruct_path(parent, move_record, end_state):
#     path = []
#     while parent[end_state]:
#         path.append(move_record[end_state])
#         end_state = parent[end_state]
#     return path[::-1]
#
if __name__ == "__main__":
    main()
