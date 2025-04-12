import sys
import numpy as np
import time
from collections import deque
from bfs import *
from dfs import *

#możliwe ruchy
direction_map = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

#kolojność przeszukowania ruchów
moves = {}

def main():
    global moves
    if len(sys.argv) < 6:
        print("Not enough arguments.")
        sys.exit(1)

    # odczytanie argumentów
    strategy = sys.argv[1]  # pierwszy argument - wybór strategii
    parameter = sys.argv[2]  # drugi argument - dodatkowy parament dla strategii
    start_file = sys.argv[3]  # trzeci argument - plik z układem początkowym
    end_file = sys.argv[4]  # czwarty argument - plik z rozwiązaniem
    additional_info_file = sys.argv[5]  # piąty argument - plik z dodatkowymi informacjami o obliczeniach

    start_time = time.time()   #początek pomiaru czasu

    def read_matrix_from_file(filename):
        matrix = []
        rows = 0
        columns = 0
        file = open(filename, 'r')
        first_line = file.readline()
        row_col = first_line.strip().split()
        rows = int(row_col[0])
        cols = int(row_col[1])
        for i in range(rows):
            line = file.readline().strip()
            values = list(map(int, line.split()))
            matrix.append(values)
        return matrix

    start_matrix = read_matrix_from_file(start_file)
    start_state = tuple(tuple(row) for row in start_matrix)


    if strategy == "bfs":
        moves = {direction: direction_map[direction] for direction in parameter}
        path, visited, processed, max_depth  = bfs(start_state, moves)
    if strategy == "dfs":
        moves = {direction: direction_map[direction] for direction in parameter}
        path, visited, processed, max_depth = dfs(start_state, moves)
    # if strategy == "astr":
    #  # dopisać

    end_time = time.time()
    runtime = end_time - start_time

    if path is not None:
        with open(end_file, 'w') as f:
            f.write(''.join(path) + '\n')
        with open(additional_info_file, 'w') as f:
            f.write(f"{len(path)}\n")
            f.write(f"{visited}\n")
            f.write(f"{processed}\n")
            f.write(f"{max_depth}\n")
            f.write(f"{runtime:.3f}\n")
    else:
        with open(end_file, 'w') as f:
            f.write("No solution\n")

#python3 main.py bfs RDUL 4x4_01_0001.txt 4x4_01_0001_bfs_rdul_sol.txt 4x4_01_0001_bfs_rdul_stats.txt
if __name__ == "__main__":
    main()
