import sys
import numpy as np
import time

from bfs import *


MOVES = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# uruchomienie: python main_poprzedni.py [parametry]
def main():
    #sprawdzenie, czy podano odpowiednią liczbę argumentów
    if len(sys.argv) < 6:
        print("Not enough arguments.")
        sys.exit(1)

    #odczytanie argumentów
    strategy = sys.argv[1]  #pierwszy argument - wybór strategii
    parameter = sys.argv[2] #drugi argument - dodatkowy parament dla strategii
    start_file = sys.argv[3]  #trzeci argument - plik z układem początkowym
    end_file = sys.argv[4] #czwarty argument - plik z rozwiązaniem
    additional_info_file = sys.argv[5] #piąty argument - plik z dodatkowymi informacjami o obliczeniach

    start_time = time.time()  #początek pomiaru czasu

    solution_length = 0
    visited_states_count = 0  # zamkniete
    processed_states_count = 0  # otwarte
    maximum_recursion_deapth = 0
    runtime = 0

    #metoda zwracająca pozycję zera z przekazanej planszy state
    def find_zero(state):
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == 0:
                    return i, j

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
    start_state = tuple(tuple(row) for row in start_matrix) #dwuwymiarową listę w krotkę krotek


    path, visited, processed, max_depth = bfs(graph, start_state)

    end_time = time.time()  #koniec pomiaru czasu
    runtime = end_time - start_time  #czas wykonania obliczeń w sekundach

if __name__ == "__main__":
    main()
