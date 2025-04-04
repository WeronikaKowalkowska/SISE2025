import sys
import numpy as np
import time


# uruchomienie: python main.py [parametry]
def main():
    #sprawdzenie, czy podano odpowiednią liczbę argumentów
    if len(sys.argv) < 5:
        print("Not enough arguments.")
        sys.exit(1)

    #odczytanie argumentów
    strategy = sys.argv[1]  #pierwszy argument - wybór strategii
    parameter = sys.argv[2] #drugi argument - dodatkowy parament dla strategii
    start_file = sys.argv[3]  #trzeci argument - plik z układem początkowym
    end_file = sys.argv[4] #czwarty argument - plik z rozwiązaniem
    additional_info_file = sys.argv[5] #piąty argument - plik z dodatkowymi informacjami o obliczeniach

    '''moves={
        'L':
        'R':
        'U':
        'D':
        }'''

    start_time = time.time()  #początek pomiaru czasu

    solution_length = 0
    vsited_states_count = 0  # zamkniete
    processed_states_count = 0  # otwarte
    maximum_recursion_deapth = 0
    runtime = 0

    #metoda zwracająca pozycję zera z przekazanej planszy state
    def find_zero(state):
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == 0:
                    return i, j

    end_time = time.time()  #koniec pomiaru czasu
    runtime = end_time - start_time  #czas wykonania obliczeń w sekundach

if __name__ == "__main__":
    main()
