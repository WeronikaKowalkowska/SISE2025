import argparse
import numpy as np
import time

'''moves={
    'L':
    'R':
    'U':
    'D':
    }'''

start_time = time.time()  # Początek pomiaru




solution_length=0
vsited_states_count=0 #zamkniete
processed_states_count=0 #otwarte
maximum_recursion_deapth=0
runtime=0

#metoda zwraca pozycję zera z przekazanej planszy state
def find_zero(state):
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i][j] == 0:
                return i, j



end_time = time.time()  # Koniec pomiaru
runtime = end_time - start_time  # Czas wykonania w sekundach

print(f"Czas wykonania: {runtime:.4f} sekundy")


def main():
    #tworzenie obiektu parsera
    parser = argparse.ArgumentParser(description="Opis działania programu.")

    #dodanie argumentów, które będą przekazywane z linii poleceń
    parser.add_argument('--matrix', type=str, help='Macierz do rozwiązania (np. a, b, c)', required=True)
    parser.add_argument('--iterations', type=int, help='Liczba iteracji do wykonania', required=False)
    parser.add_argument('--accuracy', type=float, help='Dokładność, do jakiej ma być osiągnięte rozwiązanie',
                        required=False)

    # Parsujemy argumenty
    args = parser.parse_args()

    # Możemy teraz używać tych argumentów
    print(f"Wybrana macierz: {args.matrix}")

    if args.iterations:
        print(f"Liczba iteracji: {args.iterations}")
    if args.accuracy:
        print(f"Dokładność: {args.accuracy}")

    # Tutaj możesz umieścić kod, który będzie korzystać z przekazanych argumentów
    # np. wybór odpowiedniej funkcji na podstawie 'args.matrix' itd.


if __name__ == "__main__":
    main()
