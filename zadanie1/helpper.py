#celowy stan puzzli 15
goal_state = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 0)
)

#metoda zwracająca pozycję liczby z przekazanej planszy state
def find_number(state, number):
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == number:
                return i, j

def make_move(state, dx, dy):
    x, y = find_number(state, 0) #pozycja zera

    new_x = x + dx  #pozycje po przesunięciu
    new_y = y + dy

    if 0 <= new_x < len(state) and 0 <= new_y < len(state[0]):      #sprawdzamy czy wartości po przesunięciu nie wyszły poza planszę
        new_state = []
        for row in state:
            new_row = list(row) #kopiujemy poprzedni stan zamieniając krotkę na listę
            new_state.append(new_row)

        #przesuwamy zero na wybranną pozycje
        new_state[x][y] = state[new_x][new_y]
        new_state[new_x][new_y] = state[x][y]

        return tuple(tuple(row) for row in new_state) #wztracamy nowy stan jako tuple of tuples

    return None #jeżeli ruch wyjdzie poza plansze

#metoda do odtworzenia ścieżki ruchów, które doprowadziły od stanu początkowego do stanu końcowego
def reconstruct_path(parent, move_record, end_state):
    path = []
    while parent[end_state]:    #dopóki stan ma rodzica, czyli dopóki nie jest to stan początkowy, cofamy się
        path.append(move_record[end_state]) #zapisujemy ruch, który doprowadził do bierzącego stanu
        end_state = parent[end_state]   #cofamy się do rodzica
    path.reverse()  #oswracamy ściężkę, żeby mieć wsztsko od początku w dobrzej kolejności
    return path