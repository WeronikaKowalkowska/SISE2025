from collections import deque

#celowy stan puzzli 15
goal_state = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 0)
)


def bfs(start, moves):
    queue = deque()     #otwarte stany
    visited = set()      #odwiedzone stany
    parent = {}         #rodzic bierzącego stanu
    move_record = {}    #jaki ruch doprowadził do bierzącego stanu

    queue.append((start, 0))  #pozycja startowa jest dodana do stanów otwartych z głębokością zero
    visited.add(start)  #pozycja startowa jest dodana do stanów odwiedzonych
    parent[start] = None
    move_record[start] = None

    max_depth = 0   #maksymalna głębokość
    processed = 0   #licznik przetworzonych węzłów

    while queue:
        current, depth = queue.popleft()    #pobieramy stan i jego głębokość z kolejki stanów otwartych
        processed += 1
        max_depth = max(max_depth, depth)

        if current == goal_state:
            return reconstruct_path(parent, move_record, current), len(visited), processed, max_depth

        for move in moves:
            dx, dy = moves[move]
            new_state = make_move(current, dx, dy)  #nowy stan na podstawie ruchu
            if new_state is not None:      #jeżeli nowy stan istnieje
                if new_state not in visited:     #jeżeli nowy stan nie był już odwiedzony
                    visited.add(new_state)      #dodajemy do stanów odwiedzonych
                    parent[new_state] = current     #zapisujemy poprzedni stan jako rodzica
                    move_record[new_state] = move   #zapisujemy ruch jaki nas doprowadzil do nowego stanu
                    queue.append((new_state, depth + 1))    #dodajemy do listy stanów otwartych z większą głębokością niż rodzic

    return None, len(visited), processed, max_depth

#metoda zwracająca pozycję zera z przekazanej planszy state
def find_zero(state):
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == 0:
                return i, j

def make_move(state, dx, dy):
    x, y = find_zero(state) #pozycja zera

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