from helpper import *
import heapq

#python3 main.py astar hamm 4x4_01_0001.txt 4x4_01_0001_astar_hamm_sol.txt 4x4_01_0001_astar_hamm_stats.txt
#python3 main.py astar manh 4x4_01_0001.txt 4x4_01_0001_astar_manh_sol.txt 4x4_01_0001_astar_manh_stats.txt

#----PSEUDOKOD----
'''def astar(graph, start):
    open_priority_queue = heapq()
    visited= set()
    open_priority_queue.put(start, 0)
    while not open_priority_queue.empty():
        current = open_priority_queue.get()
        if current==goal_state:
            return True #success
        visited.add(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                f = g(neighbor) + h(neighbor, graph)
                if neighbor not in open_priority_queue:
                    open_priority_queue.put(neighbor, f)
                else:
                    if priority[neighbor] > f:
                        open_priority_queue.update_priority(neighbor, f)
    return False #failure'''


'''f(n) = g(n) + h(n)
g(n) – koszt rzeczywisty od startu do węzła n -> liczba przesunięć wykonanych od stanu początkowego do aktualnego stanu
h(n) – heurystyka, oszacowany koszt z węzła n do celu -> ilu ruchów potrzeba jeszcze, żeby dojść do rozwiązania'''

#tak napisałam f-je w pliku heapd.pyi
'''def update_priority(open_priority_queue, f, index):
    old_priority, state = open_priority_queue[index]
    open_priority_queue[index] = (f, state)
    heapq.heapify(open_priority_queue)  #odtworzenie struktury kopca'''

#metryka Manhattan
def manh(state):
    dx = 0
    dy = 0

    for i in range(len(state)*len(state)):   #przechodzimy się po wszystkich liczbach
        if i != 0:  #nie uwzględniamy pozycji zera
            x_goal, y_goal = find_number(goal_state, i)
            x, y = find_number(state, i)
            dx += abs(x - x_goal)
            dy += abs(y - y_goal)

    return dx+dy

#metryka Hamminga
def hamm(state):
    distance = 0

    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:    #nie uwzględniamy pozycji zera
                distance += 1
    return distance


def astar (start, moves, h):
    open_priority_queue = []   #otwarte stany jako heapq
    visited= set()  #odwiedzone stany
    parent = {}  # rodzic bierzącego stanu
    move_record = {}  # jaki ruch doprowadził do bierzącego stanu
    g = {start: 0}    # koszt dotarcia od węzła początkowego do stanu

    heapq.heappush(open_priority_queue, (0, start)) #pozycja startowa jest dodana do stanów otwartych z priorytetem zero
    parent[start] = None
    move_record[start] = None

    max_depth = 0  # maksymalna głębokość
    processed = 0  # licznik przetworzonych węzłów

    while open_priority_queue:
        f, current = heapq.heappop(open_priority_queue)
        if current == goal_state:
            return reconstruct_path(parent, move_record, current), len(visited), processed, max_depth
        visited.add(current)
        processed += 1
        for move in moves:
            dx, dy = moves[move]
            new_state = make_move(current, dx, dy)  # nowy stan na podstawie ruchu
            if new_state is not None:  # jeżeli nowy stan istnieje
                if new_state not in visited:  # jeżeli nowy stan nie był już odwiedzony
                    move_cost_g = g[current] + 1    #koszt g dla nowego stanu jest o jeden większy niż dla poprzedniego
                    if new_state not in g or move_cost_g < g[new_state] :  #jeżeli nowy stan jeszcze nie ma swojego kosztu g -> nie ma na liście stanów otwartych lub jeżeli była odnaleziona ścieżka o niższym koszcie dla stanu ze słownika g
                        g[new_state] = move_cost_g
                        if h == "manh":
                            f = move_cost_g + manh(new_state)
                        else:
                            f = move_cost_g + hamm(new_state)
                        heapq.heappush(open_priority_queue, (f, new_state))
                        parent[new_state] = current  # zapisujemy poprzedni stan jako rodzica
                        move_record[new_state] = move  # zapisujemy ruch, jaki nas doprowadził do nowego stanu
                        max_depth = max(max_depth, move_cost_g)

    return None, len(visited), processed, max_depth

