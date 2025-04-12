from helpper import *
import heapq

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

def manh(new_state):


def hamm(new_state):




def astar (start, moves, h):
    open_priority_queue = []   #otwarte stany jako heapq
    visited= set()  #odwiedzone stany
    parent = {}  # rodzic bierzącego stanu
    move_record = {}  # jaki ruch doprowadził do bierzącego stanu
    f = None    # koszt dotarcia od węzła początkowego do celu przez bierzący węzeł

    heapq.heappush(open_priority_queue, 0, start) #pozycja startowa jest dodana do stanów otwartych z priorytetem zero
    parent[start] = None
    move_record[start] = None

    max_depth = 0  # maksymalna głębokość
    processed = 0  # licznik przetworzonych węzłów

    while open_priority_queue:
        current, depth = heapq.heappop(open_priority_queue)
        if current == goal_state:
            return reconstruct_path(parent, move_record, current), len(visited), processed, max_depth
        visited.add(current)
        processed += 1
        max_depth = max(max_depth, depth)
        for move in moves:
            dx, dy = moves[move]
            new_state = make_move(current, dx, dy)  # nowy stan na podstawie ruchu
            if new_state is not None:  # jeżeli nowy stan istnieje
                if new_state not in visited:  # jeżeli nowy stan nie był już odwiedzony
                    if h == manh:
                        f = processed + manh(new_state)
                    else:
                        f = processed + hamm(new_state)
                    if new_state not in open_priority_queue:
                        parent[new_state] = current  # zapisujemy poprzedni stan jako rodzica
                        move_record[new_state] = move  # zapisujemy ruch, jaki nas doprowadził do nowego stanu
                        heapq.heappush(open_priority_queue, f, new_state)
                    else:
                        index = open_priority_queue.index(new_state)    #zwraca indeks elementu new_state
                        if open_priority_queue[index][0]  > f:      #pod indeksem 0 jest priorytet
                            heapq.update_priority(open_priority_queue, f, index)

    return None, len(visited), processed, max_depth

