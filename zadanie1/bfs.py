from helpper import *
from collections import deque

#python3 main.py bfs RDUL 4x4_01_0001.txt 4x4_01_0001_bfs_rdul_sol.txt 4x4_01_0001_bfs_rdul_stats.txt

#----PSEUDOKOD----
'''def bfs(graph, root):
    if root==success:
        return True
    open_queue = deque([root])
    closed_set = set()
    open_queue.add([root])
    while open_queue:
        state= open_queue.popleft()
        closed_set.add(state)
        for neighbor in graph[state]:
            if neighbor not in closed_set and neighbor not in open_queue:
                if graph[state]==success:
                    return True #success
                open_queue.append(neighbor)
    return False #failure'''

def bfs(start, moves):

    if start == goal_state:
        return [], 0, 0, 0

    open_queue = deque()     #otwarte stany
    visited = set()      #odwiedzone stany
    parent = {}         #rodzic bierzącego stanu
    move_record = {}    #jaki ruch doprowadził do bierzącego stanu

    open_queue.append((start, 0))  #pozycja startowa jest dodana do stanów otwartych z głębokością zero
    parent[start] = None
    move_record[start] = None

    max_depth = 0   #maksymalna głębokość
    processed = 0   #licznik przetworzonych węzłów

    while open_queue:
        current, depth = open_queue.popleft()    #pobieramy stan i jego głębokość z kolejki stanów otwartych
        visited.add(current)  # pozycja jest dodana do stanów odwiedzonych
        processed += 1
        max_depth = max(max_depth, depth)

        for move in moves:
            dx, dy = moves[move]
            new_state = make_move(current, dx, dy)  #nowy stan na podstawie ruchu
            if new_state is not None:      #jeżeli nowy stan istnieje
                if new_state not in visited and all(state != new_state for state, _ in open_queue):     #jeżeli nowy stan nie był już odwiedzony i że nie ma go na liście stanów otwartych
                    if new_state == goal_state:
                        parent[new_state] = current #zapisujemy poprzedni stan jako rodzica
                        move_record[new_state] = move #zapisujemy ruch, jaki nas doprowadził do nowego stanu
                        return reconstruct_path(parent, move_record, new_state), len(visited), processed, max_depth+1
                    parent[new_state] = current     #zapisujemy poprzedni stan jako rodzica
                    move_record[new_state] = move   #zapisujemy ruch, jaki nas doprowadził do nowego stanu
                    open_queue.append((new_state, depth+1))    #dodajemy do stanów otwartych z większą głębokością niż rodzic

    return None, len(visited), processed, max_depth
