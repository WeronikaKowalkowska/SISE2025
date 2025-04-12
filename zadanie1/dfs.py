from helpper import *

#----PSEUDOKOD----
'''def dfs(graph, root):
    open_list = []
    visited = set()
    open_list.append(root)
    while len(open_list) != 0:
        current = open_list.pop()
        if current not in visited:
            if success == current:
                return True #success
            visited.add(current)
            for neighbor in reversed(graph[current]):
                open_list.append(neighbor)
    return False #failure'''

#wersja iteracyjna
def dfs (start, moves):
    open_list = []  #otwarte stany
    visited = set() #odwiedzone stany
    parent = {}  # rodzic bierzącego stanu
    move_record = {}  # jaki ruch doprowadził do bierzącego stanu

    open_list.append((start, 0))    #pozycja startowa jest dodana do stanów otwartych z głębokością zero
    parent[start] = None
    move_record[start] = None

    max_depth = 20  # maksymalna głębokość
    actual_max_depth = 0
    processed = 0  # licznik przetworzonych węzłów

    while open_list:
        current, depth = open_list.pop()
        if depth > max_depth:
            return None, len(visited), processed, max_depth
        if current not in visited:
            if current == goal_state:
                parent[current] = parent.get(current, None)
                move_record[current] = move_record.get(current, None)
                return reconstruct_path(parent, move_record, current), len(visited), processed, depth
            visited.add(current)
            processed += 1
            actual_max_depth = max(actual_max_depth, depth)

            for move in reversed(moves):
                dx, dy = moves[move]
                new_state = make_move(current, dx, dy)  # nowy stan na podstawie ruchu
                if new_state is not None:  # jeżeli nowy stan istnieje
                    if new_state not in visited:  # jeżeli nowy stan nie był już odwiedzony
                        parent[new_state] = current  # zapisujemy poprzedni stan jako rodzica
                        move_record[new_state] = move  # zapisujemy ruch, jaki nas doprowadził do nowego stanu
                        open_list.append((new_state, depth + 1)) #dodajemy do listy stanów otwartych z większą głębokością niż rodzic

    return None, len(visited), processed, actual_max_depth



# def dfs(graph, root):
#     open_list = []
#     visited = set()
#     open_list.append(root)
#     while len(open_list) != 0:
#         current = open_list.pop()
#         if current not in visited:
#             if success == current:
#                 return True #success
#             visited.add(current)
#             for neighbor in reversed(graph[current]):
#                 open_list.append(neighbor)
#     return False #failure