from collections import deque

success=((1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,0))

MOVES = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

# def bfs(graph, root):
#     if root==success:
#         return True
#     open_queue = deque([root])
#     closed_set = set()
#     while open_queue:
#         state= open_queue.popleft()
#         closed_set.add(state)
#         for neighbor in graph[state]:
#             if neighbor not in closed_set and neighbor not in open_queue:
#                 if graph[state]==success:
#                     return True #success
#                 open_queue.append(neighbor)
#     return False #failure
def bfs(root):
    if root==success:
        return True
    open_queue = deque([root])
    closed_set = set()
    parent={}
    move_record = {}

    open_queue.append((root, 0))  # (state, depth)
    closed_set.add(root)
    parent[root]=None
    move_record[root] = None

    max_depth = 50  #ZMNIEJSZYC
    processed = 0

    while open_queue:
        # state,depth= open_queue.popleft()
        state = open_queue.popleft()
        processed+=1
        # max_depth = max(max_depth, depth)

        if state == success:
            return True

        #closed_set.add(state)
        # for neighbor in graph[state]:
        #     if neighbor not in closed_set and neighbor not in open_queue:
        #         if graph[state]==success:
        #             return True #success
        #         open_queue.append(neighbor)
        for move, (dx, dy) in MOVES.items():
            new_state = make_move(state, dx, dy)
            if new_state and new_state not in closed_set:
                closed_set.add(new_state)
                parent[new_state] = state
                move_record[new_state] = move
                open_queue.append((new_state, processed + 1))
    return False #failure

def make_move(state, dx, dy):
    #state_list = [list(row) for row in state]

    state_list = list(state)

    x=0
    y=0
    # Znajdź pozycję 0
    for i in range(dx):
        for j in range(dy):
            if state_list[i][j] == 0:
                x, y = i, j
                break

    new_x, new_y = x + dx, y + dy

    # Sprawdź, czy nowa pozycja jest w granicach
    if 0 <= new_x < dx and 0 <= new_y < dy:
        # Zamień 0 z elementem docelowym
        state_list[x][y], state_list[new_x][new_y] = state_list[new_x][new_y], state_list[x][y]
        return tuple(tuple(row) for row in state_list)

    return None
