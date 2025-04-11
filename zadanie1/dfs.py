#wersja iteracyjna

success=([1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0])

#głębokość rekursji!!!

def dfs(graph, root):
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
    return False #failure

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