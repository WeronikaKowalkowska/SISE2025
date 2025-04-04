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
