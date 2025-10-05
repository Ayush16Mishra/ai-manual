from collections import deque
import graphviz

# ---------- Missionaries and Cannibals ----------
def is_valid_state(m_left, c_left, m_right, c_right):
    if (m_left < 0 or c_left < 0 or m_right < 0 or c_right < 0):
        return False
    if (m_left > 0 and m_left < c_left):
        return False
    if (m_right > 0 and m_right < c_right):
        return False
    return True

def get_successors(state):
    m_left, c_left, boat = state
    m_right = 3 - m_left
    c_right = 3 - c_left
    successors = []
    moves = [(1,0),(2,0),(0,1),(0,2),(1,1)]
    if boat == 0:
        for m,c in moves:
            new_state = (m_left - m, c_left - c, 1)
            if is_valid_state(new_state[0], new_state[1], 3-new_state[0], 3-new_state[1]):
                successors.append(new_state)
    else:
        for m,c in moves:
            new_state = (m_left + m, c_left + c, 0)
            if is_valid_state(new_state[0], new_state[1], 3-new_state[0], 3-new_state[1]):
                successors.append(new_state)
    return successors

def bfs_missionaries():
    start = (3,3,0)
    goal = (0,0,1)
    queue = deque([(start, [start])])
    visited = set([start])
    edges = []

    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path, edges
        for succ in get_successors(state):
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, path+[succ]))
                edges.append((state, succ))
    return None, edges

def dfs_missionaries():
    start = (3,3,0)
    goal = (0,0,1)
    stack = [(start,[start])]
    visited = set()
    edges = []

    while stack:
        state, path = stack.pop()
        if state == goal:
            return path, edges
        if state not in visited:
            visited.add(state)
            for succ in get_successors(state):
                stack.append((succ, path+[succ]))
                edges.append((state, succ))
    return None, edges

def print_path(path, label):
    print(f"\n{label} (steps = {len(path)-1}):")
    for step in path:
        print(step)

def draw_graph(edges, filename="graph"):
    dot = graphviz.Digraph()
    for parent, child in edges:
        dot.edge(str(parent), str(child))
    dot.render(filename, format='png', cleanup=True)
    print(f"Graph saved as {filename}.png")

# ---------- Rabbit Leap ----------
def get_successors_rabbit(state):
    state = list(state)
    successors = []
    for i in range(len(state)):
        if state[i] == 'E':
            if i+1 < len(state) and state[i+1] == '_':
                new_state = state.copy()
                new_state[i], new_state[i+1] = '_','E'
                successors.append(tuple(new_state))
            if i+2 < len(state) and state[i+2]=='_':
                new_state = state.copy()
                new_state[i], new_state[i+2] = '_','E'
                successors.append(tuple(new_state))
        elif state[i] == 'W':
            if i-1 >=0 and state[i-1]=='_':
                new_state = state.copy()
                new_state[i],new_state[i-1] = '_','W'
                successors.append(tuple(new_state))
            if i-2 >=0 and state[i-2]=='_':
                new_state = state.copy()
                new_state[i],new_state[i-2] = '_','W'
                successors.append(tuple(new_state))
    return successors

def bfs_rabbit():
    start = ('E','E','E','_','W','W','W')
    goal = ('W','W','W','_','E','E','E')
    queue = deque([(start,[start])])
    visited=set([start])
    edges = []

    while queue:
        state,path=queue.popleft()
        if state==goal:
            return path, edges
        for succ in get_successors_rabbit(state):
            if succ not in visited:
                visited.add(succ)
                queue.append((succ,path+[succ]))
                edges.append((state, succ))
    return None, edges

def dfs_rabbit():
    start = ('E','E','E','_','W','W','W')
    goal = ('W','W','W','_','E','E','E')
    stack=[(start,[start])]
    visited=set()
    edges = []

    while stack:
        state,path=stack.pop()
        if state==goal:
            return path, edges
        if state not in visited:
            visited.add(state)
            for succ in get_successors_rabbit(state):
                stack.append((succ,path+[succ]))
                edges.append((state, succ))
    return None, edges

def print_path_rabbit(path, label):
    print(f"\n{label} (steps = {len(path)-1}):")
    for step in path:
        print("".join(step))

def draw_linear_path(path, filename="linear_path"):
    dot = graphviz.Digraph()
    for i in range(len(path)-1):
        dot.edge("".join(path[i]), "".join(path[i+1]))
    dot.render(filename, format='png', cleanup=True)
    print(f"Linear path saved as {filename}.png")

# ---------- Main ----------
if __name__=="__main__":
    # Missionaries & Cannibals BFS
    bfs_path_mc, bfs_edges_mc = bfs_missionaries()
    print_path(bfs_path_mc, "Missionaries & Cannibals BFS")
    draw_graph(bfs_edges_mc, "missionaries_bfs")

    # Missionaries & Cannibals DFS
    dfs_path_mc, dfs_edges_mc = dfs_missionaries()
    print_path(dfs_path_mc, "Missionaries & Cannibals DFS")
    draw_graph(dfs_edges_mc, "missionaries_dfs")

    # Rabbit Leap BFS
    bfs_path_rb, bfs_edges_rb = bfs_rabbit()
    print_path_rabbit(bfs_path_rb, "Rabbit Leap BFS")
    draw_linear_path(bfs_path_rb, "rabbit_bfs")

    # Rabbit Leap DFS
    dfs_path_rb, dfs_edges_rb = dfs_rabbit()
    print_path_rabbit(dfs_path_rb, "Rabbit Leap DFS")
    draw_linear_path(dfs_path_rb, "rabbit_dfs")
