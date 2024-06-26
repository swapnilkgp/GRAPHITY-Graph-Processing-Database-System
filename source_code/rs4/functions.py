import os
import struct
from collections import deque
from collections import defaultdict
import time
import heapq
#-----------------------------------------------------------------------------------------------
KB = 128
ATTR_SIZE=16
ADJACENCY_LIST_FOLDER = "Adjacency_list"
NODE_INFO_FOLDER = "Node_info"
File_Index_FOLDER = "File_Index"
#-----------------------------------------------------------------------------------------------

num_blocks_accessed = 0
prev_file_path = ''
class Node:
    def __init__(self, node_id, adj_start_idx, in_deg, out_deg, scc_id, pg_rank, wcc_id, rank, adj_start_idx_in):
        self.node_id = node_id
        self.adj_start_idx = adj_start_idx
        self.in_deg = in_deg
        self.out_deg = out_deg
        self.scc_id = scc_id
        self.pg_rank = pg_rank
        self.wcc_id = wcc_id
        self.rank = rank
        self.adj_start_idx_in = adj_start_idx_in

def get_block():
    global num_blocks_accessed
    temp=num_blocks_accessed
    num_blocks_accessed=0
    return temp

def create_directories(base_dir_name,sub_dir_name, graph_type):
    base_dir_path = os.path.join(os.getcwd(), base_dir_name)
    os.makedirs(base_dir_path, exist_ok=True)
    subdirectories = ["Adjacency_list", "Node_info", "File_Index", "Adjacency_list_in","Rank"]
    for subdir_name in subdirectories:
        subdir_path = os.path.join(base_dir_path, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
    # Read the sample.txt file
    file_name = f"{base_dir_name}/{sub_dir_name}.txt"
    with open(file_name, 'r') as file:
        # Read the first line and split it into two parts based on space
        first_line = file.readline().strip().split()
        
        # Extract the number of nodes (n) and number of edges (m)
        n = int(first_line[0])
        m = int(first_line[1])

    # Write the metadata to Meta_data.txt file
    with open(f"{base_dir_name}/Meta_data.txt", 'w') as meta_file:
        meta_file.write(f"Nodes : {n}\nEdges : {m}\nType : {graph_type}\n")

def get_Meta_data(directory_name):
    file_path = f"{directory_name}/Meta_data.txt"
    metadata = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(' : ')
                metadata[key.strip()] = int(value.strip())
    except FileNotFoundError:
        print("Error: File not found.")
    
    return metadata

def read_nth_integer_from_binary_file(file_path, n):
    global num_blocks_accessed
    global prev_file_path
    try:
        with open(file_path, "rb") as file:
            if file_path != prev_file_path:
                prev_file_path = file_path
                num_blocks_accessed+=1
            position = (n) * 4
            file.seek(position)
            integer_bytes = file.read(4)
            if len(integer_bytes) == 4:
                integer_value = int.from_bytes(integer_bytes, byteorder='little')
                return integer_value
            else:
                return -1
    except:
        return -1

def set_nth_integer_in_binary_file(file_path, n, x):
    global num_blocks_accessed
    global prev_file_path
    try:
        with open(file_path, "r+b") as file:
            if file_path != prev_file_path:
                prev_file_path = file_path
                num_blocks_accessed+=1
            position = (n) * 4
            file.seek(position)
            x_bytes = x.to_bytes(4, byteorder='little')
            file.write(x_bytes)
    except:
        pass

def write_array_to_binary_file(file_path, integer_array):
    with open(file_path, "wb") as file:
        for integer in integer_array:
            integer_bytes = integer.to_bytes(4, byteorder='little')
            file.write(integer_bytes)

def compute_degrees_from_edge_list(directory_name):
    filename = directory_name.split('/')[-1]
    file_path = f"{directory_name}/{filename}.txt"
    # Initialize dictionaries to store indegree and outdegree of each node
    indegree = {}
    outdegree = {}
    n = -1
    m = -1
    Type = get_Meta_data(directory_name)["Type"]
    
    # Read the edge list from the file
    with open(file_path, "r") as file:
        lines = file.readlines()
    check = False
    # Process each edge and compute degrees
    for line in lines:
        # Parse the edge

        if (not check):
            check = True
            u, v = map(int, line.strip().split())
            n = u
            m = v
            continue
        if Type == 1:
            u, v = map(int, line.strip().split())
        else:
            u,v,w = map(int, line.strip().split())
        # Update outdegree of node u
        if u in outdegree:
            outdegree[u] += 1
        else:
            outdegree[u] = 1
        
        # Update indegree of node v
        if v in indegree:
            indegree[v] += 1
        else:
            indegree[v] = 1

        # Initialize outdegree and indegree for nodes not present in the edge list
        if v not in outdegree:
            outdegree[v] = 0
        if u not in indegree:
            indegree[u] = 0

    return n,m,indegree, outdegree

def get_node_info_filename(x, folder_name, directory_name):
    file_path = f"{directory_name}/{folder_name}/{x}.bin"
    try:
        with open(file_path, 'x') as file:
            pass
    except FileExistsError:
        pass
    return file_path

def make_node_info(directory_name):
    n,m,indegree, outdegree = compute_degrees_from_edge_list(directory_name)
    Type = get_Meta_data(directory_name)["Type"]
    sum=0
    sum2 = 0
    for i in range(n):
        file_no = (i*ATTR_SIZE)//KB
        if i not in indegree:
            indegree[i] = 0
        if i not in outdegree:
            outdegree[i] = 0
        file_name = get_node_info_filename(file_no, "Node_info", directory_name)
        for j in range(ATTR_SIZE):
            pos = (i*ATTR_SIZE + j) % KB
            if j == 0:
                # Node-idx
                set_nth_integer_in_binary_file(file_name,pos,i)
            elif j == 1:
                # start-idx
                set_nth_integer_in_binary_file(file_name,pos,sum)
            elif j == 2:
                # In_degree 
                set_nth_integer_in_binary_file(file_name,pos,indegree[i])
            elif j == 3:
                # Out-degree
                set_nth_integer_in_binary_file(file_name,pos,outdegree[i])
                sum += outdegree[i]*Type
            elif j == 8:
                set_nth_integer_in_binary_file(file_name,pos,sum2)
                sum2 += indegree[i]*Type
            else:
                set_nth_integer_in_binary_file(file_name,pos,0)

def make_adj_list(directory_name):
    Type = get_Meta_data(directory_name)["Type"]
    # Initialize dictionaries to store indegree and outdegree of each node
    filename = directory_name.split('/')[-1]
    file_path=f"{directory_name}/{filename}.txt"
    outdegree = {}
    indegree = {}
    # Read the edge list from the file
    with open(file_path, "r") as file:
        lines = file.readlines()
    check = False
    # Process each edge and compute degrees
    for line in lines:
        # Parse the edge
        if (not check):
            check = True
            u, v = map(int, line.strip().split())
            continue
        if Type == 1:
            u, v = map(int, line.strip().split())
        else:
            u, v, w = map(int, line.strip().split())
        x = get_start_idx(directory_name,u)
        x2 = get_start_idx2(directory_name,v)
        if u not in outdegree:
            outdegree[u] = 0
        if v not in indegree:
            indegree[v] = 0
        adj_file_path =get_node_info_filename((x+outdegree[u])//KB, "Adjacency_list", directory_name)
        adj_file_path2 =get_node_info_filename((x2+indegree[v])//KB, "Adjacency_list_in", directory_name)
        set_nth_integer_in_binary_file(adj_file_path,(x+outdegree[u])%KB,v)
        set_nth_integer_in_binary_file(adj_file_path2,(x2+indegree[v])%KB,u)
        if Type == 2:
            adj_file_path =get_node_info_filename((x+outdegree[u]+1)//KB, "Adjacency_list", directory_name)
            adj_file_path2 =get_node_info_filename((x2+indegree[v]+1)//KB, "Adjacency_list_in", directory_name)
            set_nth_integer_in_binary_file(adj_file_path,(x+outdegree[u]+1)%KB,w)
            set_nth_integer_in_binary_file(adj_file_path2,(x2+indegree[v]+1)%KB,w)
        outdegree[u] += Type
        indegree[v] += Type
    return 
    
def make_File_Index(directory_name):
    filename = directory_name.split('/')[-1]
    file_path=f"{directory_name}/{filename}.txt"
    outdegree = {}
    meta_data = get_Meta_data(directory_name)
    n = meta_data["Nodes"]
    for i in range(n):
        file_path = get_node_info_filename(i//KB,"File_Index",directory_name)
        set_nth_integer_in_binary_file(file_path,i%KB,ATTR_SIZE*i)

def get_out_degree(directory_name, node):
    file_path = get_node_info_filename((ATTR_SIZE*node)//KB,"Node_info",directory_name)
    od = read_nth_integer_from_binary_file(file_path,(ATTR_SIZE*node+3)%KB)
    return od

def get_in_degree(directory_name, node):
    file_path = get_node_info_filename((ATTR_SIZE*node)//KB,"Node_info",directory_name)
    od = read_nth_integer_from_binary_file(file_path,(ATTR_SIZE*node+2)%KB)
    return od

def get_start_idx(directory_name, node):
    file_path = get_node_info_filename((ATTR_SIZE*node)//KB,"Node_info",directory_name)
    od = read_nth_integer_from_binary_file(file_path,(ATTR_SIZE*node+1)%KB)
    return od

def get_start_idx2(directory_name, node):
    file_path = get_node_info_filename((ATTR_SIZE*node)//KB,"Node_info",directory_name)
    od = read_nth_integer_from_binary_file(file_path,(ATTR_SIZE*node+8)%KB)
    return od

cnt = 0
largest_scc = 0
def tarjan_scc(directory_name):
    global cnt
    global largest_scc
    cnt = 0
    largest_scc = 0
    def dfs(v):
        nonlocal index, stack, low, ids, on_stack
        global largest_scc
        global cnt
        low[v] = ids[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True
        out_degree=get_out_degree(directory_name,v)
        start_idx=get_start_idx(directory_name,v)
        Type = get_Meta_data(directory_name)["Type"]
        for i in range (start_idx,start_idx+Type*out_degree,Type):
            file_path = get_node_info_filename(i//KB,"Adjacency_list",directory_name)
            u = read_nth_integer_from_binary_file(file_path,i%KB)
            if ids[u] == -1:
                dfs(u)
                low[v] = min(low[v], low[u])
            elif on_stack[u]:
                low[v] = min(low[v], ids[u])
        if low[v] == ids[v]:
            cnt += 1
            scc_size = 0
            while True:
                u = stack.pop()
                on_stack[u] = False
                file_path = get_node_info_filename((ATTR_SIZE*u)//KB,"Node_info",directory_name)
                set_nth_integer_in_binary_file(file_path,(ATTR_SIZE*u+4)%KB,cnt)
                scc_size += 1
                if u == v:
                    break
            if scc_size > largest_scc:
                largest_scc = scc_size 
    meta_data = get_Meta_data(directory_name)
    n = meta_data["Nodes"]
    index = 0
    stack = []
    low = [-1] * n
    ids = [-1] * n
    on_stack = [False] * n
    for v in range(n):
        if ids[v] == -1:
            dfs(v)
    with open(f"{directory_name}/Meta_data.txt", 'a') as meta_file:
        meta_file.write(f"numSCCs : {cnt}\nlargest_SCC : {largest_scc}\n")

wcnt = 0
largest_wcc = 0
wcc_size = 0
def wcc(directory_name):
    global wcnt
    global largest_wcc
    global wcc_size
    wcnt = 0
    def dfs(directory_name, node, visited):
        global wcnt
        global largest_wcc
        global wcc_size
        visited.add(node)
        file_path = get_node_info_filename((ATTR_SIZE*node)//KB,"Node_info",directory_name)
        set_nth_integer_in_binary_file(file_path,(ATTR_SIZE*node+6)%KB,wcnt)
        wcc_size += 1
        out_degree=get_out_degree(directory_name,node)
        start_idx=get_start_idx(directory_name,node)
        Type = get_Meta_data(directory_name)["Type"]
        for i in range (start_idx,start_idx+Type*out_degree,Type):
            file_path = get_node_info_filename(i//KB,"Adjacency_list",directory_name)
            neighbor = read_nth_integer_from_binary_file(file_path,i%KB)
            if neighbor not in visited:
                dfs(directory_name, neighbor, visited)
        in_degree=get_in_degree(directory_name,node)
        start_idx2=get_start_idx2(directory_name,node)
        for i in range (start_idx2,start_idx2+in_degree*Type,Type):
            file_path = get_node_info_filename(i//KB,"Adjacency_list_in",directory_name)
            neighbor = read_nth_integer_from_binary_file(file_path,i%KB)
            if neighbor not in visited:
                dfs(directory_name, neighbor, visited)
    visited = set()
    meta_data = get_Meta_data(directory_name)
    n = meta_data["Nodes"]
    for node in range(n):
        if node not in visited:
            wcnt+=1
            wcc_size = 0
            dfs(directory_name, node, visited)
            if wcc_size > largest_wcc:
                largest_wcc = wcc_size
    with open(f"{directory_name}/Meta_data.txt", 'a') as meta_file:
        meta_file.write(f"numWCCs : {wcnt}\nlargest_WCC : {largest_wcc}\n")

def calculate_pagerank(directory_name, damping_factor=0.85, max_iterations=100, tolerance=1.0e-6):
    num_nodes=get_Meta_data(directory_name)["Nodes"]

    pagerank = {node: 1 / num_nodes for node in range(0, num_nodes)}

    # Iteratively calculate PageRank
    for _ in range(max_iterations):
        new_pagerank = {}
        for node in range(num_nodes):
            new_pagerank[node] = (1 - damping_factor) / num_nodes

        for v in range (0,num_nodes):
            out_degree=get_out_degree(directory_name,v)
            start_idx=get_start_idx(directory_name,v)
            Type = get_Meta_data(directory_name)["Type"]    
            for i in range (start_idx,start_idx+out_degree*Type,Type):
                file_path = get_node_info_filename(i//KB,"Adjacency_list",directory_name)
                u = read_nth_integer_from_binary_file(file_path,i%KB)
                num_out_links = out_degree
                new_pagerank[u] += damping_factor * pagerank[v] / num_out_links if num_out_links > 0 else 0

        # Check for convergence
        if sum(abs(new_pagerank[node] - pagerank[node]) for node in pagerank) < tolerance:
            break

        pagerank = new_pagerank

    for key,val in pagerank.items():
        val = int(val * 1e9)
        file_path = get_node_info_filename((ATTR_SIZE*key)//KB,"Node_info",directory_name)
        set_nth_integer_in_binary_file(file_path,(ATTR_SIZE*key+5)%KB,val)

def calculate_rank(directory_name):
    pairs = []
    for i in range(get_Meta_data(directory_name)["Nodes"]):
        rank = get_page_rank(directory_name,i)
        pairs.append((rank,i))

    sorted_pairs = sorted(pairs, key=lambda x: -x[0])

    cnt = 1
    for val,key in  sorted_pairs:
        file_path = get_node_info_filename((ATTR_SIZE*key)//KB,"Node_info",directory_name)
        set_nth_integer_in_binary_file(file_path,(ATTR_SIZE*key+7)%KB,cnt)
        file_path=get_node_info_filename((cnt-1)//KB,"Rank",directory_name)
        set_nth_integer_in_binary_file(file_path,(cnt-1)%KB,key)
        cnt += 1
        
def read_node_info(filename, ptr):
    global num_blocks_accessed

    # Open the binary file
    try:
        node_info = []
        with open(filename, "rb") as file:
            num_blocks_accessed += 1
            for i in range (ATTR_SIZE):
                position = (ptr + i) * 4
                file.seek(position)
                integer_bytes = file.read(4)
                if len(integer_bytes) == 4:
                    integer_value = int.from_bytes(integer_bytes, byteorder='little')
                    node_info.append(integer_value)

        node = Node(node_info[0],node_info[1],node_info[2],node_info[3],node_info[4],node_info[5],node_info[6],node_info[7],node_info[8])
    except:
        node = Node(-1,-1,-1,-1,-1,-1,-1,-1,-1)

    return node

def get_node_info(folder_name, node_idx):
    
    node_ptr = node_idx//KB
    node_btree_ptr = node_idx%KB
    filename = f"{folder_name}/{File_Index_FOLDER}/{node_ptr}.bin"

    val = read_nth_integer_from_binary_file(filename,node_btree_ptr)
    
    node_info_file = val//KB
    node_info_ptr = val%KB

    node_info_filename = f"{folder_name}/{NODE_INFO_FOLDER}/{node_info_file}.bin"

    return node_info_filename, node_info_ptr

def shortest_distance_bfs(folder_name, start_idx, end_idx):
    print("Hi baby")
    global num_blocks_accessed
    flag = get_Meta_data(folder_name)["Type"]

    if start_idx == end_idx:
        return 0

    visited = set()
    queue = deque([(start_idx, 0)])

    while queue:
        node, distance = queue.popleft()
        visited.add(node)
        
        node_file, ptr = get_node_info(folder_name, node)

        node_info = read_node_info(node_file, ptr)

        neighbor_left = node_info.out_deg
        adj_file_start = node_info.adj_start_idx%KB
        curr_file_idx = node_info.adj_start_idx//KB
        while neighbor_left>0:
            curr_file_name = f"{folder_name}/{ADJACENCY_LIST_FOLDER}/{curr_file_idx}.bin"
            adj_file_end = min(KB-1, adj_file_start + neighbor_left-1)
            with open(curr_file_name, 'rb') as file:
                num_blocks_accessed += 1
                for i in range(adj_file_start, adj_file_end+1):
                    position = (i) * 4
                    file.seek(position)
                    integer_bytes = file.read(4)
                    neighbor = -1
                    if len(integer_bytes) == 4:
                        neighbor = int.from_bytes(integer_bytes, byteorder='little')

                    if neighbor == end_idx:
                        print("bye baby")
                        return distance + 1

                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

            neighbor_left -= adj_file_end-adj_file_start+1
            curr_file_idx += 1
            adj_file_start = 0
    print("bye baby")
    return -1  # No path found

def get_scc_id(folder_name,node_idx):
    
    node_file, ptr = get_node_info(folder_name, node_idx)

    node_info = read_node_info(node_file, ptr)

    return node_info.scc_id
def get_wcc_id(folder_name,node_idx):
    
    node_file, ptr = get_node_info(folder_name, node_idx)

    node_info = read_node_info(node_file, ptr)

    return node_info.wcc_id

def checkInSameSCC(folder_name, start, end):
    return get_scc_id(folder_name,start) == get_scc_id(folder_name,end)
def checkInSameWCC(folder_name, start, end):
    return get_wcc_id(folder_name,start) == get_wcc_id(folder_name,end)

def get_page_rank(folder_name,node_idx):
    
    node_file, ptr = get_node_info(folder_name, node_idx)

    node_info = read_node_info(node_file, ptr)

    return node_info.pg_rank

def get_rank(folder_name,node_idx):
    
    node_file, ptr = get_node_info(folder_name, node_idx)

    node_info = read_node_info(node_file, ptr)

    return node_info.rank

def count_cycles(directory_name):
    cycles = []
    num_nodes = get_Meta_data(directory_name)["Nodes"]
    def dfs(node, visited, path):
        visited.add(node)
        path.append(node)
        out_degree=get_out_degree(directory_name,node)
        start_idx=get_start_idx(directory_name,node)
        Type = get_Meta_data(directory_name)["Type"]
        for i in range (start_idx,start_idx+Type*out_degree,Type):
            file_path = get_node_info_filename(i//KB,"Adjacency_list",directory_name)
            neighbor = read_nth_integer_from_binary_file(file_path,i%KB)
            if neighbor in path:
                # Cycle detected, print the cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                # print("Cycle:", cycle)
                cycles.append(cycle)
            if neighbor not in visited:
                dfs(neighbor, visited, path)
        path.pop()

    visited = set()
    for node in range(0,num_nodes):
        dfs(node, visited, [])
    with open(f"{directory_name}/Meta_data.txt", 'a') as meta_file:
        meta_file.write(f"numCycles : {len(cycles)}\n")
    return cycles

def shortest_distance_bfs(folder_name, start_idx, end_idx):
    global num_blocks_accessed
    if start_idx == end_idx:
        return 0

    visited = set()
    queue = deque([(start_idx, 0)])

    while queue:
        node, distance = queue.popleft()
        visited.add(node)
        
        node_file, ptr = get_node_info(folder_name, node)

        node_info = read_node_info(node_file, ptr)

        neighbor_left = node_info.out_deg
        adj_file_start = node_info.adj_start_idx%KB
        curr_file_idx = node_info.adj_start_idx//KB
        while neighbor_left>0:
            curr_file_name = f"{folder_name}/{ADJACENCY_LIST_FOLDER}/{curr_file_idx}.bin"
            adj_file_end = min(KB-1, adj_file_start + neighbor_left-1)
            with open(curr_file_name, 'rb') as file:
                num_blocks_accessed += 1
                for i in range(adj_file_start, adj_file_end+1):
                    position = (i) * 4
                    file.seek(position)
                    integer_bytes = file.read(4)
                    neighbor = -1
                    if len(integer_bytes) == 4:
                        neighbor = int.from_bytes(integer_bytes, byteorder='little')

                    if neighbor == end_idx:
                        return distance + 1

                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

            neighbor_left -= adj_file_end-adj_file_start+1
            curr_file_idx += 1
            adj_file_start = 0

    return -1  # No path found

def dijkstra(directory_name, start, end):
    Type = get_Meta_data(directory_name)["Type"]
    if Type == 1:
        return shortest_distance_bfs(directory_name, start, end)
    # Initialize distances dictionary to store shortest distances from start node
    distances = {node: float('inf') for node in range(get_Meta_data(directory_name)["Nodes"])}
    distances[start] = 0
    
    # Priority queue to store nodes with their current distances from start
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip processing if the current distance is greater than the previously calculated distance
        if current_distance > distances[current_node]:
            continue
        
        # Explore neighbors of the current node
        out_degree=get_out_degree(directory_name,current_node)
        start_idx=get_start_idx(directory_name,current_node)
        Type = get_Meta_data(directory_name)["Type"]
        for i in range (start_idx,start_idx+Type*out_degree,Type):
            file_path = get_node_info_filename(i//KB,"Adjacency_list",directory_name)
            neighbor = read_nth_integer_from_binary_file(file_path,i%KB)
            file_path = get_node_info_filename((i+1)//KB,"Adjacency_list",directory_name)
            if Type == 2:
                weight = read_nth_integer_from_binary_file(file_path,(i+1)%KB)
            else:
                weight = 1
            distance = current_distance + weight
            # Update distance if shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances[end]

def KNN(directory_name, start, k):
    # Initialize distances dictionary to store shortest distances from start node
    distances = {node: float('inf') for node in range(get_Meta_data(directory_name)["Nodes"])}
    distances[start] = 0
    
    # Priority queue to store nodes with their current distances from start
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip processing if the current distance is greater than the previously calculated distance
        if current_distance > distances[current_node]:
            continue
        
        # Explore neighbors of the current node
        out_degree=get_out_degree(directory_name,current_node)
        start_idx=get_start_idx(directory_name,current_node)
        Type = get_Meta_data(directory_name)["Type"]
        for i in range (start_idx,start_idx+Type*out_degree,Type):
            file_path = get_node_info_filename(i//KB,"Adjacency_list",directory_name)
            neighbor = read_nth_integer_from_binary_file(file_path,i%KB)
            file_path = get_node_info_filename((i+1)//KB,"Adjacency_list",directory_name)
            if Type == 2:
                weight = read_nth_integer_from_binary_file(file_path,(i+1)%KB)
            else:
                weight = 1
            distance = current_distance + weight
            # Update distance if shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                
    smallest_values = heapq.nsmallest(k+1, distances.items(), key=lambda x: x[1])
    d = dict(smallest_values)
    d.pop(start)
    for key,value in d.items():
        if(value==float('inf')):
            d[key]=-1
    return d

def get_ranklist(directory_name,l,r):
    ranklist = []
    for i in range(l-1, r):
        file_path = get_node_info_filename((i)//KB,"Rank",directory_name)
        od = read_nth_integer_from_binary_file(file_path,(i)%KB)
        ranklist.append(od)
    return ranklist