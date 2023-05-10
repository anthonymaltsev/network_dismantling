import networkx as nx
import numpy as np
import time
import random

### FUNCTION LIST
# greedy_algorithm(heuristic, _G, c=0.2)
# draw_greedy_algorithm(heuristic, G, c=0.2)
# remove_random_node(gcc, G)
# lightest_node(gcc, G)
# weighted_degree_centrality(gcc, G)
# betweenness_centrality_cd(gcc, G)
# betweenness_centrality_endpoint_weights(gcc, G)
# betweenness_centrality_wsp(gcc, G)
# approximate_betweenness_centrality_cd(gcc, G)
# approximate_betweenness_centrality_wsp(gcc, G)
# eigenvector_centrality_cd(gcc, G)
# power_iteration(G, nodelist, max_iter=100)
# eigenvector_centrality_weighted_adj(gcc, G)
# eigenvector_centrality_combined(gcc, G)
# simplify_and_invert_laplacian(L)
# approx_current_flow_betweenness(gcc, G, using_cd=False, using_weighted_edges=False, using_weighted_supply=False)
# cfbc_cd = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_cd=True)
# cfbc_wedge = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_weighted_edges=True)
# cfbc_wsup = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_weighted_supply=True)
# cfbc_combined = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_cd=True, using_weighted_edges=True, using_weighted_supply=True)
# laplacian_centrality(gcc, G, CD=False)
# laplacian_centrality_cd(gcc, G)
# second_eval_estimate(full_G)
# modified_laplacian_centrality(gcc, full_G, CD=False)
# modified_laplacian_centrality_cd(gcc, full_G)
# load_centrality(gcc, G)
# communicability_centrality(gcc, G)
# information_centrality(gcc, G)

### CODE BELOW

def greedy_algorithm(heuristic, _G, c=0.2):
    start = time.time()

    G = _G.copy()
    N = G.number_of_nodes()
    S = []

    # The connected components are ordered by decreasing size.
    gcc = max(nx.connected_components(G), key=len)
    size_gcc = len(list(gcc))

    while size_gcc > c * N:
        node_to_remove = heuristic(gcc, G)
        G.remove_node(node_to_remove)
        S.append(node_to_remove)

        components = nx.connected_components(G)
        gcc = max(nx.connected_components(G), key=len)
        size_gcc = len(list(gcc))

    print(f"Removed {len(S)} nodes from G.")
    cost = sum(_G.nodes[i]['node weight'] for i in S)
    print(f"Cost: {cost}")
    
    # final_ccs = list(nx.connected_components(G))
    # average_cc_size = sum(len(G) for G in final_ccs) / len(final_ccs)
    # print(f"gcc size: {size_gcc}")
    # print(f"Average cc size: {average_cc_size}")

    end = time.time()
    print(f"Time elapsed: {round(end - start, 3)}s")
    return S, G

# Runs the above algorithm. Colors nodes in S red.
def draw_greedy_algorithm(heuristic, G, c=0.2):
    colors = ["blue" for _ in range(G.number_of_nodes())]
    S, _ = greedy_algorithm(heuristic, G, c)
    for node in S:
        colors[node] = "red"
    nx.draw(G, node_color=colors)

def remove_random_node(gcc, G):
    return random.choice(G.nodes())

def lightest_node(gcc, G):
    return min(gcc, key=lambda v: G.nodes[v]['node weight'])

def weighted_degree_centrality(gcc, G):
    # Page 7, https://computationalsocialnetworks.springeropen.com/articles/10.1186/s40649-020-00081-w
    degrees = nx.degree_centrality(G.subgraph(gcc))
    weighted_degrees = {}
    for u in gcc:
        weighted_degree = 0
        for v in G.neighbors(u):
            weighted_degree += G.nodes[v]['node weight']
        weighted_degrees[u] = weighted_degree
    return max(gcc, key=lambda v: weighted_degrees[v])

def betweenness_centrality_cd(gcc, G):
    centrality_dict = nx.betweenness_centrality(G.subgraph(gcc), normalized=True)
    return max(gcc, key=lambda v: centrality_dict[v] / G.nodes[v]['node weight'])

def betweenness_centrality_endpoint_weights(gcc, G):
    shortest_paths = nx.shortest_path(G.subgraph(gcc))
    best_vertex, max_centrality = -1, -1
    for v in gcc:
        centrality = 1
        for s in gcc:
            for t in gcc:
                if v in shortest_paths[s][t]:
                    centrality += G.nodes[s]['node weight'] 
                    centrality += G.nodes[t]['node weight']
        if centrality > max_centrality:
            best_vertex, max_centrality = v, centrality
    return best_vertex

def approximate_betweenness_centrality_cd(gcc, G):
    centrality_dict = nx.betweenness_centrality(G.subgraph(gcc), k=max(1, int(np.log(len(gcc)))))
    return max(gcc, key=lambda v: centrality_dict[v] / G.nodes[v]['node weight'])


def betweenness_centrality_wsp(gcc, G):
    # https://www.hindawi.com/journals/complexity/2021/1677445/
    # We find betweenness centrality using weighted shortest paths.
    # The higher the weight of a shortest path, the heavier the nodes are on that path. Thus, disconnecting that path is more better.
    # (Is this true? TODO: Figure out what networkx is doing here: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html)
    centrality_dict = nx.betweenness_centrality(G.subgraph(gcc), weight ='edge weight')
    return max(gcc, key=lambda v: centrality_dict[v])


def approximate_betweenness_centrality_wsp(gcc, G):
    centrality_dict = nx.betweenness_centrality(G.subgraph(gcc), weight='edge weight', k=max(1, int(np.log(len(gcc)))))
    return max(gcc, key=lambda v: centrality_dict[v])

def eigenvector_centrality_cd(gcc, G):
    # Page 7, https://computationalsocialnetworks.springeropen.com/articles/10.1186/s40649-020-00081-w
    # "Eigenvector centrality is a measure where it is still open how to include the efect of node weights."
    try:
        centrality_dict = nx.eigenvector_centrality_numpy(G.subgraph(gcc))
        return max(gcc, key=lambda v: centrality_dict[v] / G.nodes[v]['node weight'])
    except:
        return lightest_node(gcc, G)

def power_iteration(G, nodelist, max_iter=100):
        A = nx.to_numpy_array(G, nodelist=nodelist, weight='node weight')
        v = np.random.rand(A.shape[1])
        for _ in range(max_iter):
                v = np.dot(A, v)
                try:
                    v = v / np.linalg.norm(v)
                except:
                    print(_, v)
                    x = 5 / 0 
        return v

def eigenvector_centrality_weighted_adj(gcc, G):
    gcc = list(gcc)
    v = power_iteration(G, gcc)
    return gcc[np.argmax(v)]


def eigenvector_centrality_combined(gcc, G):
    gcc = list(gcc)
    v = power_iteration(G, gcc)
    cd_v = [u / G.nodes[gcc[i]]['node weight'] for (i, u) in enumerate(v)]
    return max(gcc, key=lambda v: cd_v[gcc.index(v)])


def simplify_and_invert_laplacian(L):
    L = np.delete(L, 0, axis=0)
    L_tilde = np.delete(L, 0, axis=1)
    L_inv = np.linalg.inv(L_tilde)
    L_with_zeroes_on_top = np.vstack((np.zeros(L_inv.shape[1]), L_inv))
    zeroes_column = np.zeros((L_with_zeroes_on_top.shape[0],1))
    C = np.hstack((zeroes_column, L_with_zeroes_on_top))
    return C


def approx_current_flow_betweenness(gcc, G, using_cd=False, using_weighted_edges=False, using_weighted_supply=False):
    G = G.subgraph(gcc)
    gcc = sorted(list(gcc))
    n = G.number_of_nodes()

    vertices_ordered = {v: i for i, v in enumerate(G.nodes)}
    centrality = {v: 0 for v in G.nodes}

    C = simplify_and_invert_laplacian(nx.laplacian_matrix(G).toarray())

    # k=logn pivots are used for approximation
    for _ in range(int(np.log(n))):
        # Source and sink, chosen uniformly at random
        s, t = random.sample(gcc, k=2)

        # Supply vector b
        b = np.zeros(n)
        b[vertices_ordered[s]] = 1
        b[vertices_ordered[t]] = -1
        if using_weighted_supply: 
            b = b * (G.nodes[s]['node weight'] + G.nodes[t]['node weight'])

        # Potential difference vector p
        p = C @ b

        for v in G.nodes:
            for e in G.edges(v):
                u = e[0] if e[0] != v else e[1]
                direction_of_current = 1 if vertices_ordered[v] < vertices_ordered[u] else -1
                potential_difference = abs(p[vertices_ordered[v]] - p[vertices_ordered[u]])
                centrality[v] += direction_of_current * (1 / G.edges[e]['edge weight'] if using_weighted_edges else 1)

    best_v = max(gcc, key=lambda v: centrality[v] / (G.nodes[v]['node weight'] if using_cd else 1))
    return best_v 

cfbc_cd = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_cd=True)
cfbc_wedge = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_weighted_edges=True)
cfbc_wsup = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_weighted_supply=True)
cfbc_combined = lambda gcc, G: approx_current_flow_betweenness(gcc, G, using_cd=True, using_weighted_edges=True, using_weighted_supply=True)

# https://math.wvu.edu/~cqzhang/Publication-files/my-paper/INS-2012-Laplacian-W.pdf
def laplacian_centrality(gcc, G, CD=False):
        # Related to the drop in laplacian energy (the sum of the squared eigenvalues of the Laplacian) after removing a vertex.
        # TODO: Figure out how networkx uses weights.
        centrality_dict = nx.laplacian_centrality(G.subgraph(gcc), weight='edge weight')
        if not CD:
            return max(gcc, key=lambda v: centrality_dict[v])
        elif CD :
            return max(gcc, key=lambda v: centrality_dict[v] / G.nodes[v]['node weight'])

def laplacian_centrality_cd(gcc, G) :
    return laplacian_centrality(gcc, G, CD=True)

# this is a modified version of laplacian centrality where only the effect on the 
# second eigenvalue of removing a node is considered. ie laplacian energy is 
# redefined as (lambda_2)^2.
def second_eval_estimate(full_G) :

    L = nx.laplacian_matrix(G)
    try :
        v_2 = nx.fiedler_vector(G)
    except :
        return 0
    v_2 = v_2 / np.linalg.norm(v_2)
    return ((L@v_2) @ v_2)**2

def modified_laplacian_centrality(gcc, full_G, CD=False) :
    G = full_G.subgraph(gcc)
    curr_MLE = second_eval_estimate(G) # current modified laplacian energy
    if CD :
        MLCs = [(node, (curr_MLE - second_eval_estimate(G.subgraph(set(G.nodes) - {node})))/G.nodes[node]['node weight']) for node in G.nodes]
    else :
        MLCs = [(node, curr_MLE - second_eval_estimate(G.subgraph(set(G.nodes) - {node}))) for node in G.nodes]
    return max(MLCs, key=lambda x: x[1])[0]

def modified_laplacian_centrality_cd(gcc, full_G) :
    return modified_laplacian_centrality(gcc, full_G, CD=True)

# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.87.278701
def load_centrality(gcc, G):
        # Related to the fraction of all shortest paths that pass through a node.
        # TODO: Figure out how this is different from betweenness centrality
        centrality_dict = nx.load_centrality(G.subgraph(gcc), weight='edge weight')
        return max(gcc, key=lambda v: centrality_dict[v])

# https://arxiv.org/abs/0905.4102
def communicability_centrality(gcc, G):
        # Related to the number of walks connecting every pair of nodes, as opposed to the number of shortest paths.
        # TODO: Work with node weights
        centrality_dict = nx.communicability_betweenness_centrality(G.subgraph(gcc))
        return max(gcc, key=lambda v: centrality_dict[v])

# https://link.springer.com/chapter/10.1007/978-3-540-31856-9_44
# Also called 'current flow centrality'.
def information_centrality(gcc, G):
    # TODO: Look into this
    centrality_dict = nx.information_centrality(G.subgraph(gcc), weight='edge weight')
    return max(gcc, key=lambda v: centrality_dict[v])

