import networkx as nx
import numpy as np
import random

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

