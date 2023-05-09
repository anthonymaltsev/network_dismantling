import numpy as np
import networkx as nx

# GND algorithm
# This file implements the GND algorithm developed by Ren et al in their paper found here:
#   https://www.pnas.org/doi/10.1073/pnas.1806108116

def get_second_eigenvector_approx(L_tilde, iters) :
  # draw a v unif random from sphere by drawing each point
  #   as a gaussian and rescaling (as suggested by GND paper)
  v = np.random.normal(0,1, size=L_tilde.shape[0])
  v = v / np.linalg.norm(v, ord=2) # normalize to unit sphere
  v_1 = np.ones(L_tilde.shape[0])
  v = v - (v @ v_1)/(v_1 @ v_1) * v_1 # make v orthogonal to v_1
  # now do iterative approximation for v_2 :
  for _ in range(iters) :
    L_v = L_tilde @ v
    v = L_v / np.linalg.norm(L_v)
  return v

def get_max_degree(G):
  # sorts by degree in descending order, returns first element
  return sorted(G.degree, key=lambda x: x[1], reverse=True)[0]

def nx_wvc_source(G, weight=None) :
  cost = dict(G.nodes(data=weight, default=1))
  # While there are uncovered edges, choose an uncovered and update
  # the cost of the remaining edges.
  cover = set()
  for u, v in G.edges():
    if u in cover or v in cover:
      continue
    if cost[u] <= cost[v]:
      cover.add(u)
      cost[v] -= cost[u]
    else:
      cover.add(v)
      cost[u] -= cost[v]
  return cover

def gnd(G, c, verbose=False):
  G = G.copy()
  removed_nodes = set()
  d_max = get_max_degree(G)[1]
  ccs = nx.connected_components(G)  
  gcc_node_set = list(max(ccs, key=len))
  gcc_size = len(gcc_node_set)
  gcc = G.subgraph(gcc_node_set).copy()

  if verbose :
    iter_tracker = 1

  while gcc_size > c:
    if verbose: 
      print(f"--- Starting iter {iter_tracker}")

    v_2_fiedler = nx.fiedler_vector(gcc, weight='edge weight')
    v_2_rounded = np.sign(v_2_fiedler)
    nodes_in_M = [gcc_node_set[i] for i in range(len(v_2_rounded)) if v_2_rounded[i] >= 0]

    edges_across_cut = []
    for u, v in gcc.edges:
      if (u in nodes_in_M) != (v in nodes_in_M):
        edges_across_cut.append((u, v))

    # compute optimal cut
    G_star = gcc.edge_subgraph(edges_across_cut).copy()
    S = nx_wvc_source(G_star, weight='node weight')
    removed_nodes.update(S)

    # Updates
    G.remove_nodes_from(list(S))
    ccs = nx.connected_components(G) 
    gcc_node_set = list(max(ccs, key=len))
    gcc_size = len(gcc_node_set)
    gcc = G.subgraph(gcc_node_set).copy()
    
    if verbose:
      print(f"--- Ending iter {iter_tracker}")
      print(edges_across_cut)
      print(len(S))
      iter_tracker += 1



  return removed_nodes