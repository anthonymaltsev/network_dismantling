import networkx as nx

### FUNCTION LIST
# draw_graph_with_nodes(G, nodes,node_size=100)
# draw_graph_ccs(G,node_size=100)

### CODE

def draw_graph_with_nodes(G, nodes,node_size=100) :
  for node in G.nodes:
    if node in nodes:
      G.nodes[node]['color'] = 'r'
    else :
      G.nodes[node]['color'] = 'b'
  nx.draw(G, node_color=[G.nodes[v]['color'] for v in G.nodes],node_size=node_size)

def draw_graph_ccs(G,node_size=100) :
  ccs = nx.connected_components(G)
  poss_colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
  pc_ind = 0
  for cc in ccs :
    rand_color = poss_colors[pc_ind]
    for node in cc :
      G.nodes[node]['color'] = rand_color
    pc_ind += 1
    pc_ind %= len(poss_colors)
  nx.draw(G, node_color=[G.nodes[v]['color'] for v in G.nodes],node_size=node_size)