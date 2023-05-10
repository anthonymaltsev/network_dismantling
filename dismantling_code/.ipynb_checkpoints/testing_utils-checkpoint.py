import networkx as nx
import random
from dismantling_code.centrality_measures import greedy_algorithm
from dismantling_code.gnd import gnd

### FUNCTION LIST
# susc_generator(num_spikes, path_length, p=0)
# assign_weights_randomly(G)
# assign_node_weights_randomly_edge_weights_sum(G)
# assign_node_weights_randomly(G)
# test_heuristics(G, heuristics, timeout=300, name="No name provided", c=0.2, gnd=False)
# draw_results(results)
# test_gnd(G, timeout=300, name="No name provided", c=3)
# test_graphs(graphs, g_type, heuristics, timeout=300, c=3, gnd=False)

### CODE

# Generate a sea-urchin-sistine-chapel (SUSC) graph
def susc_generator(num_spikes, path_length, p=0):
    
    assert p <= 1
    assert p >= 0

    G = nx.Graph()
    k = 0 # constant to differentiate the two urchins
    
    # Urchin 1
    urchin_1 = 0
    G.add_node(urchin_1)
    k += 1

    for i in range(num_spikes):
        G.add_edge(k, urchin_1) # Add spike to urchin
        k += 1

    # Probabilistically add edges between spikes

    if p > 0:
        for i in range(num_spikes):
            for j in range(num_spikes):
                if i == j:
                    continue
                spike_1 = i + 1 + urchin_1
                spike_2 = j + 1 + urchin_1
                if random.uniform(0, 1) <= p:
                    G.add_edge(spike_1, spike_2)


    urchin_2 = k
    
    G.add_node(urchin_2)

    k += 1

    # Urchin 2


    for i in range(num_spikes):
        G.add_edge(k, urchin_2) # Add spike to urchin
        k += 1

    if p > 0:
        for i in range(num_spikes):
            for j in range(num_spikes):
                if i == j:
                    continue
                spike_1 = i + urchin_2 + 1
                spike_2 = j + urchin_2 + 1
                if random.uniform(0, 1) <= p:
                    G.add_edge(spike_1, spike_2)


    # Path

    if path_length == 0:
        G.add_edge(urchin_1, urchin_2)
    else:
        count = G.number_of_nodes() + 1
        start = count

        for i in range(path_length - 1):
            G.add_edge(count, count + 1)
            count += 1
        
        end = count

        G.add_edge(start, urchin_1)
        G.add_edge(end, urchin_2)

    return G

def assign_weights_randomly(G):
    H = G.copy()
    for n in H.nodes:
        H.nodes[n]['node weight'] = 100 * (1 - random.uniform(0, 1))
    for e in H.edges:
        H.edges[e]['edge weight'] = 100 * (1 - random.uniform(0, 1))
    return H

def assign_node_weights_randomly_edge_weights_sum(G):
    H = G.copy()
    for n in H.nodes:
        H.nodes[n]['node weight'] = 100 * (1 - random.uniform(0, 1))
    for e in H.edges:
        u, v = e
        H.edges[e]['edge weight'] = H.nodes[u]['node weight'] + H.nodes[v]['node weight']
    return H

def assign_node_weights_randomly(G):
    H = G.copy()
    for n in H.nodes:
        H.nodes[n]['node weight'] = np.random.randint(1,10)
    return H

def test_heuristics(G, heuristics, timeout=300, name="No name provided", c=0.2, gnd=False):
    results = []
    print("Graph:", name)
    print("n:", G.number_of_nodes(), "m:", G.number_of_edges())
    print("")
    for heuristic in heuristics:
        name = heuristic[0]
        h = heuristic[1]
        print("Testing", name)
        results.append((name, greedy_algorithm(h, G, c, timeout=timeout)))
        print("")

    return results

def draw_results(results):
    for result in results:
        name = result[0]
        S, graph = result[1]
        print("Results for", name)
        print("Set S of removed vertices:", S)
        print("Resulting graph, G':")
        nx.draw(graph)

def test_gnd(G, timeout=300, name="No name provided", c=3):
    results = []
    print("Graph:", name)
    print("n:", G.number_of_nodes(), "m:", G.number_of_edges())
    print("")
    results.append(gnd(G, c, False))
    print("")
    
    return results

def test_graphs(graphs, g_type, heuristics, timeout=300, c=3, gnd=False):
    graph_results = []
    print("--- Running tests for", g_type, "graphs ---")
    print("")
    
    for graph in graphs:
        name = graph[0]
        G = graph[1]
        if gnd:
            graph_results.append(test_gnd(G, timeout, name, c))
        else: 
            graph_results.append(test_heuristics(G, heuristics, timeout, name, c))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("")
    
    print("")
    print("--- All tests for", g_type, "graphs size have now finished ---")
    return graph_results