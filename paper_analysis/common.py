import os, pickle, networkx as nx
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'city_cache')
def load_graph(name):
    G = nx.Graph(pickle.load(open(os.path.join(CACHE, name + '.pkl'), 'rb')))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G
