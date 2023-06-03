# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:23:37 2022

@author: Aitor
"""
import networkx as nx

"""funciones"""
def build_graph(tweets,distancia):
    """Building graph"""
    G = nx.Graph()
    grafos = []
    for tweet in tweets:
        F = nx.Graph()
        for word in tweet:
            F.add_node(word[0])
        nodos = list(F.nodes)
        for i in range(len(nodos)):
            for j in range(i+1, i+distancia+1):
                if j < len(nodos):
                    F.add_edge(nodos[i],nodos[j])
        grafos.append(F)
    G = nx.compose_all(grafos)
    return G

def grado(G):
    return dict(G.degree())

def betweenness_centrality(G, n):
    return nx.edge_betweenness_centrality(G, k=n)

def eigenvector_centrality(G):
    return nx.eigenvector_centrality(G)
    
def graph_intersection(G,F):
    return nx.intersection(G,F)
        
def draw_graph(G, file):
    A = nx.nx_agraph.to_agraph(G)
    A.layout('dot')
    A.draw(file)
