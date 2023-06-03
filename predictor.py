#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:12:17 2022

@author: aitor
"""

def similarity_nodos(G_consulta, G_referencia):
    if len(G_consulta.nodes) == 0:
        return 0.0
    comun = 0
    nodos_consulta = G_consulta.nodes
    nodos_referencia = G_referencia.nodes
    for nodo in nodos_consulta:
    	if nodo in nodos_referencia:
    		comun += 1
                
    return comun / len(G_consulta.nodes)

def similarity_aristas(G_consulta, G_referencia):
    if len(G_consulta.edges) == 0:
        return 0.0
    aristas_consulta = G_consulta.edges
    aristas_referencia = G_referencia.edges
    comun = 0
    for arista in aristas_consulta:
        reversed = (arista[1],arista[0])
        if arista in aristas_referencia or reversed in aristas_referencia:
            comun += 1
    return comun / len(G_consulta.edges) 

        
def betweenness(G_tweet, dic):
    b = 0
    for arista in G_tweet.edges:
        if arista in dic:
            b += dic[arista]
    return float(b)

def eigen_vector(G_tweet, dic):
    v = 0
    for nodo in G_tweet.nodes:
        if nodo in dic:
            v += dic[nodo]
    return float(v)
    
def precision(realidad,prediccion):
    aciertos = 0
    N = len(realidad)
    for i in range(N):
        if realidad[i] == prediccion[i]:
            aciertos += 1
    return aciertos / N
