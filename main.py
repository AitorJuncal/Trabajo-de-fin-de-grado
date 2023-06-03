# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:01:23 2022

@author: Aitor
"""

"""imports"""
import sys
from collections import defaultdict
import utils_graph as utils
import predictor as pred
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import time
import random
import numpy as np
import preprocess as pre

inicio_total = time.time()
dataset = sys.argv[1]
tweets_train, tweets_test, clases_train, clases_test = pre.read_dataset(dataset)
print("dataset leido")
print(len(tweets_train)+len(tweets_test))
tweets_train = pre.preprocess(tweets_train, len(tweets_train))
tweets_test = pre.preprocess(tweets_test, len(tweets_test))
print("preprocesamiento completado")

acc = 0.0
pred_neg = 0.0
pred_pos = 0.0
rec_neg = 0.0
rec_pos = 0.0
f1_neg = 0.0
f1_pos = 0.0

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6))
negative_tweets = []
positive_tweets = []
for iteracion in range(1):
#for fold, (train, test) in enumerate(cv.split(tweets_train,clases_train)):
    for i in range(len(tweets_train)):
        if clases_train[i] == 0:
            negative_tweets.append(tweets_train[i])
        if clases_train[i] == 1:
            positive_tweets.append(tweets_train[i])
            
    distancia = 1
    positive_graph = utils.build_graph(positive_tweets,distancia)
    negative_graph = utils.build_graph(negative_tweets,distancia)

    classifier = MLPClassifier(solver='adam',hidden_layer_sizes = (32,128,128,128,32,8), random_state=1)
    
    dic_positivo = utils.eigenvector_centrality(positive_graph)
    dic_negativo = utils.eigenvector_centrality(negative_graph)
    bet_negativo = utils.betweenness_centrality(negative_graph, int(len(negative_graph.nodes())/20))
    bet_positivo = utils.betweenness_centrality(positive_graph, int(len(positive_graph.nodes())/20))
    print("calculo de metricas terminado")

    X = []
    for tweet in tweets_train:
        G = utils.build_graph([tweet],distancia)
        x0 = pred.eigen_vector(G, dic_negativo)
        x1 = pred.eigen_vector(G, dic_positivo)
        x2 = pred.similarity_nodos(G, negative_graph)
        x3 = pred.similarity_nodos(G, positive_graph)
        x4 = pred.similarity_aristas(G, negative_graph)
        x5 = pred.similarity_aristas(G, positive_graph)
        x6 = pred.betweenness(G, bet_negativo)
        x7 = pred.betweenness(G, bet_positivo)
        X.append([x0,x1,x2,x3,x4,x5,x6,x7])

    inicio_train = time.time()
    classifier.fit(X, clases_train)
    print("Tiempo tardado en el entrenamiento = ", time.time()-inicio_train)

    X = []
    for tweet in tweets_test:
        G = utils.build_graph([tweet],distancia)
        x0 = pred.eigen_vector(G, dic_negativo)
        x1 = pred.eigen_vector(G, dic_positivo)
        x2 = pred.similarity_nodos(G, negative_graph)
        x3 = pred.similarity_nodos(G, positive_graph)
        x4 = pred.similarity_aristas(G, negative_graph)
        x5 = pred.similarity_aristas(G, positive_graph)
        x6 = pred.betweenness(G, bet_negativo)
        x7 = pred.betweenness(G, bet_positivo)
        X.append([x0,x1,x2,x3,x4,x5,x6,x7])

    predicciones = classifier.predict(X)
    acc = accuracy_score(clases_test, predicciones)
    r = precision_recall_fscore_support(clases_test, predicciones, average=None, labels=[0,1])
    pred_neg = r[0][0]
    pred_pos = r[0][1]
    rec_neg = r[1][0]
    rec_pos = r[1][1]
    f1_neg = r[2][0]
    f1_pos = r[2][1]

    
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X,
        clases_test,
        name=f"ROC fold {iteracion}",
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    print("AUC -> ", viz.roc_auc)
    aucs.append(viz.roc_auc)
    print("Iteracion número ", iteracion+1, " completada.")
    if iteracion == 0:
        break
print("Accuracy -> ", acc)
print("Precisión negativa -> ", pred_neg)
print("Precisión positiva -> ", pred_pos)
print("recall negativo -> ", rec_neg)
print("recall positivo -> ", rec_pos)
print("f1 negativo -> ", f1_neg)
print("f1 positivo -> ", f1_pos)

ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label '{'clases'}')",
)
ax.axis("square")
ax.legend(loc="lower right")
#plt.show()

print("TIEMPO TOTAL = ", time.time()-inicio_total)
