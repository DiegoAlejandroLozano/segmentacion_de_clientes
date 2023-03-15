"""En este módulo se crean las funciones que se utilizan como utilidades durante
el desarrollo del proyecto"""

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Construyendo una función que ejecute el método del elbow
def metodo_elbow(n_grupos:int, dataset:pd.DataFrame) -> None:
    """Función encargada de graficar la distorsión vs el número de grupos
    
    Parámetros de entrada:
    
    n_grupos:int -> Número de grupos que quiero visualizar
    dataset:pd.DataFrame -> Dataset al cual se le aplicará el análisis"""

    distortions = []
    for i in range(1,n_grupos+1):
        km = KMeans(
            n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0
        )

        km.fit(X=dataset)
        distortions.append(km.inertia_)

    plt.plot(range(1,n_grupos+1), distortions, marker='o')
    plt.title('Método elbow')
    plt.xlabel('Número de clusters')
    plt.ylabel('Distortion')
    plt.grid()
    plt.show()

#Construyendo una función que ejecute el método de las siluetas
def metodo_silueta(n_grupos:int, dataset:pd.DataFrame) -> None:
    """Función encargada de graficar el puntaje del coeficiente de siluetas vs
    el número de cluster (k)
    
    Parámetros de entrada:
    
    n_grupos:int -> Número de grupos que quiero visualizar
    dataset:pd.DataFrame -> Dataset al cual se le aplicará el análisis"""
    coef_siluetas_score = []
    for i in range(2, n_grupos+1):
        km = KMeans(
            n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0
        )
        y_km = km.fit_predict(X=dataset)
        coef_siluetas_score.append(silhouette_score(dataset, y_km, metric='euclidean'))
    
    plt.plot(range(2, n_grupos+1), coef_siluetas_score, marker='o')
    plt.title("Coeficiente de silueta Vs Número de cluster's")
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Puntaje coeficiente de silueta')
    plt.grid()
    plt.show()
