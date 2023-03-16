"""En este módulo se crean las funciones que se utilizan como utilidades durante
el desarrollo del proyecto"""

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
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
def metodo_silueta(n_grupos:int, dataset:pd.DataFrame, agrupamiento:str='kmeans') -> None:
    """Función encargada de graficar el puntaje del coeficiente de siluetas vs
    el número de cluster (k)
    
    Parámetros de entrada:
    
    n_grupos:int -> Número de grupos que quiero visualizar
    dataset:pd.DataFrame -> Dataset al cual se le aplicará el análisis
    agrupamiento:str -> especifica el tipo de agrupamiento al cual se le aplicará el
    método de siluetas. Las opciones son las siguientes: 'kmeans' o 'aglomerativo'"""
    
    if agrupamiento not in ['kmeans','aglomerativo']:
        raise Exception('Error en la selección del tipo de agrupamiento')
    
    coef_siluetas_score = []

    if agrupamiento == 'kmeans':
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
    elif agrupamiento == 'aglomerativo':
        for i in range(2, n_grupos+1):
            ac = AgglomerativeClustering(
                n_clusters=i,
                metric='euclidean',
                linkage='complete'
            )
            y_ac = ac.fit_predict(X=dataset)
            coef_siluetas_score.append(silhouette_score(X=dataset, labels=y_ac))
                
    plt.plot(range(2, n_grupos+1), coef_siluetas_score, marker='o')
    plt.title("Coeficiente de silueta Vs Número de cluster's")
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Puntaje coeficiente de silueta\nPuntaje máximo: {:.2f}'.format(max(coef_siluetas_score)))
    plt.grid()
    plt.show()
