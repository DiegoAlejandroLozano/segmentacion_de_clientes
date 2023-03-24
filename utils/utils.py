"""En este módulo se crean las funciones que se utilizan como utilidades durante
el desarrollo del proyecto"""

import pandas 

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def metodo_elbow_kmeans(n_grupos:int, dataset:pandas.DataFrame) -> None:
    """Función encargada de graficar la distorsión vs el número de grupos
    
    Parámetros de entrada:
    
    n_grupos:int --> Número de grupos que quiero visualizar
    dataset:pandas.DataFrame --> Dataset al cual se le aplicará el análisis"""

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

def metodo_silueta(n_grupos:int, dataset:pandas.DataFrame, agrupamiento:str='kmeans') -> None:
    """Función encargada de graficar el puntaje del coeficiente de siluetas vs
    el número de cluster (k) para los algoritmos de kmeans y AgglomerativeClustering
    
    Parámetros de entrada:
    
    n_grupos:int --> Número de grupos que quiero visualizar
    dataset:pandas.DataFrame --> Dataset al cual se le aplicará el análisis
    agrupamiento:str --> Especifica el tipo de agrupamiento al cual se le aplicará el
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
                
    print('Puntaje máximo: {:.4f}'.format(max(coef_siluetas_score)))
    print('Número de cluster: {}'.format(coef_siluetas_score.index(max(coef_siluetas_score))+2))

    plt.plot(range(2, n_grupos+1), coef_siluetas_score, marker='o')
    plt.title("Coeficiente de silueta Vs Número de cluster's")
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Puntaje coeficiente de silueta')
    plt.grid()
    plt.show()

def pca_componentes_optimos(dataset:pandas.DataFrame)->None:
    """Función encargada de graficar la varianza acumulada de todas
    las variables de un dataset
    
    Parámetros:
    
    dataset:pandas.DataFrame --> dataset al cual se le calculará la varianza
    acumulada de sus variables"""

    pca = PCA(n_components=None)
    pca.fit(dataset)

    suma_acumulada = pca.explained_variance_ratio_.cumsum()*100

    #Mostrando la suma acumulada del ratio de la varianza explicada
    print('Suma acumulada de la varianza explicada (%): ')
    print(suma_acumulada)

    #Creando un plot para mostrar el impacto de la cantidad de componentes
    plt.plot(suma_acumulada, marker='o')
    plt.xlabel('Número de componentes (dimensiones)')
    plt.ylabel('Suma acumulada varianza explicada (%)')
    plt.grid()
    plt.show()
