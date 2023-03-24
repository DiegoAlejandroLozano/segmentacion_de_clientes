## Segmentación de clientes

Librerías utilizadas: `Pandas`, `Numpy`, `Matplotlib`, `Seaborn` y `Scikit-Learn`

En este proyecto se trabaja con la segmentación de clientes de un supermercado. Dentro del conjunto de datos se tiene como características el ID del cliente, la edad, el género, los ingresos anuales y un puntaje de gasto (el dataset se encuentra ubicado en el siguiente [link](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)). Para realizar la segmentación de los clientes se utilizó K-means y el algoritmo de agrupamiento jerárquico aglomerativo. Como métrica de desempeño se utilizó el método de las siluetas. 

Para correr el repositorio de forma local se debe crear un entorno virtual con el siguiente comando:

    Python3 -m venv nombre_entorno_virtual

Se debe activar el entorno virtual e instalar las librerías requeridas a través del archivo requirements.txt

    pip install -r requirements.txt
