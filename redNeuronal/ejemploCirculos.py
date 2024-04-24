"""
En este ejemplo creamos y entrenamos una red neuronal para separar dos nubes de
puntos en forma de círculo. Como se puede ver, la red tiene 3 capas de 2, 3 y 1
neuronas. La segunda capa tiene como función de activación la tangente
hiperbólica, mientras que la última tiene la función sigmoide, ya que
necesitamos que el resultado de las predicciones esté entre 0 y 1. La primera
capa simplemente almacena los valores de entrada, así que no tiene función de
activación.

"""

import numpy as np
import matplotlib.pyplot as plt

from redNeuronal import RedNeuronal
from sklearn.datasets import make_circles
from IPython.display import clear_output

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Variables globales
np.random.seed(1234)
nObservaciones = 500
nPuntosLadoCuadricula = 50
nFotogramas = 75
nIteraciones = 5 # Aumentar si se ve muy lento

# Creamos los dos círculos con la función make_circles
X, Y = make_circles(n_samples = nObservaciones, factor = 0.4, noise = 0.1)
Y = Y[:, np.newaxis]

# Mostramos los círculos
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c = 'skyblue')
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c = 'salmon')
plt.axis('equal')
plt.show()

# Creamos la red
rn = RedNeuronal()
rn.adjuntarCapa(2, 3, 'tanH')
rn.adjuntarCapa(3, 1, 'sigmoide')

# Preparamos la cuadrícula sobre la que iremos mostrando las predicciones
_x = np.linspace(-1.5, 1.5, nPuntosLadoCuadricula)
_y = np.linspace(-1.5, 1.5, nPuntosLadoCuadricula)
costes = []

for i in range(nFotogramas):
    
    # Entrenamos con 1 iteración para que vaya mostrando el avance poco a poco
    rn.entrenarRed(X, Y, ratioAprendizaje = 0.01, nIteraciones = nIteraciones)
    
    # Calculamos y aadimos el nuevo coste
    costes.append(rn.calcularCoste(X, Y))
    
    # Limpiamos la cuadrícula
    _Z = np.zeros((nPuntosLadoCuadricula, nPuntosLadoCuadricula))
    
    # Rellenamos la cuadrícula con la predicción de la red
    for i, x in enumerate(_x):
        for j, y in enumerate(_y):
            _Z[j, i] = rn.procesarRed(np.array([x, y]).reshape(1, 2))[0][0]
    
    # Mostramos el avance de los costes (comentar bloque para vídeo más guay)
    # plt.plot(range(len(costes)), costes)
    # plt.show()
    
    # Mostramos el avance de la predicción
    plt.pcolormesh(_x, _y, _Z, cmap = 'coolwarm')
    plt.axis('equal')
    plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c = 'skyblue')
    plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c = 'salmon')
    clear_output(wait = True)
    plt.show()
