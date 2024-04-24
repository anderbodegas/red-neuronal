import numpy as np

from funciones import diccionarioFunciones
from capa import Capa

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

class RedNeuronal():
    
    def __init__(self, fCoste: str = 'errorCuadratico'):
        """
        Este método inicializa los objetos de la clase RedNeuronal.

        Parámetros
        ----------
        fCoste : str, opcional
            Función de coste de la red. Por defecto es 'errorCuadratico'.

        Devuelve
        -------
        None.

        """
        self.fCoste = diccionarioFunciones[fCoste]
        self.red = []
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def adjuntarCapa(self, nConexiones: int, nNeuronas: int,
                    fActivacion: str = 'sigmoide'):
        """
        Este método adjunta una nuevo objeto de la clase capa a la red. 

        Parámetros
        ----------
        nConexiones : int
            Número de conexiones de la capa, es decir, el número de neuronas
            de la capa anterior.
        nNeuronas : int
            Número de neuronas de la capa.
        fActivacion : str
            Función de activación de la capa ('sigmoide', 'tanH' o 'ReLu').
            Por defecto es 'sigmoide'.

        Devuelve
        -------
        None.

        """
        self.red.append(Capa(nConexiones, nNeuronas, fActivacion))
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def procesarRed(self, X: np.array):
        """
        Este método realiza los cálculos de la red con unos datos de entrada.

        Parámetros
        ----------
        X : np.array
            Matriz de datos de entrada, debe ser de dimensiones
            (nObservaciones x nVariables).

        Devuelve
        -------
        A : np.array
            Matriz resultado del cálculo de la red. Tiene dimensiones
            (nObservaciones x 1).

        """
        A = X
        for capa in self.red:
            _, A = capa.procesarCapa(A)
        return A
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def calcularCoste(self, X: np.array, y: np.array):
        """
        Este método calcula el coste medio de todas las observaciones.

        Parámetros
        ----------
        X : np.array
            Matriz de datos de entrada, debe ser de dimensiones
            (nObservaciones x nVariables).
        y : np.array
            Matriz de datos de respuesta, debe ser de dimensiones
            (nObservaciones x 1).

        Devuelve
        -------
        float
            El valor del coste medio.

        """
        return np.mean(self.fCoste(self.procesarRed(X), y))
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def inicializarGradientes(self):
        """
        Este método inicializa los gradientes de los sesgos (b) y pesos (W) de
        todas las capas de la red a 0.

        Devuelve
        -------
        None.

        """
        for capa in self.red:
            capa.inicializarGradientes()
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    def actualizarGradientes(self, X: np.array, y: np.array):
        """
        Este método actualiza los gradientes de los sesgos (b) y pesos (W) de
        todas las capas de la red mediante la regla de la cadena.

        Parámetros
        ----------
        X : np.array
            Matriz de datos de entrada, debe ser de dimensiones
            (nObservaciones x nVariables).
        y : np.array
            Matriz de datos de respuesta, debe ser de dimensiones
            (nObservaciones x 1).

        Devuelve
        -------
        None.

        """
        listaZ = []
        listaA = [X]
        for capa in self.red:
            Z, A = capa.procesarCapa(listaA[-1])
            listaZ.append(Z)
            listaA.append(A)
        for l in range(1, len(self.red) + 1):
            if l == 1:
                delta = self.fCoste(listaA[-l], y, derivada = True)
            else:
                delta = np.dot(delta, self.red[-l + 1].W.transpose())
            delta *= self.red[-l].fActivacion(listaZ[-l], derivada = True)
            self.red[-l].gradienteb = np.sum(delta, axis = 0, keepdims = True)
            self.red[-l].gradienteW = np.dot(listaA[-l - 1].transpose(), delta)
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            
    def actualizarParametros(self, ratioAprendizaje: float):
        """
        Este método actualiza los sesgos (b) y los pesos (W) de todas las
        capas de la red mediante el método de descenso de gradiente. Para ello
        los gradientes han de estar actualizados con el cálculo apropiado, ya
        que de valer 0 este método no hará nada.

        Parámetros
        ----------
        ratioAprendizaje : float
            Valor que regula la velocidad a la que se efectúa el descenso de
            gradiente. Un valor muy bajo puede conllevar un descenso muy lento
            y un valor muy alto puede hacer que el método no converja.

        Devuelve
        -------
        None.

        """
        for capa in self.red:
            capa.actualizarParametros(ratioAprendizaje)
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            
    def entrenarRed(self, X: np.array, y: np.array,
                    ratioAprendizaje: float = 0.01, nIteraciones: int = 500):
        """
        Este método entrena la red aplicando de forma iterativa el método de
        descenso de gradiente sobre los sesgos (b) y pesos (W) de todas las
        capas de la red.

        Parámetros
        ----------
        X : np.array
            Matriz de datos de entrada, debe ser de dimensiones
            (nObservaciones x nVariables).
        y : np.array
            Matriz de datos de respuesta, debe ser de dimensiones
            (nObservaciones x 1).
        ratioAprendizaje : float
            Valor que regula la velocidad a la que se efectúa el descenso de
            gradiente. Un valor muy bajo puede conllevar un descenso muy lento
            y un valor muy alto puede hacer que el método no converja.
            Por defecto es 0.01.
        nIteraciones : int, opcional
            Número de veces a aplicar el descenso de gradiente.
            Por defecto es 500.

        Devuelve
        -------
        None.

        """
        for _ in range(nIteraciones):
            self.inicializarGradientes()
            self.actualizarGradientes(X, y)
            self.actualizarParametros(ratioAprendizaje)
