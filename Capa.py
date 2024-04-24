import numpy as np

from funciones import diccionarioFunciones

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

class Capa():
   
    def __init__(self, nConexiones: int, nNeuronas: int, fActivacion: str):
        """
        Este método inicializa los objetos de la clase Capa.

        Parámetros
        ----------
        nConexiones : int
            Número de conexiones de la capa, es decir, el número de neuronas
            de la capa anterior.
        nNeuronas : int
            Número de neuronas de la capa.
        fActivacion : str
            Función de activación de la capa ('sigmoide', 'tanH' o 'ReLu').

        Devuelve
        -------
        None.

        """
        self.nConexiones = nConexiones
        self.nNeuronas = nNeuronas
        self.fActivacion = diccionarioFunciones[fActivacion]
        self.b = np.random.rand(1, self.nNeuronas)
        self.W = np.random.rand(self.nConexiones, self.nNeuronas)
        self.inicializarGradientes()
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def inicializarGradientes(self):
        """
        Este método inicializa los gradientes de los sesgos (b) y pesos (W) de
        la capa a 0.

        Devuelve
        -------
        None.

        """
        self.gradienteb = np.zeros(self.b.shape)
        self.gradienteW = np.zeros(self.W.shape)
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        
    def procesarCapa(self, X: np.array):
        """
        Este método realiza los cálculos de la capa con los datos de la capa
        anterior.

        Parámetros
        ----------
        X : np.array
            Matriz de datos de la capa anterior, debe ser de dimensiones
            (nObservaciones x nConexiones)

        Devuelve
        -------
        Z : np.array
            Matriz resultado del cálculo lineal de la capa. Tiene dimensiones
            (nObservaciones x nNeuronas)
        A : np.array
            Matriz resultado del cálculo no lineal de la capa. Tiene
            dimensiones (nObservaciones x nNeuronas).

        """
        Z = np.dot(X, self.W) + self.b
        A = self.fActivacion(Z)
        return Z, A
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def actualizarParametros(self, ratioAprendizaje: float):
        """
        Este método actualiza los sesgos (b) y los pesos (W) de la capa
        mediante el método de descenso de gradiente. Para ello los gradientes
        han de estar actualizados con el cálculo apropiado, ya que de valer 0
        este método no hará nada.

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
        self.b -= ratioAprendizaje * self.gradienteb
        self.W -= ratioAprendizaje * self.gradienteW