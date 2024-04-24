import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# f: R --> [0, 1]
def sigmoide(x: np.array, derivada: bool = False):
    """
    Función sigmoide y su derivada. Esta función mapea toda la recta real en el
    intervalo [0, 1], lo cual es útil para representar un comportamiento de
    'no activación' / 'activación' en las neuronas de la capa en la que se use
    como función de activación.

    Parámetros
    ----------
    x : np.array
        Matriz con los valores de entrada.
    derivada : bool, opcional
        Valor booleano que indica si se desea aplicar la primera derivada de la
        función. Por defecto es False.

    Devuelve
    -------
    np.array
        Imagen de la matriz de entrada.

    """
    if derivada: return (sigmoide(x) * (1 - sigmoide(x)))
    return (1 / (1 + np.exp(-x)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# f: R --> [-1, 1]
def tanH(x: float, derivada: bool = False):
    """
    Función tangente hiperbólica y su derivada. Esta función mapea toda la
    recta real en el intervalo [-1, 1], lo cual es útil para representar un
    comportamiento de 'activación negativa / no activación' / 'activación
    positiva' en las neuronas de la capa en la que se use como función de
    activación.

    Parámetros
    ----------
    x : np.array
        Matriz con los valores de entrada.
    derivada : bool, opcional
        Valor booleano que indica si se desea aplicar la primera derivada de la
        función. Por defecto es False.

    Devuelve
    -------
    np.array
        Imagen de la matriz de entrada.

    """
    if derivada: return (1 - tanH(x) ** 2)
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# f: R --> R+
def ReLU(x: np.array, derivada: bool = False):
    """
    Función ReLU ('Rectified Linear Unit') y su derivada. Esta función es nula
    para valores negativos y la función identidad para el resto de valores.

    Parámetros
    ----------
    x : np.array
        Matriz con los valores de entrada.
    derivada : bool, opcional
        Valor booleano que indica si se desea aplicar la primera derivada de la
        función. Por defecto es False.

    Devuelve
    -------
    resultado : np.array
        Imagen de la matriz de entrada.

    """
    resultado = x.copy()
    if derivada: return 1 * (resultado > 0)
    resultado[resultado < 0] = 0
    return resultado

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def errorCuadratico(x: np.array, y: np.array, derivada = False):
    """
    Función que calcula elemento a elemento la diferencia al cuadrado de dos
    matrices de la misma dimensión. La derivada se calcula con respecto a x.

    Parámetros
    ----------
    x : np.array
        Primera matriz con los valores de entrada.
    y : np.array
        Segunda matriz con los valores de entrada.
    derivada : bool, opcional
        Valor booleano que indica si se desea aplicar la primera derivada de la
        función. Por defecto es False.

    Devuelve
    -------
    np.array
        Imagen de las matrices de entrada.

    """
    if derivada: return (2 * (x - y))
    return ((x - y) ** 2)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

diccionarioFunciones = {
    'sigmoide': sigmoide,
    'tanH': tanH,
    'ReLU': ReLU,
    'errorCuadratico': errorCuadratico
    }