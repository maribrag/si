import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import numpy as np
from numpy.random import binomial
from src.si.Neural_networks.layers import Layer

class Dropout(Layer):
    def __init__(self, probability):
        """
        Initialize the Dropout layer.

        Parameters
        ----------
        probability: float
            The dropout rate, between 0 and 1.
        """
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input

        if training:
            scaling_factor = 1.0 / (1.0 - self.probability)
            self.mask = binomial(n=1, p=1 - self.probability, size=input.shape)
            self.output = input * self.mask * scaling_factor
        else:
            self.output = input  # During inference, return the input unchanged

        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        numpy.ndarray
            The input error of the layer.
        """
        return output_error * self.mask if self.mask is not None else output_error

    def output_shape(self) -> tuple:
        """
        Returns the input shape.

        Returns
        -------
        tuple
            The shape of the input to the layer.
        """
        return self.input_shape()

    def parameters(self) -> int:
        """
        Returns 0 as dropout layers do not have learnable parameters.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
    
    
if __name__ == '__main__':
    # Criando uma instância de Dropout com probabilidade 0.25
    dropout_layer = Dropout(probability=0.25)

    # Definindo um array de entrada para teste
    input_array = np.array([[1, 2, 3], [4, 5, 6]])

    # Realizando a propagação direta (forward propagation) no modo de treinamento
    output_train = dropout_layer.forward_propagation(input_array, training=True)
    print("Output in training mode:")
    print(output_train)

    # Realizando a propagação direta (forward propagation) no modo de inferência
    output_inference = dropout_layer.forward_propagation(input_array, training=False)
    print("\nOutput in inference mode:")
    print(output_inference)

    # Realizando a propagação inversa (backward propagation)
    output_error = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    backward_output = dropout_layer.backward_propagation(output_error)
    print("\nOutput of backward propagation:")
    print(backward_output)