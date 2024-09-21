import numpy as np
import logging


logger = logging.getLogger(__file__)


class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]
    
class SubBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, -gradient]

class ScalarMulBackward:
    def __init__(self, x, scalar):
        self.input = [x]
        self.scalar = scalar

    def backward(self, gradient):
        return [gradient * self.scalar]

class ElementwiseMulBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x = self.input[0]
        y = self.input[1]
        return [y * gradient, x * gradient]

class TransposeBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        return [np.transpose(gradient)]

class MatmulBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x, y = self.input
        if x.ndim != y.ndim:
            x = np.expand_dims(x, axis=0)
            gradient = np.expand_dims(gradient, axis=0)            
            r1 = np.squeeze(np.matmul(y, np.transpose(gradient)))
            r2 = np.matmul(np.transpose(x), gradient)            
            return [r1, r2]
        else:
            r1 = np.matmul(gradient, np.transpose(y))
            r2 = np.matmul(np.transpose(x), gradient)
            return [r1, r2]

class PowBackward:
    def __init__(self, base, exp):
        self.input = [base, exp]
        
    def backward(self, gradient):
        base, exp = self.input
        
        if isinstance(base, (int, float)):
            grad_base = np.multiply(gradient, np.pow(base, exp - 1))
            grad_exp = np.multiply(np.multiply(gradient, np.pow(base, exp)), np.log(base))
        
        else:
            grad_base = np.multiply(np.multiply(gradient, exp), np.pow(base, exp - 1))
            grad_exp = np.multiply(np.multiply(gradient, np.pow(base, exp)), np.log(base))
            
        return [grad_base, grad_exp]
    
class LogBackward:
    def __init__(self, x):
        self.input = [x]
        
    def backward(self, gradient):
        grad_input = np.divide(gradient, self.input[0])
        return [grad_input]

class DivisionBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x, y = self.input
        
        grad_x = np.divide(gradient, y)
        grad_y = np.multiply(-1, np.multiply(gradient, np.divide(x, np.multiply(y, y))))

        return [grad_x, grad_y]