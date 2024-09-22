import string
import secrets
import logging
import numpy as np

from .autograd.functions import *

logger = logging.getLogger(__file__)


def tensor(value, requires_grad=None):
    if isinstance(value, np.ndarray):
        tensor = value.view(Tensor)
    else:
        tensor = np.array(value).view(Tensor)
    
    tensor.requires_grad = requires_grad
    tensor.grad = None
    tensor.grad_fn = None
    tensor.id = ''.join(secrets.choice(string.ascii_letters) for _ in range(6))
    tensor.value = value
    
    return tensor


class Tensor(np.ndarray):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requires_grad = None
        self.grad = None
        self.grad_fn = None
        self.id = None
        self.value = None

    def backward(self, gradient=None):
        if not self.requires_grad:
            return

        if gradient is None:
            if self.shape == ():
                gradient = tensor(1, 'root')
            else:
                raise RuntimeError()
        else:
          gradient = tensor(gradient, 'root')

        stack = [(self, gradient)]
        visited = set()

        while stack:
            tr, grad = stack.pop()
            
            if tr.grad is None:
                tr.grad = grad
            else:
                tr.grad += grad

            # Propagate gradients
            if tr.grad_fn is not None:
                grads = tr.grad_fn.backward(grad)
                for t, grad in zip(tr.grad_fn.input, grads):
                    if isinstance(t, Tensor) and t.id not in visited:
                        stack.append((t, grad))
                        visited.add(t.id)

    def zero_grad(self):
        self.grad = None
        
    def detach(self):
        self.grad = None
        self.grad_fn = None
        
    def __requires_grad(self, first, second):
        if not hasattr(first, 'requires_grad'):
            if isinstance(first, (int, float)):
                first = tensor(first)
            else:
                first.requires_grad = None
        if not hasattr(second, 'requires_grad'):
            if isinstance(second, (int, float)):
                second = tensor(second)
            else:
                second.requires_grad = None
        
        return first.requires_grad or second.requires_grad
        
    def __getitem__(self, indices):
        value = super().__getitem__(indices)
        return value
    
    def __str__(self) -> str:
        return super().__str__()
    
    def __repr__(self) -> str:
        return super().__str__()
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, requires_grad=False)
        result = np.add(self, other)
        requires_grad = self.__requires_grad(self, other)
        result = tensor(result, requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = AddBackward(self, other)
        
        return result

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, requires_grad=False)            
        result = np.add(other, self)
        requires_grad = self.__requires_grad(self, other)
        result = tensor(result, requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = AddBackward(other, self)
        
        return result

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, requires_grad=False)
        result = np.subtract(self, other)
        requires_grad = self.__requires_grad(self, other)
        result = tensor(result, requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SubBackward(self, other)

        return result

    def __rsub__(self, other):
        if isinstance(self, (int, float)):
            self = tensor(self, requires_grad=False)
        result = np.subtract(other, self)
        requires_grad = self.__requires_grad(self, other)
        result = tensor(result, requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SubBackward(other, self)

        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, requires_grad=False)
            result = np.dot(self, other)
            requires_grad = self.__requires_grad(self, other)
            result = tensor(result, requires_grad=requires_grad)
            
            if result.requires_grad:
                result.grad_fn = ScalarMulBackward(self, other)
                
            return result
        
        elif isinstance(other, Tensor):
            result = np.multiply(self, other)
            requires_grad = self.__requires_grad(self, other)
            result = tensor(result, requires_grad=requires_grad)
            
            if result.requires_grad:
                result.grad_fn = ElementwiseMulBackward(self, other)
                
            return result
        else:
            raise TypeError()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            result = np.matmul(self, other)
            requires_grad = self.__requires_grad(self, other)
            result = tensor(result, requires_grad=requires_grad)
            
            if result.requires_grad:
                result.grad_fn = MatmulBackward(self, other)
            return result

    def __pow__(self, other):
        result = np.pow(self, other)
        requires_grad = self.__requires_grad(self, other)
        result = tensor(result, requires_grad=requires_grad)
            
        if result.requires_grad:
            result.grad_fn = PowBackward(self, other)
        return result

    def __rpow__(self, other):
        result = np.pow(self, other)
        requires_grad = self.__requires_grad(self, other)
        result = tensor(result, requires_grad=requires_grad)
            
        if result.requires_grad:
            result.grad_fn = PowBackward(other, self)
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = float(other)
            result = np.divide(self, other)
            requires_grad = self.__requires_grad(self, other)
            result = tensor(result, requires_grad=requires_grad)
            
            if result.requires_grad:
                result.grad_fn = DivisionBackward(self, other)
        
        elif isinstance(self, Tensor) and isinstance(other, Tensor):
            result = np.divide(self, other)
            requires_grad = self.__requires_grad(self, other)
            result = tensor(result, requires_grad=requires_grad)
            
            if result.requires_grad:
                result.grad_fn = DivisionBackward(self, other)

        return result

    def __rtruediv__(self, other):
        other = float(other)
        result = np.divide(self, other)
        requires_grad = self.__requires_grad(self, other)
        result = tensor(result, requires_grad=requires_grad)

        if result.requires_grad:
            result.grad_fn = DivisionBackward(other, self)

        return result

    def log(self):
        result = np.log(self)
        result = tensor(result, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.grad_fn = LogBackward(self)

        return result
    
    def sigmoid(self):
        result = np.divide(1, np.add(1, np.exp(np.multiply(-1, self))))
        result = tensor(result, requires_grad=self.requires_grad)

        if result.requires_grad:
            result.grad_fn = SigmoidBackward(self)

        return result

    def sum(self, axis=None, keepdim=False):
        result = np.sum(self.view(np.ndarray), axis=axis, keepdims=keepdim)
        result = tensor(result, requires_grad=self.requires_grad)

        if result.requires_grad:
            result.grad_fn = SumBackward(self)

        return result

    @property
    def T(self):
        result = np.transpose(self)
        result = tensor(result, requires_grad=self.requires_grad)

        if result.requires_grad:
            result.grad_fn = TransposeBackward(self)
            
        return result
    