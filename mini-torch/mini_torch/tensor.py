import uuid
import logging
import numpy as np

from .autograd.functions import *
from typing import Any



logger = logging.getLogger(__file__)


def tensor(value, name, requires_grad=None):
    if isinstance(value, (list, int, float)):
        tensor = np.array(value).view(Tensor)
    elif isinstance(value, np.ndarray):
        tensor = value.view(Tensor)
    else:
        tensor = np.array(value).view(Tensor)
    
    tensor.requires_grad = requires_grad
    tensor.grad = None
    tensor.grad_fn = None
    tensor.name = name
    tensor.value = value
    
    return tensor


class Tensor(np.ndarray):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requires_grad = None
        self.grad = None
        self.grad_fn = None
        self.name = None
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
            
            logger.info("current=%s", tr.name)
            
            if tr.grad is None:
                tr.grad = grad
            else:
                tr.grad += grad

            # Propagate gradients
            if tr.grad_fn is not None:
                grads = tr.grad_fn.backward(grad)               
                for t, grad in zip(tr.grad_fn.input, grads):
                    if isinstance(t, Tensor) and t.name not in visited:
                        stack.append((t, grad))
                        visited.add(t.name)

    def zero_grad(self):
        self.grad = None
        
    def detach(self):
        self.grad = None
        self.grad_fn = None
        
    def __getitem__(self, indices):
        value = super().__getitem__(indices)
        return value
    
    def __str__(self) -> str:
        return super().__str__()
    
    def __repr__(self) -> str:
        return super().__str__()
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, uuid.uuid4(), requires_grad=False)
        result = np.add(self, other)
        requires_grad = self.requires_grad or other.requires_grad
        result = tensor(result, uuid.uuid4(), requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = AddBackward(self, other)
        
        return result

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, uuid.uuid4(), requires_grad=False)
        result = np.add(other, self)
        requires_grad = self.requires_grad or other.requires_grad
        result = tensor(result, uuid.uuid4(), requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = AddBackward(other, self)
        
        return result
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, uuid.uuid4(), requires_grad=False)
        result = np.subtract(self, other)
        requires_grad = self.requires_grad or other.requires_grad
        result = tensor(result, uuid.uuid4(), requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SubBackward(self, other)

        return result
    
    def __rsub__(self, other):
        if isinstance(self, (int, float)):
            self = tensor(self, uuid.uuid4(), requires_grad=False)
        result = np.subtract(other, self)
        requires_grad = self.requires_grad or other.requires_grad
        result = tensor(result, uuid.uuid4(), requires_grad=requires_grad)
        
        if result.requires_grad:
            result.grad_fn = SubBackward(other, self)

        return result
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = tensor(other, uuid.uuid4(), requires_grad=False)
            result = np.dot(self, other)
            requires_grad = self.requires_grad or other.requires_grad
            result = tensor(result, uuid.uuid4(), requires_grad=requires_grad)
            
            if result.requires_grad:
                result.grad_fn = ScalarMulBackward(self, other)
                
            return result
        
        elif isinstance(other, Tensor):
            result = np.multiply(self, other)
            requires_grad = self.requires_grad or other.requires_grad
            result = tensor(result, uuid.uuid4(), requires_grad=requires_grad)
            
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
            requires_grad = self.requires_grad or other.requires_grad
            result = tensor(result, uuid.uuid4(), requires_grad=requires_grad)
            
            if result.requires_grad:
                result.grad_fn = MatmulBackward(self, other)
            return result
