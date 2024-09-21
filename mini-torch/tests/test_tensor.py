import pytest
import logging
import numpy as np


import mini_torch.tensor as tr


logger = logging.getLogger(__file__)

def test_add_scalar_and_tensor():
    x = np.array(20)
    
    x = tr.tensor(x, "x", requires_grad=True)
        
    z = 10 + x
    
    z.backward()
    
    logger.info("z = %s x.grad = %s", z, x.grad)
    
    
def test_add_scalar_and_ndim_tensor():
    x = np.array([1,2,3])
    
    x = tr.tensor(x, "x", requires_grad=True)
        
    z = 10 + x
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s", z, x.grad)

    
def test_add_two_tensor():
    x = np.array(20)
    y = np.array(30)
    
    x = tr.tensor(x, "x", requires_grad=True)
    y = tr.tensor(y, "y", requires_grad=True)
    
    z = x + y
    
    z.backward()
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)
    
    
def test_add_ndim_tensor_and_tensor():
    x = np.array([1,2,3,4,5])
    y = np.array(5)
    
    x = tr.tensor(x, "x", requires_grad=True)
    y = tr.tensor(y, "y", requires_grad=True)
    
    z = x + y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)
    
    
def test_add_ndim_tensor_and_ndim_tensor():
    x = np.array([1,2,3,4,5])
    y = np.array([2,4,6,8,10])
    
    x = tr.tensor(x, "x", requires_grad=True)
    y = tr.tensor(y, "y", requires_grad=True)
    
    z = x + y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)
    
    
def test_sub_scalar_and_tensor():
    x = np.array(5)
    
    x = tr.tensor(x, "x", requires_grad=True)
        
    z = 10 - x
    
    z.backward()
    
    logger.info("z = %s x.grad = %s", z, x.grad)
    
    
def test_sub_scalar_and_ndim_tensor():
    x = np.array([1,2,3])
    
    x = tr.tensor(x, "x", requires_grad=True)
        
    z = 10 - x
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s", z, x.grad)

    
def test_sub_two_tensor():
    x = np.array(40)
    y = np.array(30)
    
    x = tr.tensor(x, "x", requires_grad=True)
    y = tr.tensor(y, "y", requires_grad=True)
    
    z = x - y
    
    z.backward()
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)
    
    
def test_sub_ndim_tensor_and_tensor():
    x = np.array([10,20,30,40,50])
    y = np.array(5)
    
    x = tr.tensor(x, "x", requires_grad=True)
    y = tr.tensor(y, "y", requires_grad=True)
    
    z = x - y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)
    
    
def test_sub_ndim_tensor_and_ndim_tensor():
    x = np.array([10,20,30,40,50])
    y = np.array([2,4,6,8,10])
    
    x = tr.tensor(x, "x", requires_grad=True)
    y = tr.tensor(y, "y", requires_grad=True)
    
    z = x - y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_mul_scalar_and_tensor():
    x = tr.tensor(20, "x", requires_grad=True)
    
    z = 10 * x
    
    z.backward()
    
    logger.info("z = %s x.grad = %s", z, x.grad)
    
    
def test_mul_two_tensors():
    x = tr.tensor(20, "x", requires_grad=True)
    y = tr.tensor(5, "y", requires_grad=True)
    
    z = x * y
    
    z.backward()
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_mul_two_ndim_tensors():
    x = tr.tensor([2,4,6,8,10], "x", requires_grad=True)
    y = tr.tensor([5,10,15,20,25], "y", requires_grad=True)
    
    z = x * y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_mul_and_add():
    x = tr.tensor(2, "x", requires_grad=True)
    y = tr.tensor(3.1, "y", requires_grad=True)
    z = tr.tensor(1.7, "z", requires_grad=True)
    
    a = x * (y + z)
    
    a.backward()
    
    logger.info("a = %s z.grad = %s x.grad = %s and y.grad = %s", a, z.grad, x.grad, y.grad)
    
    
def test_mul_and_add2():
    x = tr.tensor(2, "x", requires_grad=True)
    y = tr.tensor(3.1, "y", requires_grad=True)
    z = tr.tensor(1.7, "z", requires_grad=True)
    
    a = x + (y * z)
    
    a.backward()
    
    logger.info("a = %s z.grad = %s x.grad = %s and y.grad = %s", a, z.grad, x.grad, y.grad)
    
    
def test_matmul1():
    x = tr.tensor([10., 10.], "x", requires_grad=True)
    y = tr.tensor([20., 20.], "y", requires_grad=True)
    
    a = x * y
    
    a.backward(gradient=np.ones_like(x))
    
    logger.info("a = %s x.grad = %s and y.grad = %s", a, x.grad, y.grad)
    

def test_matmul2():
    x = tr.tensor([10., 10.], "x", requires_grad=True)
    y = tr.tensor([20., 20.], "y", requires_grad=True)
    
    z = x * y    
    a = 2 * z
    
    a.backward(gradient=np.ones_like(x))
    
    logger.info("a = %s z.grad = %s x.grad = %s and y.grad = %s", a, z.grad, x.grad, y.grad)


def test_matmul3():
    x = tr.tensor([10., 10.], "x", requires_grad=True)
    y = tr.tensor([[20., 20.], [2.5, 2.5]], "y", requires_grad=True)
    
    z = x @ y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_matmul4():
    x = tr.tensor([[10., 10.], [1.5, 1.5]], "x", requires_grad=True)
    y = tr.tensor([[20., 20.], [2.5, 2.5]], "y", requires_grad=True)
    
    z = x @ y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)
    
def test_matmul5():
    x = tr.tensor([[10., 10.], [1.5, 1.5], [2.5, 2.5]], "x", requires_grad=True)
    y = tr.tensor([[20., 20., 15.], [2.5, 2.5, 15.]], "y", requires_grad=True)
    
    z = x @ y
    
    z.backward(gradient=np.ones((3,3)))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)