import pytest
import logging
import numpy as np

import torch as t
import mini_torch.tensor as mt


logger = logging.getLogger(__file__)

def test_add_scalar_and_tensor():
    x = np.array(20.)
    
    x = mt.tensor(x, requires_grad=True)
    y = t.tensor(20., requires_grad=True)
        
    z = 10 + x
    j = 10 + y
    
    z.backward()
    j.backward()
    
    logger.info("z = %s x.grad = %s", z, x.grad)
    logger.info("z = %s x.grad = %s", j, y.grad)
    
    assert j.detach().numpy() == z
    assert y.grad.detach().numpy() == x.grad
    
    
def test_add_scalar_and_ndim_tensor():
    x = np.array([1.,2.,3.])
    
    x = mt.tensor(x, requires_grad=True)
    y = t.tensor([1.,2.,3.], requires_grad=True)
        
    z = 10 + x
    a = 10 + y
    
    z.backward(gradient=np.ones_like(x))
    a.backward(gradient=t.ones(3))
    
    logger.info("z = %s x.grad = %s", z, x.grad)
    
    assert a.detach().numpy().all() == z.all()
    assert all(y.grad.detach().numpy()) == all(x.grad)


def test_add_two_tensor():
    x = np.array(20)
    y = np.array(30)
    
    x = mt.tensor(x, requires_grad=True)
    y = mt.tensor(y, requires_grad=True)
    
    z = x + y
    
    z.backward()
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_add_ndim_tensor_and_tensor():
    x = np.array([1,2,3,4,5])
    y = np.array(5)
    
    x = mt.tensor(x, requires_grad=True)
    y = mt.tensor(y, requires_grad=True)
    
    z = x + y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_add_ndim_tensor_and_ndim_tensor():
    x = np.array([1,2,3,4,5])
    y = np.array([2,4,6,8,10])
    
    x = mt.tensor(x, requires_grad=True)
    y = mt.tensor(y, requires_grad=True)
    
    z = x + y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_sub_scalar_and_tensor():
    x = np.array(5)
    
    x = mt.tensor(x, requires_grad=True)
        
    z = 10 - x
    
    z.backward()
    
    logger.info("z = %s x.grad = %s", z, x.grad)


def test_sub_scalar_and_ndim_tensor():
    x = np.array([1,2,3])
    
    x = mt.tensor(x, requires_grad=True)
        
    z = 10 - x
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s", z, x.grad)


def test_sub_two_tensor():
    x = np.array(40)
    y = np.array(30)
    
    x = mt.tensor(x, requires_grad=True)
    y = mt.tensor(y, requires_grad=True)
    
    z = x - y
    
    z.backward()
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_sub_ndim_tensor_and_tensor():
    x = np.array([10,20,30,40,50])
    y = np.array(5)
    
    x = mt.tensor(x, requires_grad=True)
    y = mt.tensor(y, requires_grad=True)
    
    z = x - y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_sub_ndim_tensor_and_ndim_tensor():
    x = np.array([10,20,30,40,50])
    y = np.array([2,4,6,8,10])
    
    x = mt.tensor(x, requires_grad=True)
    y = mt.tensor(y, requires_grad=True)
    
    z = x - y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_mul_scalar_and_tensor():
    x = mt.tensor(20, requires_grad=True)
    
    z = 10 * x
    
    z.backward()
    
    logger.info("z = %s x.grad = %s", z, x.grad)


def test_mul_two_tensors():
    x = mt.tensor(20, requires_grad=True)
    y = mt.tensor(5, requires_grad=True)
    
    z = x * y
    
    z.backward()
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_mul_two_ndim_tensors():
    x = mt.tensor([2,4,6,8,10], requires_grad=True)
    y = mt.tensor([5,10,15,20,25], requires_grad=True)
    
    z = x * y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_mul_and_add():
    x = mt.tensor(2, requires_grad=True)
    y = mt.tensor(3.1, requires_grad=True)
    z = mt.tensor(1.7, requires_grad=True)
    
    a = x * (y + z)
    
    a.backward()
    
    logger.info("a = %s z.grad = %s x.grad = %s and y.grad = %s", a, z.grad, x.grad, y.grad)


def test_mul_and_add2():
    x = mt.tensor(2, requires_grad=True)
    y = mt.tensor(3.1, requires_grad=True)
    z = mt.tensor(1.7, requires_grad=True)
    
    a = x + (y * z)
    
    a.backward()
    
    logger.info("a = %s z.grad = %s x.grad = %s and y.grad = %s", a, z.grad, x.grad, y.grad)


def test_matmul1():
    x = mt.tensor([10., 10.], requires_grad=True)
    y = mt.tensor([20., 20.], requires_grad=True)
    
    a = x * y
    
    a.backward(gradient=np.ones_like(x))
    
    logger.info("a = %s x.grad = %s and y.grad = %s", a, x.grad, y.grad)


def test_matmul2():
    x = mt.tensor([10., 10.], requires_grad=True)
    y = mt.tensor([20., 20.], requires_grad=True)
    
    z = x * y    
    a = 2 * z
    
    a.backward(gradient=np.ones_like(x))
    
    logger.info("a = %s z.grad = %s x.grad = %s and y.grad = %s", a, z.grad, x.grad, y.grad)


def test_matmul3():
    x = mt.tensor([10., 10.], requires_grad=True)
    y = mt.tensor([[20., 20.], [2.5, 2.5]], requires_grad=True)
    
    z = x @ y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_matmul4():
    x = mt.tensor([[10., 10.], [1.5, 1.5]], requires_grad=True)
    y = mt.tensor([[20., 20.], [2.5, 2.5]], requires_grad=True)
    
    z = x @ y
    
    z.backward(gradient=np.ones_like(x))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_matmul5():
    x = mt.tensor([[10., 10.], [1.5, 1.5], [2.5, 2.5]], requires_grad=True)
    y = mt.tensor([[20., 20., 15.], [2.5, 2.5, 15.]], requires_grad=True)
    
    z = x @ y
    
    z.backward(gradient=np.ones((3,3)))
    
    logger.info("z = %s x.grad = %s and y.grad = %s", z, x.grad, y.grad)


def test_pow1():
    x = mt.tensor([[10., 10.], [1.5, 1.5]], requires_grad=True)
    y = 2
    
    x1 = t.tensor([[10., 10.], [1.5, 1.5]], requires_grad=True)
    
    z = x ** y
    z1 = x1 ** y
    
    z.backward(gradient=np.ones((2,2)))
    z1.backward(gradient=t.ones((2,2)))
    
    logger.info("z = %s, x.grad = %s", z, x.grad)
    
    assert np.array_equal(z1.detach().numpy(), z)
    assert np.array_equal(x1.grad.detach().numpy(), x.grad)


def test_div():
    x = mt.tensor([[10., 10.], [10., 10.]], requires_grad=True)
    y = 2
    
    x1 = t.tensor([[10., 10.], [10., 10.]], requires_grad=True)
    
    z = x / y
    z1 = x1 / 2
    
    z.backward(gradient=np.ones((2,2)))
    z1.backward(gradient=t.ones((2,2)))
    
    logger.info("z = %s, x.grad = %s", z, x.grad)
    
    assert z1.detach().numpy().all() == z.all()
    assert np.array_equal(x1.grad.detach().numpy(), x.grad)


def test_transpose():
    x = mt.tensor([[20., 20., 15.], [2.5, 2.5, 15.]], requires_grad=True)
    y = mt.tensor([[10., 10., 10.], [1.5, 1.5, 1.5]], requires_grad=True)
    
    x1 = t.tensor([[20., 20., 15.], [2.5, 2.5, 15.]], requires_grad=True)
    y1 = t.tensor([[10., 10., 10.], [1.5, 1.5, 1.5]], requires_grad=True)
    
    z = x @ y.T
    z1 = x1 @ y1.T
    
    z.backward(gradient=np.ones((2,2)))
    z1.backward(gradient=t.ones((2,2)))
    
    logger.info("z = %s, x.grad = %s y.grad = %s", z, x.grad, y.grad)
    
    assert z1.detach().numpy().all() == z.all()
    assert np.array_equal(x1.grad.detach().numpy(), x.grad)
    assert np.array_equal(y1.grad.detach().numpy(), y.grad)


def test_log():
    x = mt.tensor([[10., 10.], [10., 10.]], requires_grad=True)
    x1 = t.tensor([[10., 10.], [10., 10.]], requires_grad=True)
    
    z = 1 * x.log()
    z.backward(gradient=np.ones((2,2)))
    
    z1 = 1 * x1.log()
    z1.backward(gradient=t.ones((2,2)))
    
    logger.info("\nz = %s, xlog.grad = %s", z, x.grad)
    logger.info("\nz1 = %s, x1log.grad = %s", z1, x1.grad)
    
    assert z1.detach().numpy().all() == z.all()
    assert np.array_equal(np.round(np.subtract(x1.grad.detach().numpy(), x.grad)), np.zeros((2,2)))


def test_sigmoid():
    x = mt.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    x1 = t.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    
    z = 1 * x.sigmoid()
    z.backward(gradient=np.ones((1,5)))
    
    z1 = 1 * x1.sigmoid()
    z1.backward(gradient=t.ones(1,5))
    
    logger.info("\nz = %s, xSigmoid.grad = %s", z, x.grad)
    logger.info("\nz1 = %s, x1Sigmoid.grad = %s", z1, x1.grad)


def test_sum():
    x = mt.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    x1 = t.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    
    z = 1 * x.sum()
    z.backward(gradient=np.ones((5,)))
    
    z1 = 1 * x1.sum()
    z1.backward()
    
    logger.info("\nz = %s, xSum.grad = %s", z, x.grad)
    logger.info("\nz1 = %s, x1Sum.grad = %s", z1, x1.grad)

def test_sum1():
    x = mt.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    x1 = t.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    
    z = 1 * x.sum(axis=1)
    z.backward(gradient=np.ones((5,)))
    
    z1 = 1 * x1.sum(axis=1)
    z1.backward()
    
    logger.info("\nz = %s, xSum.grad = %s", z, x.grad)
    logger.info("\nz1 = %s, x1Sum.grad = %s", z1, x1.grad)
    
def test_sum2():
    x = mt.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    x1 = t.tensor([[1.2, 1.3, 1.4, 1.5, 1.6]], requires_grad=True)
    
    z = 1 * x.sum(axis=1, keepdim=True)
    z.backward(gradient=np.ones((5,)))
    
    z1 = 1 * x1.sum(axis=1, keepdim=True)
    z1.backward()
    
    logger.info("\nz = %s, xSum.grad = %s", z, x.grad)
    logger.info("\nz1 = %s, x1Sum.grad = %s", z1, x1.grad)
