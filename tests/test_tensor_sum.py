import pytest

from autograd.tensor import Tensor


def test_tensor_sum():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = t1.sum()

    t2.backward()

    assert t1.grad.data.tolist() == [1, 1, 1]


def test_sum_with_grad():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = t1.sum()

    t2.backward(Tensor(3))

    assert t1.grad.data.tolist() == [3, 3, 3]
