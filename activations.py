import numpy as np

class Relu(Layer):
  def __init__(self, prev):
    nodes = prev.nodes
    super().__init__(prev, nodes)
    self.f = lambda x: np.maximum(x, 0)
  def forward(self, x):
    self.activation = self.f(x)
    return self.activation
  def backward(self, error, lr):
    error = error * (self.activation > 0)
    self.prev.backward(error, lr)

class Sigmoid(Layer):
  def __init__(self, prev):
    nodes = prev.nodes
    super().__init__(prev, nodes)
    self.f = lambda x: 1/(1+np.exp(-x))
  def forward(self, x):
    self.activation = self.f(x)
    return self.activation
  def backward(self, error, lr):
    self.prev.backward(self.activation * (1 - self.activation) * error, lr)

class Softmax(Layer):
  def __init__(self, prev):
    nodes = prev.nodes
    super().__init__(prev, nodes)
    self.f = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
  def forward(self, x):
    self.activation = self.f(x - x.max())
    return self.activation
  def backward(self, error, lr):
    return self.prev.backward(error, lr)