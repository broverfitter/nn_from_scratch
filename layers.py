import numpy as np

class Layer:
  def __init__(self, prev, nodes):
    self.prev = prev
    self.nodes = nodes
    self.activation = np.zeros(nodes)
  def __call__(self, x):
    if self.prev:
      return self.forward(self.prev(x))
    else:
      return self.forward(x)

class Input(Layer):
  def __init__(self, shape):
    super().__init__(None, shape)
  def forward(self, x):
    self.activation = x
    return self.activation
  def backward(self, error, lr):
    pass

class Dense(Layer):
  def __init__(self, prev, nodes):
    super().__init__(prev, nodes)
    self.weights = np.random.randn(self.prev.nodes, self.nodes) * np.sqrt(2. / self.prev.nodes)
    self.biases = np.random.normal(0, 1e-2, nodes)
  def forward(self, x):
    self.activation = np.dot(x, self.weights) + self.biases
    return self.activation
  def backward(self, error, lr):
    dw = np.dot(self.prev.activation.T, error)
    dw = np.clip(dw, -1, 1)
    self.weights += dw * lr
    self.biases += np.sum(error, 0) * lr
    self.prev.backward(error.dot(self.weights.T) * self.prev.activation, lr)