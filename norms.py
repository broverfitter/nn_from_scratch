import numpy as np

class BatchNormalisation(Layer):
    def __init__(self, prev):
        nodes = prev.nodes
        super().__init__(prev, nodes)
        self.epsilon = 1e-7

    def forward(self, x):
        self.mu = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.xmu = x - self.mu
        self.inv_std = 1. / np.sqrt(self.var + self.epsilon)
        self.activation = self.xmu * self.inv_std
        return self.activation

    def backward(self, error, lr):
        N, D = error.shape

        dxhat = error * self.inv_std
        dvar = np.sum(error * self.xmu, axis=0) * -0.5 * (self.inv_std**3)
        dmu = np.sum(error * -self.inv_std, axis=0) + dvar * np.mean(-2. * self.xmu, axis=0)

        dx = dxhat + (2.0/N) * dvar * self.xmu + dmu/N
        self.prev.backward(dx, lr)