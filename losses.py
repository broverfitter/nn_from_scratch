import numpy as np

class Categorical_Crossentropy():
  def __call__(self, ypred, ytrue):
    return -np.mean(ytrue * np.log(ypred))
  def backward(self, ypred, ytrue):
    error = ytrue - ypred
    return error

class MSE():
  def __call__(self, ypred, ytrue):
    return np.mean(0.5 * (ytrue - ypred)**2)
  def backward(self, ypred, ytrue):
    return (ytrue - ypred)