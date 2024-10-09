import numpy as np
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv('sample_data/mnist_train_small.csv', header=None).to_numpy()

X = df[:,1:] / np.max(df)
Y = to_categorical(df[:,0])

loss = Categorical_Crossentropy()

x = Input(784)
x = Dense(x, 16)
x = Relu(x)
x = Dense(x, 16)
x = Relu(x)
x = Dense(x, 10)
x = Softmax(x)

losses = []

for step in range(100):
  ypred = x(X)
  losses += [loss(ypred, Y)]
  if step % 10 == 0:
    accuracy = sum(np.argmax(Y, axis=1) == np.argmax(x(X), axis=1)) / Y.shape[0]
    print(step, losses[-1], accuracy)
  else:
    print(step, losses[-1])
  x.backward(loss.backward(ypred, Y), 7e-5)