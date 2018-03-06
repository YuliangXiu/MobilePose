import numpy as np

h = 224
w = 224

x = np.arange(0, h)
y = np.arange(0, w)
x, y = np.meshgrid(x, y)
x = x[:,:, np.newaxis]
y = y[:,:, np.newaxis]

print(np.mean(x))
print(np.mean(y))
print(np.std(x))
print(np.std(y))
