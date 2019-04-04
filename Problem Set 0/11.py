import numpy as np
from numpy import linalg as LA

A = np.array([[1, 0], [1, 3]])
w, v = LA.eig(A)
result = v[np.argmax(w)]

print(result)