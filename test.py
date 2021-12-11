from re import X
from numpy import linalg as LA
import numpy
from scipy.spatial.distance import cdist

a = numpy.arange(12)-4

b = a.reshape((4,3))
print(b)
print(LA.norm(b,ord=2,axis=1))


print(x)
print(min_value)
print(min_value.min())
