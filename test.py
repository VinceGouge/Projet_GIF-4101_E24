from re import X
from numpy import linalg as LA
import numpy
from scipy.spatial.distance import cdist

matrice_fixed_parameter = numpy.array([[0,1,2,3,4,5,6],
[1,2,3,4,5,6,0],
[2,3,4,5,6,0,1],
[3,4,5,6,0,1,2],
[4,5,6,0,1,2,3],
[5,6,0,1,2,3,4],
[6,0,1,2,3,4,5]])

for i in matrice_fixed_parameter:
    print(i)
print(matrice_fixed_parameter)

print(matrice_fixed_parameter[1,:])