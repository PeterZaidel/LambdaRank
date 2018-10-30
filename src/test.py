import numpy as np
import lambda_obj_py
from lambda_obj_py import lambda_objective

Y = np.array([0, 1, 2])
F = np.array([0, 1, 2])
group = np.array([2, 1])
grad, hess = lambda_objective(Y, F, 1.0, group)

print("GRAD: ", grad)
print("HESS: ", hess)
