import ctypes, os
import numpy as np
from numpy.ctypeslib import ndpointer

#print("loading...")
# libc = ctypes.cdll["src/lambda_obj.so"]
#print("laoded")

print("VERSION: 1.1")

lib = ctypes.cdll["src/lambda_obj.so"]
LambdaRankObjective_c = lib['LambdaRankObjective']
LambdaRankObjective_c.argtypes = [ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_double,
                                ctypes.c_long,
                                ctypes.POINTER(ctypes.c_long),
                                ctypes.c_long,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),]


def lambda_objective(Y, F, sigma, group):
    #print("AAA")
    Y = Y.astype(np.float64)
    F = F.astype(np.float64)

    group = group.astype(np.int32)

    size = Y.shape[0]
    group_size = group.shape[0]

    grad = np.zeros(size, dtype=np.float64)
    hess = np.zeros(size, dtype=np.float64)


    ptr_Y = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_F = F.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_group = group.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    ptr_grad = grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ptr_hess = hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    #print("hello")
    LambdaRankObjective_c(ptr_Y, ptr_F, sigma, size, ptr_group, group_size, ptr_grad, ptr_hess)
    #print("PYTHON")
    #print("PYTHON_GRAD: ", grad)
    #print("PYTHON_HESS: ", hess)

    print("GRAD_NORM: ", np.linalg.norm(grad))
    print("HESS_NORM: ", np.linalg.norm(hess))

    return grad, hess
