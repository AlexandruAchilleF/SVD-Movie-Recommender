import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper Functions
def vector_norm(x):
    """
    Calculates the Euclidean norm of a vector.
    """
    return np.sqrt(np.sum(x**2))

def frobenius_norm(A):
    """
    Calculates the Frobenius norm of a matrix.
    """
    return np.sqrt(np.sum(A**2))

# 1. QR Factorization using Householder Reflections (Course 6)
def qr_householder(A):
    """
    Computes the QR factorization of a matrix A using Householder reflections.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    R = A.copy()
    Q = np.eye(m)
    
    for k in range(n):
        x = R[k:m, k]
        
        norm_x = vector_norm(x)
        
        if norm_x < 1e-15:
            continue
            
        v = x.copy()
        
        # sgn(v[0]) * ||v||_2
        sign_v0 = 1.0 if v[0] >= 0 else -1.0
        v[0] = v[0] + sign_v0 * norm_x
        
        # Calculate dot product (v^T * v)
        v_norm_sq = np.sum(v**2)
        if v_norm_sq < 1e-15:
            continue
            
        # Hv = I_{m-k} - 2 * (v v^T) / (v^T v)
        Hv = np.eye(m - k) - 2.0 * np.outer(v, v) / v_norm_sq
        
        H = np.eye(m)
        H[k:m, k:m] = Hv
        
        # Apply reflection
        R = H @ R
        Q = Q @ H
        
    return Q, R

if __name__ == "__main__":
    A_test = np.array([[12, -51, 4],
                       [6, 167, -68],
                       [-4, 24, -41]])
    
    Q, R = qr_householder(A_test)
    
    print("QR Householder Implementation")
    print("Matrix Q:\n", np.round(Q, 4))
    print("Matrix R:\n", np.round(R, 4))
    print("\nVerificare Ortogonalitate (Q^T @ Q):")
    identitate_aprox = Q.T @ Q
    print(np.round(identitate_aprox, 4))