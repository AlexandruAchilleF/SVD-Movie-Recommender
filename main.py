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

# 2. SVD using QR Iteration
def custom_svd(A, k=None, max_iter=200, tol=1e-12):
    """
    Computes the Singular Value Decomposition (SVD) of matrix A
    using the QR Iteration algorithm on A^T * A.
    Returns U, Sigma, VT restricted to the top k components.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    if k is None:
        k = min(m, n)
        
    # Initialize T = A^T * A and V = I
    T = A.T @ A
    V = np.eye(n)
    
    # QR Iteration for symmetric matrices
    for _ in range(max_iter):
        Q_k, R_k = qr_householder(T)
        
        # T_{k+1} = R_k * Q_k
        T_next = R_k @ Q_k
        
        # Accumulate eigenvectors in V
        V = V @ Q_k
        
        # Check convergence
        diag_diff = frobenius_norm(np.diag(T_next) - np.diag(T))
        if diag_diff < tol:
            T = T_next
            break
        T = T_next
        
    # Singular values are sqrt of eigenvalues of T
    eigenvalues = np.diag(T)
    
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    
    # Sort singular values and corresponding eigenvectors descending
    sorted_indices = np.argsort(singular_values)[::-1]
    singular_values = singular_values[sorted_indices]
    V = V[:, sorted_indices]
    
    # Keep only the top k components for dimensionality reduction
    singular_values = singular_values[:k]
    V = V[:, :k]
    
    # Calculate U matrix (u_i = A * v_i / sigma_i)
    U = np.zeros((m, k))
    for i in range(k):
        if singular_values[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(m)
            
    # Form the diagonal matrix Sigma and transpose V
    Sigma = np.diag(singular_values)
    VT = V.T
    
    return U, Sigma, VT

if __name__ == "__main__":
    A_test = np.array([[12, -51, 4],
                       [6, 167, -68],
                       [-4, 24, -41],
                       [1, 2, 3]])
    
    print("Computing Custom SVD")
    U_custom, Sigma_custom, VT_custom = custom_svd(A_test, k=3)

    U_np, S_np, VT_np = np.linalg.svd(A_test, full_matrices=False)
    
    print("Custom Singular Values:\n", np.round(np.diag(Sigma_custom), 4))
    print("NumPy Singular Values:\n", np.round(S_np[:3], 4))
    
    error_svd = vector_norm(np.diag(Sigma_custom) - S_np[:3])
    print(f"\nDifference (Error) between custom and numpy values: {error_svd:.4e}")