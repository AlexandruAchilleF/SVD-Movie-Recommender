import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper Functions

def vector_norm(x):
    """Calculates the Euclidean norm of a vector"""
    return np.sqrt(np.sum(x**2))

def frobenius_norm(A):
    """Calculates the Frobenius norm of a matrix"""
    return np.sqrt(np.sum(A**2))

# 1. QR Factorization using Householder Reflections

def qr_householder(A):
    """
    Computes the QR factorization of a matrix A using Householder reflections
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    R = A.copy()
    Q = np.eye(m)
    
    for k in range(min(m, n)):
        x = R[k:m, k]

        if len(x) <= 1:
            continue
            
        norm_x = vector_norm(x)
        
        if norm_x < 1e-15:
            continue
            
        v = x.copy()
        
        # sgn(v[0]) * ||v||_2
        if v[0] >= 0:
            sign_v0 = 1.0
        else:
            sign_v0 = -1.0

        v[0] = v[0] + sign_v0 * norm_x
        
        v_norm_sq = np.sum(v**2)
        if v_norm_sq < 1e-15:
            continue
            
        # Hv = I_{m-k} - 2 * (v v^T) / (v^T v)
        Hv = np.eye(len(x)) - 2.0 * np.outer(v, v) / v_norm_sq
        
        H = np.eye(m)
        H[k:m, k:m] = Hv
        
        R = H @ R
        Q = Q @ H
        
    return Q, R

# 2. SVD using QR Iteration
def custom_svd(A, k=None, max_iter=2000, tol=1e-12):
    """
    Computes the Singular Value Decomposition (SVD) of matrix A
    using the QR Iteration algorithm on A^T * A.
    Returns U, Sigma, VT restricted to the top k components.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    if k is None:
        k = min(m, n)
        
    T = A.T @ A
    V = np.eye(n)
    
    # QR Iteration for symmetric matrices
    for _ in range(max_iter):
        Q_k, R_k = qr_householder(T)
        
        T_next = R_k @ Q_k
        
        # FIX 1: Enforce symmetry to prevent floating point drift
        T_next = (T_next + T_next.T) / 2.0
        
        V = V @ Q_k
        
        # FIX 2: Check convergence on the strictly lower triangular part
        lower_tri = np.tril(T_next, -1)
        if frobenius_norm(lower_tri) < tol:
            T = T_next
            break
            
        T = T_next
        
    eigenvalues = np.diag(T)
    # Handle tiny negative values caused by floating point precision
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
            
    Sigma = np.diag(singular_values)
    VT = V.T
    
    return U, Sigma, VT

# 3. Recommender System Data Prep & Prediction

def generate_mock_data():
    """Generates a simulated dataset of users and movie ratings."""
    R = np.array([
        [5, 4, np.nan, 1, 1, np.nan, 2, np.nan],
        [4, 5, 4, 1, np.nan, 1, 2, 1],
        [5, np.nan, 5, np.nan, 1, 1, np.nan, 1],
        [1, 1, np.nan, 5, 4, 5, 1, 4],
        [1, np.nan, 1, 4, 5, 4, 1, 5],
        [np.nan, 1, 1, 5, 4, np.nan, 2, 4],
        [4, 4, 5, np.nan, 1, 1, 5, np.nan],
        [1, 2, 1, 4, np.nan, 5, np.nan, 5],
    ], dtype=float)

    titles = {
        0: "Toy Story", 1: "Star Wars", 2: "Return of the Jedi",
        3: "Titanic", 4: "The English Patient", 5: "Sense and Sensibility",
        6: "Raiders of the Lost Ark", 7: "Sleepless in Seattle"
    }
    return R, titles

def preprocess_ratings(R):
    """Fills missing values with user mean and centers the data."""
    R = np.array(R, dtype=float)
    mask_observed = ~np.isnan(R)
    
    user_means = np.nanmean(R, axis=1)
    
    R_filled = R.copy()
    for i in range(R.shape[0]):
        R_filled[i, np.isnan(R_filled[i])] = user_means[i]
        
    A = R_filled - user_means[:, None]
    
    return A, user_means, mask_observed

def predict_ratings(R, k):
    """Predicts missing ratings using custom SVD."""
    A, user_means, mask_observed = preprocess_ratings(R)
    
    U, Sigma, VT = custom_svd(A, k=k)
    
    A_hat = U @ Sigma @ VT
    
    R_hat = A_hat + user_means[:, None]
    R_hat = np.clip(R_hat, 1.0, 5.0)
    
    return R_hat, mask_observed

def recommend_movies(R, titles, user_id, top_n=2, k=3):
    """Recommends top unseen movies for a user."""
    R_hat, mask_observed = predict_ratings(R, k)
    
    user_scores = R_hat[user_id].copy()
    user_scores[mask_observed[user_id]] = -np.inf
    
    top_indices = np.argsort(user_scores)[::-1][:top_n]
    
    recommendations = []
    for idx in top_indices:
        if np.isfinite(user_scores[idx]):
            recommendations.append((titles.get(idx, f"Movie {idx}"), float(user_scores[idx])))
            
    return recommendations

# 4. Evaluation and Plotting

def calculate_rmse(R_true, R_pred, mask):
    """Calculates RMSE only for observed ratings."""
    error = R_true[mask] - R_pred[mask]
    return np.sqrt(np.mean(error**2))

def plot_singular_values(A):
    """Plots the decay of singular values."""
    _, Sigma, _ = custom_svd(A)
    sv = np.diag(Sigma)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(sv) + 1), sv, marker='o', linestyle='-', color='b')
    plt.title("Decay of Singular Values")
    plt.xlabel("Index of Singular Value")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("singular_values.png")
    print("Saved plot: singular_values.png")

def plot_rmse_vs_k(R, max_k):
    """Plots RMSE for different values of k."""
    rmses = []
    k_values = range(1, max_k + 1)
    
    A, _, mask_observed = preprocess_ratings(R)
    
    for k in k_values:
        R_hat, _ = predict_ratings(R, k)
        rmse = calculate_rmse(R, R_hat, mask_observed)
        rmses.append(rmse)
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, rmses, marker='s', color='r')
    plt.title("Reconstruction RMSE vs Number of Latent Features (k)")
    plt.xlabel("Rank (k)")
    plt.ylabel("RMSE")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rmse_vs_k.png")
    print("Saved plot: rmse_vs_k.png")

# 5. Main Execution

if __name__ == "__main__":

    # 1. Generate Dataset
    R, titles = generate_mock_data()
    A, _, _ = preprocess_ratings(R)
    
    # 2. Validation against NumPy
    print("Validating custom SVD against NumPy...")
    _, Sigma_custom, _ = custom_svd(A, k=min(A.shape))
    _, S_np, _ = np.linalg.svd(A, full_matrices=False)
    
    error_svd = vector_norm(np.diag(Sigma_custom) - S_np[:len(S_np)])
    print(f"L2 Norm Error vs NumPy: {error_svd:.4e}\n")
    
    # 3. Generate Recommendations
    k_features = 3
    user_to_test = 0 
    
    print(f"Generating recommendations for User {user_to_test} using k={k_features} latent features...")
    recs = recommend_movies(R, titles, user_id=user_to_test, top_n=3, k=k_features)
    
    print(f"\nTop Recommendations for User {user_to_test}:")
    for movie, score in recs:
        print(f"  -> {movie} (Estimated Rating: {score:.2f} / 5.00)")
        
    # 4. Generate Plots
    print("\nGenerating evaluation plots...")
    plot_singular_values(A)
    plot_rmse_vs_k(R, max_k=min(R.shape)-1)
    
    print("\nProcess completed successfully!")