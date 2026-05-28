import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
import os
import zipfile
import urllib.request

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Functii ajutatoare

def vector_norm(x):
    """Calculeaza norma euclidiana a unui vector"""
    return np.sqrt(np.sum(x**2))

def frobenius_norm(A):
    """Calculeaza norma Frobenius a unei matrici"""
    return np.sqrt(np.sum(A**2))

# 1. Factorizarea QR folosind Reflexii Householder

def qr_householder(A):
    """
    Calculeaza factorizarea QR a unei matrici A folosind reflexii Householder.
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

# 2. SVD folosind Iterația QR

def custom_svd(A, k=None, max_iter=2000, tol=1e-12):
    """
    Calculeaza Descompunerea Valorilor Singulare (SVD) a matricii A
    folosind algoritmul de Iteratie QR pe A^T * A.
    Returneaza U, Sigma, VT restrictionate la primele k componente.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    if k is None:
        k = min(m, n)
        
    T = A.T @ A
    V = np.eye(n)
    
    # Iteratia QR pentru matrici simetrice
    for _ in range(max_iter):
        Q_k, R_k = qr_householder(T)
        
        T_next = R_k @ Q_k
        
        # FIX 1: Forțăm simetria pentru a preveni erorile de rotunjire (floating point drift)
        T_next = (T_next + T_next.T) / 2.0
        
        V = V @ Q_k
        
        # FIX 2: Verificăm convergența pe partea strict inferior triunghiulară
        lower_tri = np.tril(T_next, -1)
        if frobenius_norm(lower_tri) < tol:
            T = T_next
            break
            
        T = T_next
        
    eigenvalues = np.diag(T)
    # Gestionăm valorile negative minuscule cauzate de precizia în virgulă mobilă
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    
    # Sortăm valorile singulare și vectorii proprii corespunzători descrescător
    sorted_indices = np.argsort(singular_values)[::-1]
    singular_values = singular_values[sorted_indices]
    V = V[:, sorted_indices]
    
    # Păstrăm doar primele k componente pentru reducerea dimensionalității
    singular_values = singular_values[:k]
    V = V[:, :k]
    
    # Calculăm matricea U (u_i = A * v_i / sigma_i)
    U = np.zeros((m, k))
    for i in range(k):
        if singular_values[i] > 1e-12:
            U[:, i] = (A @ V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(m)
            
    Sigma = np.diag(singular_values)
    VT = V.T
    
    return U, Sigma, VT

# 3. Pregătirea Datelor și Predicția pentru Sistemul de Recomandare

def descarca_movielens_100k(folder="ml-100k"):
    """
    Descarcă baza de date MovieLens 100k de pe internet dacă nu există local.
    """
    if os.path.exists(folder):
        return folder

    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "ml-100k.zip"

    try:
        print("Se descarcă baza de date reală MovieLens 100k...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(".")
        return folder
    except Exception as exc:
        print("Eroare la descărcare:", exc)
        return None

def incarca_date_reale(max_users=100, max_movies=130):
    """
    Încarcă un subset din MovieLens pentru a păstra un timp de execuție rezonabil.
    """
    folder = descarca_movielens_100k()
    
    ratings_path = os.path.join(folder, "u.data")
    items_path = os.path.join(folder, "u.item")

    # Încărcăm notele (ratings)
    ratings = pd.read_csv(ratings_path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

    # Încărcăm titlurile
    items = pd.read_csv(items_path, sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])

    # Filtrăm primii n utilizatori și primele n filme
    ratings = ratings[(ratings["user_id"] <= max_users) & (ratings["movie_id"] <= max_movies)]

    # Creăm matricea utilizator-film
    R_df = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
    
    # Creăm dicționarul de titluri cu indexare de la 0 pentru codul nostru
    movie_ids = list(R_df.columns)
    title_map_raw = dict(zip(items["movie_id"], items["title"]))
    
    titles = {j: title_map_raw.get(movie_id, f"Film {movie_id}") for j, movie_id in enumerate(movie_ids)}
    R = R_df.to_numpy(dtype=float)
    
    return R, titles

def preprocess_ratings(R):
    """Umple valorile lipsă cu media utilizatorului și centrează datele."""
    R = np.array(R, dtype=float)
    mask_observed = ~np.isnan(R)
    
    user_means = np.nanmean(R, axis=1)
    
    R_filled = R.copy()
    for i in range(R.shape[0]):
        R_filled[i, np.isnan(R_filled[i])] = user_means[i]
        
    A = R_filled - user_means[:, None]
    
    return A, user_means, mask_observed

def predict_ratings(R, k):
    """Prezice notele lipsă folosind funcția custom SVD."""
    A, user_means, mask_observed = preprocess_ratings(R)
    
    U, Sigma, VT = custom_svd(A, k=k)
    
    A_hat = U @ Sigma @ VT
    
    R_hat = A_hat + user_means[:, None]
    R_hat = np.clip(R_hat, 1.0, 5.0)
    
    return R_hat, mask_observed

def recommend_movies(R, titles, user_id, top_n=2, k=3):
    """Recomandă cele mai bune filme nevăzute pentru un utilizator."""
    R_hat, mask_observed = predict_ratings(R, k)
    
    user_scores = R_hat[user_id].copy()
    user_scores[mask_observed[user_id]] = -np.inf
    
    top_indices = np.argsort(user_scores)[::-1][:top_n]
    
    recommendations = []
    for idx in top_indices:
        if np.isfinite(user_scores[idx]):
            recommendations.append((titles.get(idx, f"Film {idx}"), float(user_scores[idx])))
            
    return recommendations

# 4. Evaluare și Generare Grafice

def calculate_rmse(R_true, R_pred, mask):
    """Calculează RMSE doar pentru notele observate."""
    error = R_true[mask] - R_pred[mask]
    return np.sqrt(np.mean(error**2))

def plot_singular_values(A):
    """Generează graficul de descreștere a valorilor singulare."""
    _, Sigma, _ = custom_svd(A)
    sv = np.diag(Sigma)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(sv) + 1), sv, marker='o', linestyle='-', color='b')
    plt.title("Scăderea Valorilor Singulare")
    plt.xlabel("Indexul Valorii Singulare")
    plt.ylabel("Magnitudine")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("singular_values.png")
    print("Grafic salvat: singular_values.png")

def plot_rmse_vs_k(R, max_k):
    """Generează graficul RMSE pentru diferite valori ale lui k (Optimizat)."""
    rmses = []
    k_values = range(1, max_k + 1)
    
    A, user_means, mask_observed = preprocess_ratings(R)
    
    print("Calculăm SVD o singură dată pentru grafic...")
    # Calculăm SVD o singură dată la dimensiunea maximă necesară
    U, Sigma, VT = custom_svd(A, k=max_k)
    
    for k in k_values:
        # Reconstruim folosind doar primele k componente deja calculate
        U_k = U[:, :k]
        Sigma_k = Sigma[:k, :k]
        VT_k = VT[:k, :]
        
        A_hat = U_k @ Sigma_k @ VT_k
        
        # Readucem la scala notelor de la 1 la 5
        R_hat = A_hat + user_means[:, None]
        R_hat = np.clip(R_hat, 1.0, 5.0)
        
        # Calculăm eroarea
        rmse = calculate_rmse(R, R_hat, mask_observed)
        rmses.append(rmse)
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, rmses, marker='s', color='r')
    plt.title("Eroarea de Reconstrucție (RMSE) vs Numărul de Trăsături Latente (k)")
    plt.xlabel("Rang (k)")
    plt.ylabel("RMSE")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rmse_vs_k.png")
    print("Grafic salvat: rmse_vs_k.png")

# 5. Execuția Principală

if __name__ == "__main__":

    # 1. Încărcarea setului de date
    print("Se încarcă Baza de Date MovieLens...")
    R, titles = incarca_date_reale(max_users=100, max_movies=130)
    A, _, _ = preprocess_ratings(R)
    
    # 2. Validarea matematică comparativ cu NumPy
    print("Se validează SVD custom comparativ cu NumPy...")
    _, Sigma_custom, _ = custom_svd(A, k=min(A.shape))
    _, S_np, _ = np.linalg.svd(A, full_matrices=False)
    
    error_svd = vector_norm(np.diag(Sigma_custom) - S_np[:len(S_np)])
    print(f"Eroarea (Norma L2) față de NumPy: {error_svd:.4e}\n")
    
    # 3. Generarea Recomandărilor
    k_features = 3
    user_to_test = 1
    
    print(f"Se generează recomandări pentru Utilizatorul {user_to_test} folosind k={k_features} trăsături latente...")
    recs = recommend_movies(R, titles, user_id=user_to_test, top_n=3, k=k_features)
    
    print(f"\nTop Recomandări pentru Utilizatorul {user_to_test}:")
    for movie, score in recs:
        print(f"  -> {movie} (Notă estimată: {score:.2f} / 5.00)")
        
    # 4. Generarea Graficelor
    print("\nSe generează graficele de evaluare...")
    plot_singular_values(A)
    plot_rmse_vs_k(R, max_k=15)
    
    # 5. Afișarea istoricului de vizionare pentru utilizator
    print(f"\nIstoricul real de vizionare pentru Utilizatorul {user_to_test}:")
    
    # Extragem linia cu notele utilizatorului din matricea R
    note_utilizator = R[user_to_test]
    
    # Găsim indecșii filmelor care NU sunt np.nan (adică filmele văzute)
    filme_vazute_idx = np.where(~np.isnan(note_utilizator))[0]
    
    # Creăm o listă cu (Titlu, Notă) și o sortăm descrescător după notă
    istoric = [(titles[idx], note_utilizator[idx]) for idx in filme_vazute_idx]
    istoric.sort(key=lambda x: x[1], reverse=True)
    
    for film, nota in istoric:
        print(f"  -> {film} (Notă acordată: {nota})")
        
    print("\nProces finalizat cu succes!")