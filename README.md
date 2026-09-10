# Movie Recommendation System using SVD

This project is an implementation of a recommendation system (Collaborative Filtering) based on the **MovieLens 100k** dataset. The main feature of this project is the **from-scratch mathematical implementation of Singular Value Decomposition (SVD)**, without using the predefined SVD functions from numerical computation libraries.

To obtain the SVD factorization ($A = U \Sigma V^T$), the algorithm uses **QR Factorization via Householder reflections** followed by the **QR Algorithm (Iteration)**.

## Key Features

* **Matrix Factorization:** Custom implementation of the SVD algorithm to extract latent features of users and movies.
* **Automatic Data Download:** The script automatically downloads the MovieLens 100k dataset if it is not found locally.
* **Processing and Prediction:** Handles missing values (unseen movies) using the mean rating of each user and predicts ratings using a low-rank approximation ($k$).
* **Recommendation System:** Analyzes a user's viewing history and recommends the most relevant movies they haven't seen yet.
* **Mathematical Validation:** Compares the custom SVD results with the highly optimized `numpy.linalg.svd` function to demonstrate accuracy (error in the order of $10^{-14}$).
* **Data Visualization:** Generates plots to analyze the model's performance.

## System Requirements

The project runs on Python 3 and requires the following installed libraries:

```bash
pip install numpy pandas matplotlib
```

*(The `ssl`, `os`, `zipfile`, and `urllib.request` libraries are part of the standard Python library and do not require separate installation).*

## How to Run

1. Save the source code into a file, for example, `svd_recommender.py`.
2. Open the terminal in that directory and run:

```bash
python svd_recommender.py
```

On the first run, the script will download the `ml-100k.zip` archive from the GroupLens servers and extract the necessary data. To maintain a reasonable execution time for the custom SVD algorithm, the script automatically extracts a subset of the database (e.g., the first 100 users and 130 movies).

## What Does the Script Output?

Once executed, the script will provide the following results:

**1. Console Output:**
* Data download and loading status.
* The error ($L_2$ Norm) between manually calculated singular values and those calculated by NumPy.
* Top recommendations (movies and estimated ratings) for a test user.
* The actual viewing history of that user to provide context for the recommendations.

**2. Locally Saved Plots:**
* `singular_values.png` - Shows the importance of each latent feature (the exponential decay of singular values).
* `rmse_vs_k.png` - Analyzes how the prediction error (RMSE) decreases as we increase the number of latent features ($k$) retained in the matrix reconstruction.

## Mathematical Structure

* `qr_householder(A)`: Applies orthogonal transformations (Householder reflections) to decompose matrix $A$ into an orthogonal matrix $Q$ and an upper triangular matrix $R$.
* `custom_svd(A)`: Calculates the eigenvalues and eigenvectors for $A^T A$ by applying QR iterations based on the previous function, forcing symmetry at each step to avoid floating-point precision errors. It then extracts matrices $U$, $\Sigma$, and $V^T$.
