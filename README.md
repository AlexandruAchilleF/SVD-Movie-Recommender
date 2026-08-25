# Sistem de Recomandare Filme cu SVD (De la zero)

Acest proiect este o implementare a unui sistem de recomandare (Collaborative Filtering) bazat pe dataset-ul **MovieLens 100k**. Caracteristica principală a acestui proiect este **implementarea matematică de la zero a Descompunerii Valorilor Singulare (SVD)**, fără a folosi funcțiile predefinite de SVD din bibliotecile de calcul numeric.

Pentru a obține factorizarea SVD ($A = U \Sigma V^T$), algoritmul folosește **Factorizarea QR prin reflexii Householder** urmată de **Iterația QR**.

## Funcționalități Principale

* **Factorizare Matriceală:** Implementare proprie a algoritmului SVD pentru a extrage trăsăturile latente ale utilizatorilor și filmelor.
* **Descărcare Automată a Datelor:** Scriptul descarcă automat setul de date MovieLens 100k dacă nu este găsit local.
* **Procesare și Predicție:** Tratează valorile lipsă (filmele nevăzute) folosind media notelor fiecărui utilizator și prezice ratingurile folosind o aproximare de rang redus ($k$).
* **Sistem de Recomandare:** Analizează istoricul unui utilizator și îi recomandă cele mai relevante filme pe care nu le-a vizionat încă.
* **Validare Matematică:** Compară rezultatele SVD-ului custom cu funcția super-optimizată `numpy.linalg.svd` pentru a demonstra acuratețea (eroare de ordinul $10^{-14}$).
* **Vizualizare Date:** Generează grafice pentru analiza performanței modelului.

## Cerințe de Sistem

Proiectul rulează în Python 3 și necesită următoarele biblioteci instalate:

```bash
pip install numpy pandas matplotlib
```

*(Bibliotecile `ssl`, `os`, `zipfile` și `urllib.request` fac parte din biblioteca standard Python și nu necesită instalare separată).*

## Cum se rulează

1. Salvează codul sursă într-un fișier, de exemplu `recomandare_svd.py`.
2. Deschide terminalul în directorul respectiv și rulează:

```bash
python recomandare_svd.py
```

La prima rulare, scriptul va descărca arhiva `ml-100k.zip` de pe serverele GroupLens și va extrage datele necesare. Pentru a păstra un timp de execuție rezonabil pentru algoritmul SVD custom, scriptul extrage automat un subset din baza de date (ex: primii 100 de utilizatori și 130 de filme).

## Ce generează scriptul?

Odată rulat, scriptul va oferi următoarele rezultate:

**1. Output în consolă:**
* Statusul descărcării și încărcării datelor.
* Eroarea (Norma $L_2$) dintre valorile singulare calculate manual și cele calculate de NumPy.
* Top recomandări (filme și note estimate) pentru un utilizator de test.
* Istoricul real de vizionare al acelui utilizator pentru a oferi context asupra recomandărilor.

**2. Grafice salvate local:**
* `singular_values.png` - Arată importanța fiecărei trăsături latente (descreșterea exponențială a valorilor singulare).
* `rmse_vs_k.png` - Analizează modul în care eroarea de predicție (RMSE) scade pe măsură ce mărim numărul de trăsături latente ($k$) reținute în reconstrucția matricii.

## Structura Matematică

* `qr_householder(A)`: Aplică transformări ortogonale (reflexii Householder) pentru a descompune matricea $A$ într-o matrice ortogonală $Q$ și una superior triunghiulară $R$.
* `custom_svd(A)`: Calculează valorile și vectorii proprii pentru $A^T A$ aplicând iterații QR pe baza funcției anterioare, forțând simetria la fiecare pas pentru a evita erorile de precizie în virgulă mobilă. Apoi extrage matricile $U$, $\Sigma$ și $V^T$.
