import sqlite3
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import csv


def cos_distance_bytes(v1, v2):
    u = np.frombuffer(v1, np.float32)
    w = np.frombuffer(v2, np.float32)

    un = np.linalg.norm(u)
    wn = np.linalg.norm(w)
    if un == 0 or wn == 0:
        return np.nan

    return 1.0 - (np.dot(u, w) / (un * wn))


def stats_for_array(x):
    """Compute summary statistics for a 1D numpy array."""
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
    }


def main(TARGET=1, LSA_K=800):

    conn = sqlite3.connect("lyrics.db")
    curs = conn.cursor()

    # Load target genre + LSA
    curs.execute("SELECT genres FROM artists WHERE id=?", (TARGET,))
    row = curs.fetchone()
    if row is None:
        conn.close()
        return None
    target_genre_vec = row[0]

    curs.execute(f"SELECT vec FROM artist_lsa_{LSA_K} WHERE artist_id=?", (TARGET,))
    row = curs.fetchone()
    if row is None:
        conn.close()
        return None
    target_lsa_vec = row[0]

    # Load all others
    curs.execute("SELECT id, genres FROM artists WHERE id != ?", (TARGET,))
    genre_map = {row[0]: row[1] for row in curs.fetchall()}

    curs.execute(
        f"SELECT artist_id, vec FROM artist_lsa_{LSA_K} WHERE artist_id != ?",
        (TARGET,),
    )
    lsa_map = {row[0]: row[1] for row in curs.fetchall()}

    conn.close()

    # Build distance lists
    gdist_list = []
    ldist_list = []

    for nid in lsa_map:
        if nid not in genre_map:
            continue

        gdist = cos_distance_bytes(target_genre_vec, genre_map[nid])
        ldist = cos_distance_bytes(target_lsa_vec, lsa_map[nid])

        if np.isnan(gdist) or np.isnan(ldist):
            continue

        gdist_list.append(gdist)
        ldist_list.append(ldist)

    gdist_list = np.array(gdist_list, dtype=np.float64)
    ldist_list = np.array(ldist_list, dtype=np.float64)

    n = len(gdist_list)

    if n < 3:
        return {
            "target_id": TARGET,
            "lsa_k": LSA_K,
            "n_pairs": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "genre_mean": np.nan,
            "genre_std": np.nan,
            "genre_min": np.nan,
            "genre_max": np.nan,
            "genre_range": np.nan,
            "lsa_mean": np.nan,
            "lsa_std": np.nan,
            "lsa_min": np.nan,
            "lsa_max": np.nan,
            "lsa_range": np.nan,
        }

    # correlation tests
    pearson_r, pearson_p = pearsonr(gdist_list, ldist_list)
    spearman_r, spearman_p = spearmanr(gdist_list, ldist_list)

    # stats
    gstats = stats_for_array(gdist_list)
    lstats = stats_for_array(ldist_list)
    diff_stats = stats_for_array(abs(gdist_list-ldist_list))

    return {
        "target_id": TARGET,
        "lsa_k": LSA_K,
        "n_pairs": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "genre_mean": gstats["mean"],
        "genre_std": gstats["std"],
        "genre_min": gstats["min"],
        "genre_max": gstats["max"],
        "genre_range": gstats["range"],
        "lsa_mean": lstats["mean"],
        "lsa_std": lstats["std"],
        "lsa_min": lstats["min"],
        "lsa_max": lstats["max"],
        "lsa_range": lstats["range"],
        "lsa_mean": lstats["mean"],
        "diff_std": lstats["std"],
        "diff_min": lstats["min"],
        "diff_max": lstats["max"],
        "diff_range": lstats["range"]

    }


if __name__ == '__main__':
    results = []

    for i in tqdm(range(1, 10), desc="Targets"):
        for k in (10, 50, 100, 300, 500, 800):
            stats = main(i, k)
            if stats:
                results.append(stats)

    # Write CSV
    fieldnames = list(results[0].keys())

    with open("genre_lsa_correlations_full.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Wrote", len(results), "rows to genre_lsa_correlations_full.csv")
