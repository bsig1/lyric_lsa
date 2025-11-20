import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


DB_PATH = "lyrics.db"


def cos_distance_bytes(v1, v2):
    u = np.frombuffer(v1, np.float32)
    w = np.frombuffer(v2, np.float32)

    un = np.linalg.norm(u)
    wn = np.linalg.norm(w)
    if un == 0 or wn == 0:
        return np.nan

    return 1.0 - (np.dot(u, w) / (un * wn))


def get_distances_for_artist(target_id, lsa_k=800, max_points=None):
    """
    Returns genre_distances, lsa_distances as 1D numpy arrays
    for a given target artist and LSA dimension k.
    Optionally subsamples to max_points for plotting.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Target vectors
    cur.execute("SELECT genres FROM artists WHERE id=?", (target_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No genres vector for artist {target_id}")
    target_genre_vec = row[0]

    cur.execute(f"SELECT vec FROM artist_lsa_{lsa_k} WHERE artist_id=?", (target_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"No LSA vec for artist {target_id} in artist_lsa_{lsa_k}")
    target_lsa_vec = row[0]

    # All other artists
    cur.execute("SELECT id, genres FROM artists WHERE id != ?", (target_id,))
    genre_map = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute(
        f"SELECT artist_id, vec FROM artist_lsa_{lsa_k} WHERE artist_id != ?",
        (target_id,),
    )
    lsa_map = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()

    gdist_list = []
    ldist_list = []

    for nid, lvec in lsa_map.items():
        gvec = genre_map.get(nid)
        if gvec is None:
            continue

        gdist = cos_distance_bytes(target_genre_vec, gvec)
        ldist = cos_distance_bytes(target_lsa_vec, lvec)

        if np.isnan(gdist) or np.isnan(ldist):
            continue

        gdist_list.append(gdist)
        ldist_list.append(ldist)

    gdist_arr = np.array(gdist_list, dtype=np.float64)
    ldist_arr = np.array(ldist_list, dtype=np.float64)

    return gdist_arr, ldist_arr


def plot_artist_scatter(target_id, lsa_k=800, max_points=5000):
    """
    Make a scatterplot of genre_distance vs lsa_distance for a target artist,
    with a least-squares regression line overlaid.
    """
    x, y = get_distances_for_artist(target_id, lsa_k=lsa_k, max_points=max_points)
    n = len(x)
    if n < 3:
        print(f"Not enough points to plot for artist {target_id}, k={lsa_k}")
        return

    # LSRL using scipy
    slope, intercept, r_val, p_val, stderr = linregress(x, y)

    # Scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.4, s=10)

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = intercept + slope * x_line
    plt.plot(x_line, y_line, linewidth=2,color='red')

    plt.xlabel("Genre cosine distance")
    plt.ylabel(f"LSA cosine distance (k={lsa_k})")
    plt.title(f"Artist {target_id}: Genre vs LSA distance (n={n})")

    # Annotate with r and p
    plt.text(
        0.5,
        0.9,
        f"LSRL: y = {slope:.3f}x + {intercept:.3f}\nR^2: {(r_val**2):.3f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", alpha=0.2),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # example: artist 1, k=800
    plot_artist_scatter(target_id=1, lsa_k=10, max_points=500000)
