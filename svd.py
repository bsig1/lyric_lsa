import sqlite3
from tqdm import tqdm
from time import time
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import numpy as np

ct = time()
def stop_watch():
    global ct
    time_spent = time()-ct
    hours = time_spent//3600
    minutes = (time_spent%3600)//60
    seconds = time_spent%60

    output = []
    if hours!=0: output.append(f"{int(hours)} H")
    if minutes!=0: output.append(f"{int(minutes)} M")
    if seconds!=0: output.append(f"{seconds:.2f} S")

    print(f"Executed In: {' '.join(output)}\n")
    ct = time()

conn = sqlite3.connect("lyrics.db")
cur = conn.cursor()

# Get all distinct term_ids and artist_ids
print("Fetching Data...")
cur.execute("SELECT DISTINCT term_id FROM artist_term_counts")
term_ids = [row[0] for row in cur.fetchall()]

cur.execute("SELECT DISTINCT artist_id FROM artist_term_counts")
artist_ids = [row[0] for row in cur.fetchall()]
stop_watch()

# Map DB IDs → 0..n-1 indices
term_to_row   = {tid: i for i, tid in enumerate(term_ids)}
artist_to_col = {aid: j for j, aid in enumerate(artist_ids)}

n_terms   = len(term_ids)
n_artists = len(artist_ids)

# 2. Build sparse term × artist count matrix
print("Building Matrix...")
print(f"n_terms: {n_terms:,} X n_artists: {n_artists:,}")
rows, cols, data = [], [], []

for artist_id, term_id, count in cur.execute("""
    SELECT artist_id, term_id, count
    FROM artist_term_counts
"""):
    # look up row/col indices
    r = term_to_row[term_id]
    c = artist_to_col[artist_id]
    rows.append(r)
    cols.append(c)
    data.append(float(count))

X = coo_matrix((data, (rows, cols)), shape=(n_terms, n_artists))
stop_watch()

# 3. TF–IDF weighting (term × artist)
k = 800
print("Running SVD")
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X) 
X_tfidf_T = X_tfidf.T# Transpose: now matrix is (artists × terms)
artist_lsa = TruncatedSVD(n_components=k, random_state=0).fit_transform(X_tfidf_T)
print("Artist LSA shape:", artist_lsa.shape)  # (n_artists, k)
stop_watch()


# Normalize to unit length: v = v / ||v||
norms = np.linalg.norm(artist_lsa, axis=1, keepdims=True)
artist_lsa = artist_lsa / norms

print("Building LSA Table")

cur.execute(f"""
    CREATE TABLE IF NOT EXISTS artist_lsa_{k} (
        artist_id INTEGER PRIMARY KEY,
        vec BLOB
    )
""")
conn.commit()

# cast to float32 to halve size
artist_lsa32 = artist_lsa.astype("float32", copy=False)

def row_generator():
    for j, artist_id in enumerate(artist_ids):
        vec_bytes = artist_lsa32[j].tobytes()
        yield (artist_id, vec_bytes)

cur.executemany(f"""
    INSERT OR REPLACE INTO artist_lsa_{k} (artist_id, vec)
    VALUES (?, ?)
""", tqdm(row_generator(), total=len(artist_ids), desc="Inserting"))

conn.commit()
conn.close()

