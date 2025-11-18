import numpy as np
import sqlite3
import random
from itertools import combinations
from tqdm import tqdm

def main(k):
    ARTIST_TABLE_LEN = 727_824
    BATCH_SIZE = 100
    K_VAL = k
    MAX_ITERATIONS = 10_000

    conn = sqlite3.connect("lyrics.db")
    curs = conn.cursor()

    curs.execute(f"""
    CREATE TABLE IF NOT EXISTS artist_cosine_distance_{K_VAL}(
        artist_id1 INTEGER NOT NULL,
        artist_id2 INTEGER NOT NULL,
        distance REAL NOT NULL,
        PRIMARY KEY (artist_id1,artist_id2),
        FOREIGN KEY (artist_id1) REFERENCES artists(id),
        FOREIGN KEY (artist_id2) REFERENCES artists(id)
    );
    """)
    conn.commit()

    # Optional but good: one big transaction around all inserts
    conn.execute("BEGIN")

    for _ in tqdm(range(MAX_ITERATIONS), desc="Batches"):
        start_idx = random.randint(0, ARTIST_TABLE_LEN - BATCH_SIZE - 1)

        curs.execute(f"""
            SELECT artist_id, vec
            FROM artist_lsa_{K_VAL}
            ORDER BY artist_id
            LIMIT ? OFFSET ?
        """, (BATCH_SIZE, start_idx))

        rows = curs.fetchall()
        n = len(rows)
        if n < 2:
            continue

        artist_ids = np.array([r[0] for r in rows], dtype=np.int64)
        vecs = [np.frombuffer(r[1], np.float32) for r in rows]
        M = np.vstack(vecs)  # shape: (n, K_VAL)

        # Normalize rows to unit length
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        # Avoid divide-by-zero
        norms[norms == 0] = 1.0
        M = M / norms

        # Cosine similarity matrix
        G = M @ M.T  # shape: (n, n)

        # Upper triangle indices (i < j)
        iu, ju = np.triu_indices(n, k=1)

        artist1 = artist_ids[iu]
        artist2 = artist_ids[ju]
        distances = (1.0 - G[iu, ju]).astype(float)

        inserts = list(zip(artist1.tolist(),
                        artist2.tolist(),
                        distances.tolist()))

        curs.executemany(f"""
            INSERT OR IGNORE INTO artist_cosine_distance_{K_VAL}
            (artist_id1, artist_id2, distance)
            VALUES (?,?,?)
        """, inserts)

    # Commit once
    conn.commit()
    conn.close()

if __name__ == '__main__':
    main(800)