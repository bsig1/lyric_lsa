import numpy as np
import sqlite3
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

TARGET_ARTIST_ID = 429061

def main(k: int):
    conn = sqlite3.connect("lyrics.db")
    curs = conn.cursor()

    # Dynamic table names based on k
    lsa_table = f"artist_lsa_{k}"
    out_table = f"artist_cosine_distance_{k}"  # or "artist_cosine_distance_{k}" if that's your schema

    # Get all other artists' vectors
    curs.execute(
        f"""
        SELECT artist_id, vec
        FROM {lsa_table}
        WHERE artist_id != ?
        ORDER BY artist_id
        """,
        (TARGET_ARTIST_ID,)
    )
    rows = curs.fetchall()

    # Get target artist vector
    curs.execute(
        f"""
        SELECT artist_id, vec
        FROM {lsa_table}
        WHERE artist_id = ?
        """,
        (TARGET_ARTIST_ID,)
    )
    wf = curs.fetchone()
    if wf is None:
        raise ValueError(f"Artist {TARGET_ARTIST_ID} not found in {lsa_table}")

    wf_id = wf[0]
    wf_vec = np.frombuffer(wf[1], np.float32)

    inserts = []
    for row in tqdm(rows, desc=f"k={k}"):
        artist_id = row[0]
        vec = np.frombuffer(row[1], np.float32)

        sim = cosine_similarity([wf_vec], [vec])[0][0]
        distance = float(1.0 - sim)

        inserts.append((wf_id, artist_id, distance))

    curs.executemany(
        f"""
        INSERT INTO {out_table}
        (artist_id1, artist_id2, distance)
        VALUES (?, ?, ?)
        """,
        inserts
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main(800)
