import sqlite3
import numpy as np
from tqdm import tqdm

conn = sqlite3.connect("lyrics.db")
curs = conn.cursor()

curs.execute("""
SELECT artist_id, genre_id, COUNT(*)
FROM tracks
GROUP BY artist_id, genre_id
ORDER BY artist_id;
""")

rows = curs.fetchall()

artist_vecs = []
artist_genres = {}
current_artist = None

def finalize_artist(artist_id, genres_dict):
    """Convert genre counts to a normalized vector."""
    if not genres_dict:
        return None
    
    vec = [0] * 6
    total = sum(genres_dict.values())

    for genre_id, count in genres_dict.items():
        vec[genre_id - 1] = count / total  # genre IDs are 1–6

    vec = np.array(vec, dtype=np.float32)
    vec /= np.linalg.norm(vec)  # normalize
    return vec.tobytes()

for artist_id, genre_id, count in tqdm(rows):

    if current_artist is None:
        current_artist = artist_id

    if artist_id != current_artist:
        vec_bytes = finalize_artist(current_artist, artist_genres)
        if vec_bytes:
            artist_vecs.append([vec_bytes, current_artist])

        artist_genres = {}          # reset for next artist
        current_artist = artist_id  # update pointer

    # accumulate genre counts
    artist_genres[genre_id] = count

# finalize last artist
vec_bytes = finalize_artist(current_artist, artist_genres)
if vec_bytes:
    artist_vecs.append([vec_bytes, current_artist])

# -------- update DB ----------
curs.executemany("""
    UPDATE artists
    SET genres = ?
    WHERE id = ?
""", artist_vecs)

conn.commit()
conn.close()
