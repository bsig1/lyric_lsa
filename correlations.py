from scipy.stats import pearsonr
import sqlite3
from tqdm import tqdm

conn = sqlite3.connect("lyrics.db")
curs = conn.cursor()

k = 800
restrict_genre_id = -1

genreid = f"WHERE shared_genre_id IS NULL OR shared_genre_id={restrict_genre_id}" if restrict_genre_id != -1 else ""

curs.execute(f"""
    SELECT (1.0-distance) as cos_sim,genre_score as gen_score FROM artist_cosine_distance_{k} {genreid}
""")
distance_sims = []
genre_sims = []
for row in tqdm(curs.fetchall()):
    distance_sims.append(row[0])
    genre_sims.append(row[1])

r, p = pearsonr(distance_sims, genre_sims)
print("Correlation:", r)
print("p-value:", p)