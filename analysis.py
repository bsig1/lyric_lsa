import sqlite3
import spacy
from collections import Counter
import time

# -----------------------
# spaCy + tokenizer setup
# -----------------------
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner", "textcat", "tok2vec"]
)

_ALLOWED = set("abcdefghijklmnopqrstuvwxyz'")
_TRANS = str.maketrans(
    {chr(i): " " for i in range(128) if chr(i) not in _ALLOWED}
)

def c_fast_tokenize(text: str):
    """
    C-optimized tokenizer:
    - Lowercases entire string (C)
    - Translates all NON-[a-z'] chars to space (C)
    - Splits on whitespace (C)
    """
    return text.lower().translate(_TRANS).split()

def lemmatize_docs(texts, batch_size=512, n_process=12):
    """
    texts: iterable of *strings* (one lyric per item)
    yields: Counter of lemmas per doc
    """
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        yield Counter(t.lemma_ for t in doc if not t.is_space)

def main():
    ARTIST_ID = 429061
    db_path = "lyrics.db"

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    cur = conn.cursor()

    term_id_cache = {}

    print("Loading tracks for artist", ARTIST_ID)
    t0 = time.time()

    cur.execute("""
        SELECT id, lyrics
        FROM tracks
        WHERE lyrics IS NOT NULL
          AND lyrics != ''
          AND artist_id = ?
    """, (ARTIST_ID,))
    rows = cur.fetchall()

    print(f"Found {len(rows)} tracks for artist {ARTIST_ID} in {time.time() - t0:.2f}s")

    if not rows:
        conn.close()
        return

    # -----------------------
    # Pre-clean lyrics
    # -----------------------
    ids = []
    cleaned_texts = []

    for track_id, lyrics in rows:
        ids.append(track_id)
        tokens = c_fast_tokenize(lyrics or "")
        cleaned_texts.append(" ".join(tokens))

    # -----------------------
    # Batched lemmatization
    # -----------------------
    print("Starting lemmatization...")
    t1 = time.time()
    lemma_counters = list(lemmatize_docs(cleaned_texts))
    print("Lemmatization done in", time.time() - t1, "seconds")

    # -----------------------
    # DB updates
    # -----------------------
    t2 = time.time()
    for track_id, lemma_counts in zip(ids, lemma_counters):
        # lemma_counts is Counter(lemma -> count) for THIS track
        for lemma, count in lemma_counts.items():
            # 1) get or create term_id (with cache)
            term_id = term_id_cache.get(lemma)
            if term_id is None:
                # insert if new
                conn.execute("""
                    INSERT OR IGNORE INTO terms (term)
                    VALUES (?);
                """, (lemma,))

                # fetch id
                cur.execute("SELECT id FROM terms WHERE term = ?;", (lemma,))
                row = cur.fetchone()
                if row is None:
                    continue
                term_id = row[0]
                term_id_cache[lemma] = term_id

            # 3) update per-track count
            conn.execute("""
                INSERT INTO track_term_counts (track_id, term_id, count)
                VALUES (?, ?, ?)
                ON CONFLICT(track_id, term_id) DO UPDATE SET
                    count = track_term_counts.count + excluded.count;
            """, (track_id, term_id, count))

    conn.commit()
    print("DB updates done in", time.time() - t2, "seconds")
    conn.close()

if __name__ == '__main__':
    main()
