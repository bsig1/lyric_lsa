import csv
import sqlite3

conn = sqlite3.connect("lyrics.db")
cursor = conn.cursor()

with open("output.csv","r",encoding='utf-8') as f:
    reader = csv.DictReader(f)

    for row in reader:
        statement = """
        INSERT INTO tracks (title,tag,lyrics,year,artist,features,views,language,language_cld3,language_ft)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        cursor.execute(statement,
            [
                row['title'],
                row['tag'],
                row['lyrics'],
                row['year'],
                row['artist'],
                row['features'],
                row['views'],
                row['language'],
                row['language_cld3'],
                row['language_ft']
            ]
        )
        conn.commit()
cursor.close()