"""Build data/songs.csv from a curated list of REAL, well-known songs.

IMPORTANT: the audio-feature values below are my own approximate estimates
(roughly in the style of Spotify's audio features), NOT exact API values.
They are good enough to demo the recommender. For your final submission, replace
data/songs.csv with the real Kaggle "Spotify Tracks Dataset" - see README.

Per-song values: danceability, energy, speechiness, acousticness,
instrumentalness, valence, tempo(BPM). Loudness (dB) and liveness are derived.
"""
import os
import numpy as np
import pandas as pd

# (title, artist, danceability, energy, speechiness, acousticness, instrumentalness, valence, tempo)
SONGS = {
    "Pop": [
        ("Blinding Lights", "The Weeknd", 0.51, 0.80, 0.06, 0.00, 0.00, 0.33, 171),
        ("Shape of You", "Ed Sheeran", 0.83, 0.65, 0.08, 0.58, 0.00, 0.93, 96),
        ("Levitating", "Dua Lipa", 0.70, 0.83, 0.06, 0.01, 0.00, 0.91, 103),
        ("Uptown Funk", "Mark Ronson ft. Bruno Mars", 0.86, 0.61, 0.08, 0.01, 0.00, 0.93, 115),
        ("bad guy", "Billie Eilish", 0.70, 0.43, 0.38, 0.33, 0.13, 0.56, 135),
        ("Happy", "Pharrell Williams", 0.65, 0.82, 0.05, 0.22, 0.00, 0.96, 160),
        ("Shake It Off", "Taylor Swift", 0.65, 0.80, 0.14, 0.04, 0.00, 0.94, 160),
        ("Rolling in the Deep", "Adele", 0.73, 0.77, 0.03, 0.14, 0.00, 0.51, 105),
        ("Someone Like You", "Adele", 0.56, 0.32, 0.03, 0.89, 0.00, 0.29, 135),
        ("Don't Start Now", "Dua Lipa", 0.79, 0.79, 0.05, 0.01, 0.00, 0.68, 124),
        ("Watermelon Sugar", "Harry Styles", 0.55, 0.82, 0.05, 0.12, 0.00, 0.56, 95),
        ("Dance Monkey", "Tones and I", 0.82, 0.59, 0.09, 0.69, 0.00, 0.51, 98),
    ],
    "Rock": [
        ("Bohemian Rhapsody", "Queen", 0.39, 0.40, 0.05, 0.29, 0.00, 0.23, 72),
        ("Smells Like Teen Spirit", "Nirvana", 0.50, 0.91, 0.06, 0.00, 0.00, 0.72, 117),
        ("Sweet Child O' Mine", "Guns N' Roses", 0.45, 0.90, 0.05, 0.04, 0.00, 0.60, 125),
        ("Hotel California", "Eagles", 0.58, 0.51, 0.03, 0.01, 0.00, 0.61, 147),
        ("Back In Black", "AC/DC", 0.31, 0.70, 0.04, 0.01, 0.00, 0.76, 189),
        ("Stairway to Heaven", "Led Zeppelin", 0.33, 0.34, 0.03, 0.58, 0.00, 0.20, 82),
        ("Come Together", "The Beatles", 0.53, 0.38, 0.04, 0.04, 0.20, 0.19, 82),
        ("Seven Nation Army", "The White Stripes", 0.74, 0.46, 0.08, 0.01, 0.40, 0.32, 124),
        ("Mr. Brightside", "The Killers", 0.36, 0.93, 0.08, 0.00, 0.00, 0.24, 148),
        ("Dream On", "Aerosmith", 0.38, 0.51, 0.03, 0.12, 0.00, 0.24, 81),
        ("Wonderwall", "Oasis", 0.38, 0.68, 0.03, 0.03, 0.00, 0.40, 87),
        ("Do I Wanna Know?", "Arctic Monkeys", 0.55, 0.54, 0.03, 0.19, 0.00, 0.40, 85),
    ],
    "Hip-Hop": [
        ("Lose Yourself", "Eminem", 0.69, 0.74, 0.26, 0.01, 0.00, 0.06, 171),
        ("God's Plan", "Drake", 0.75, 0.45, 0.11, 0.03, 0.00, 0.36, 77),
        ("HUMBLE.", "Kendrick Lamar", 0.91, 0.60, 0.10, 0.00, 0.00, 0.42, 150),
        ("SICKO MODE", "Travis Scott", 0.83, 0.73, 0.22, 0.01, 0.00, 0.45, 155),
        ("In Da Club", "50 Cent", 0.90, 0.71, 0.37, 0.26, 0.00, 0.78, 90),
        ("Empire State of Mind", "Jay-Z ft. Alicia Keys", 0.50, 0.82, 0.16, 0.04, 0.00, 0.42, 173),
        ("Gold Digger", "Kanye West", 0.62, 0.69, 0.28, 0.09, 0.00, 0.64, 93),
        ("Juicy", "The Notorious B.I.G.", 0.89, 0.81, 0.34, 0.11, 0.00, 0.65, 94),
        ("rockstar", "Post Malone ft. 21 Savage", 0.59, 0.52, 0.07, 0.12, 0.00, 0.12, 160),
        ("Sunflower", "Post Malone & Swae Lee", 0.76, 0.48, 0.05, 0.56, 0.00, 0.91, 90),
        ("Lucid Dreams", "Juice WRLD", 0.51, 0.56, 0.20, 0.35, 0.00, 0.22, 84),
        ("Mask Off", "Future", 0.83, 0.43, 0.08, 0.01, 0.00, 0.28, 150),
    ],
    "Classical": [
        ("Moonlight Sonata", "Ludwig van Beethoven", 0.25, 0.05, 0.04, 0.99, 0.90, 0.05, 56),
        ("Für Elise", "Ludwig van Beethoven", 0.35, 0.07, 0.04, 0.99, 0.90, 0.30, 138),
        ("Clair de Lune", "Claude Debussy", 0.30, 0.01, 0.04, 0.99, 0.90, 0.05, 66),
        ("Canon in D", "Johann Pachelbel", 0.30, 0.12, 0.04, 0.95, 0.90, 0.20, 100),
        ("Spring (The Four Seasons)", "Antonio Vivaldi", 0.35, 0.35, 0.04, 0.90, 0.90, 0.60, 130),
        ("Symphony No. 5", "Ludwig van Beethoven", 0.25, 0.45, 0.04, 0.80, 0.90, 0.15, 108),
        ("Eine kleine Nachtmusik", "Wolfgang Amadeus Mozart", 0.45, 0.40, 0.04, 0.90, 0.90, 0.85, 130),
        ("Gymnopédie No. 1", "Erik Satie", 0.40, 0.03, 0.04, 0.99, 0.95, 0.15, 72),
        ("Air on the G String", "Johann Sebastian Bach", 0.20, 0.10, 0.04, 0.97, 0.90, 0.15, 60),
        ("Ride of the Valkyries", "Richard Wagner", 0.25, 0.55, 0.04, 0.80, 0.90, 0.15, 140),
        ("Nocturne Op. 9 No. 2", "Frédéric Chopin", 0.30, 0.05, 0.04, 0.99, 0.90, 0.15, 60),
        ("Ode to Joy", "Ludwig van Beethoven", 0.35, 0.25, 0.04, 0.90, 0.90, 0.50, 120),
    ],
    "Jazz": [
        ("Take Five", "Dave Brubeck Quartet", 0.46, 0.38, 0.04, 0.55, 0.80, 0.60, 174),
        ("So What", "Miles Davis", 0.55, 0.25, 0.04, 0.68, 0.85, 0.45, 136),
        ("Fly Me to the Moon", "Frank Sinatra", 0.40, 0.28, 0.04, 0.83, 0.00, 0.45, 119),
        ("What a Wonderful World", "Louis Armstrong", 0.30, 0.15, 0.04, 0.85, 0.00, 0.35, 70),
        ("Summertime", "Ella Fitzgerald & Louis Armstrong", 0.38, 0.12, 0.04, 0.95, 0.00, 0.30, 70),
        ("My Favorite Things", "John Coltrane", 0.45, 0.40, 0.04, 0.50, 0.80, 0.65, 170),
        ("Blue in Green", "Miles Davis", 0.30, 0.08, 0.03, 0.95, 0.90, 0.15, 70),
        ("'Round Midnight", "Thelonious Monk", 0.40, 0.08, 0.04, 0.90, 0.90, 0.20, 85),
        ("The Girl from Ipanema", "Stan Getz & João Gilberto", 0.55, 0.20, 0.04, 0.80, 0.10, 0.60, 128),
        ("Autumn Leaves", "Cannonball Adderley", 0.45, 0.35, 0.04, 0.70, 0.80, 0.60, 120),
        ("Feeling Good", "Nina Simone", 0.45, 0.55, 0.04, 0.75, 0.00, 0.35, 85),
        ("Georgia on My Mind", "Ray Charles", 0.30, 0.20, 0.04, 0.80, 0.00, 0.30, 75),
    ],
    "Electronic": [
        ("Titanium", "David Guetta ft. Sia", 0.60, 0.79, 0.10, 0.07, 0.00, 0.30, 126),
        ("Levels", "Avicii", 0.60, 0.89, 0.05, 0.02, 0.70, 0.46, 126),
        ("Wake Me Up", "Avicii", 0.53, 0.78, 0.05, 0.04, 0.00, 0.66, 124),
        ("Animals", "Martin Garrix", 0.60, 0.90, 0.05, 0.01, 0.60, 0.50, 128),
        ("Strobe", "deadmau5", 0.45, 0.62, 0.04, 0.01, 0.90, 0.15, 128),
        ("Around the World", "Daft Punk", 0.74, 0.80, 0.06, 0.04, 0.70, 0.65, 121),
        ("One More Time", "Daft Punk", 0.62, 0.70, 0.07, 0.01, 0.00, 0.52, 123),
        ("Scary Monsters and Nice Sprites", "Skrillex", 0.70, 0.90, 0.08, 0.01, 0.10, 0.35, 140),
        ("Clarity", "Zedd ft. Foxes", 0.51, 0.77, 0.04, 0.04, 0.00, 0.33, 128),
        ("Closer", "The Chainsmokers ft. Halsey", 0.75, 0.52, 0.03, 0.41, 0.00, 0.66, 95),
        ("Don't You Worry Child", "Swedish House Mafia", 0.61, 0.65, 0.04, 0.12, 0.00, 0.28, 129),
        ("Opus", "Eric Prydz", 0.55, 0.80, 0.04, 0.00, 0.90, 0.12, 126),
    ],
    "Folk": [
        ("Blowin' in the Wind", "Bob Dylan", 0.40, 0.20, 0.04, 0.90, 0.00, 0.55, 174),
        ("Take Me Home, Country Roads", "John Denver", 0.57, 0.60, 0.04, 0.55, 0.00, 0.65, 82),
        ("Fast Car", "Tracy Chapman", 0.62, 0.30, 0.04, 0.65, 0.00, 0.35, 104),
        ("Hallelujah", "Jeff Buckley", 0.28, 0.22, 0.04, 0.64, 0.00, 0.10, 56),
        ("Skinny Love", "Bon Iver", 0.40, 0.35, 0.04, 0.70, 0.00, 0.20, 120),
        ("Ho Hey", "The Lumineers", 0.62, 0.67, 0.04, 0.55, 0.00, 0.80, 80),
        ("Riptide", "Vance Joy", 0.50, 0.73, 0.04, 0.43, 0.00, 0.78, 102),
        ("I Will Wait", "Mumford & Sons", 0.40, 0.75, 0.04, 0.15, 0.00, 0.48, 120),
        ("The Boxer", "Simon & Garfunkel", 0.50, 0.40, 0.03, 0.75, 0.00, 0.55, 100),
        ("Wagon Wheel", "Old Crow Medicine Show", 0.60, 0.55, 0.03, 0.30, 0.00, 0.75, 148),
        ("Landslide", "Fleetwood Mac", 0.45, 0.30, 0.04, 0.80, 0.00, 0.35, 70),
        ("Hey There Delilah", "Plain White T's", 0.66, 0.29, 0.03, 0.86, 0.00, 0.40, 104),
    ],
    "Bollywood": [
        ("Tum Hi Ho", "Arijit Singh", 0.52, 0.45, 0.04, 0.80, 0.00, 0.26, 85),
        ("Kal Ho Naa Ho", "Sonu Nigam", 0.40, 0.45, 0.04, 0.65, 0.00, 0.40, 80),
        ("Chaiyya Chaiyya", "Sukhwinder Singh & Sapna Awasthi", 0.78, 0.88, 0.05, 0.20, 0.00, 0.80, 100),
        ("Jai Ho", "A. R. Rahman", 0.60, 0.88, 0.06, 0.10, 0.00, 0.75, 130),
        ("Senorita", "Farhan Akhtar, Hrithik Roshan, Abhay Deol", 0.66, 0.60, 0.04, 0.35, 0.00, 0.75, 110),
        ("Badtameez Dil", "Benny Dayal", 0.75, 0.85, 0.05, 0.10, 0.00, 0.85, 130),
        ("Kesariya", "Arijit Singh", 0.55, 0.55, 0.04, 0.40, 0.00, 0.50, 90),
        ("Ghungroo", "Arijit Singh & Shilpa Rao", 0.80, 0.80, 0.05, 0.15, 0.00, 0.70, 105),
        ("Channa Mereya", "Arijit Singh", 0.45, 0.38, 0.04, 0.65, 0.00, 0.20, 85),
        ("Dil Chahta Hai", "Shankar Mahadevan, Udit Narayan, Siddharth Mahadevan", 0.65, 0.70, 0.04, 0.30, 0.00, 0.70, 110),
        ("Kabira", "Tochi Raina & Rekha Bhardwaj", 0.50, 0.65, 0.04, 0.45, 0.00, 0.50, 100),
        ("Tujhe Dekha To", "Kumar Sanu & Lata Mangeshkar", 0.60, 0.60, 0.04, 0.45, 0.00, 0.80, 100),
    ],
    "Metal": [
        ("Master of Puppets", "Metallica", 0.40, 0.90, 0.06, 0.00, 0.15, 0.50, 212),
        ("Enter Sandman", "Metallica", 0.58, 0.90, 0.04, 0.00, 0.00, 0.52, 123),
        ("Paranoid", "Black Sabbath", 0.45, 0.88, 0.04, 0.00, 0.00, 0.65, 163),
        ("Ace of Spades", "Motörhead", 0.30, 0.95, 0.08, 0.00, 0.00, 0.55, 140),
        ("Painkiller", "Judas Priest", 0.35, 0.97, 0.05, 0.00, 0.10, 0.35, 120),
        ("Raining Blood", "Slayer", 0.30, 0.98, 0.07, 0.00, 0.20, 0.15, 123),
        ("Holy Wars... The Punishment Due", "Megadeth", 0.30, 0.95, 0.06, 0.00, 0.40, 0.25, 145),
        ("Du Hast", "Rammstein", 0.55, 0.85, 0.05, 0.05, 0.00, 0.50, 92),
        ("Toxicity", "System of a Down", 0.40, 0.90, 0.06, 0.00, 0.00, 0.44, 150),
        ("Chop Suey!", "System of a Down", 0.42, 0.89, 0.06, 0.00, 0.00, 0.25, 127),
        ("Duality", "Slipknot", 0.40, 0.97, 0.08, 0.00, 0.00, 0.50, 138),
        ("Symphony of Destruction", "Megadeth", 0.55, 0.80, 0.04, 0.00, 0.00, 0.35, 125),
    ],
    "R&B": [
        ("Crazy in Love", "Beyoncé ft. Jay-Z", 0.65, 0.75, 0.10, 0.05, 0.00, 0.70, 99),
        ("No Scrubs", "TLC", 0.70, 0.60, 0.05, 0.15, 0.00, 0.60, 93),
        ("Say My Name", "Destiny's Child", 0.71, 0.74, 0.08, 0.15, 0.00, 0.55, 138),
        ("Redbone", "Childish Gambino", 0.74, 0.43, 0.06, 0.20, 0.00, 0.60, 160),
        ("Thinking Bout You", "Frank Ocean", 0.45, 0.40, 0.04, 0.55, 0.00, 0.30, 134),
        ("Earned It", "The Weeknd", 0.65, 0.40, 0.04, 0.25, 0.00, 0.35, 120),
        ("Adorn", "Miguel", 0.67, 0.57, 0.04, 0.20, 0.00, 0.60, 90),
        ("Location", "Khalid", 0.74, 0.65, 0.05, 0.40, 0.00, 0.50, 80),
        ("Best Part", "Daniel Caesar ft. H.E.R.", 0.45, 0.38, 0.04, 0.60, 0.00, 0.40, 90),
        ("Let's Stay Together", "Al Green", 0.60, 0.35, 0.04, 0.40, 0.00, 0.80, 100),
        ("Superstition", "Stevie Wonder", 0.65, 0.65, 0.07, 0.10, 0.00, 0.85, 100),
        ("Ordinary People", "John Legend", 0.50, 0.30, 0.04, 0.80, 0.00, 0.30, 70),
    ],
}

rng = np.random.default_rng(7)
rows = []
for genre, songs in SONGS.items():
    for title, artist, dance, energy, speech, acoustic, instr, valence, tempo in songs:
        # derived features (rough approximations)
        loudness = -20 + 17 * energy - 4 * acoustic + rng.normal(0, 1.0)
        liveness = float(np.clip(rng.normal(0.17, 0.06), 0.05, 0.5))
        rows.append({
            "track_name": title, "artist": artist, "genre": genre,
            "danceability": dance, "energy": energy, "loudness": round(loudness, 2),
            "speechiness": speech, "acousticness": acoustic,
            "instrumentalness": instr, "liveness": round(liveness, 3),
            "valence": valence, "tempo": tempo,
        })

df = pd.DataFrame(rows)
df.insert(0, "track_id", range(len(df)))
os.makedirs("data", exist_ok=True)
df.to_csv("data/songs.csv", index=False)
print(f"Saved data/songs.csv with {len(df)} real songs across {df.genre.nunique()} genres")
