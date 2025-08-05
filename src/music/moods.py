MOOD_RANGES = {
    "Happy": {
        "danceability": (0.7, 1.0),
        "energy": (0.7, 1.0),
        "valence": (0.7, 1.0),
        "loudness": (-6, 0),
        "acousticness": (0.0, 0.3),
        "tempo": (100, 140),
        "mode": 1
    },
    "Sad": {
        "danceability": (0.0, 0.5),
        "energy": (0.2, 0.5),
        "valence": (0.0, 0.4),
        "loudness": (-30, -8),
        "acousticness": (0.4, 1.0),
        "tempo": (60, 90),
        "mode": 0
    },
    "Angry": {
        "danceability": (0.4, 0.7),
        "energy": (0.8, 1.0),
        "valence": (0.0, 0.3),
        "loudness": (-5, 0),
        "acousticness": (0.0, 0.2),
        "tempo": (110, 160),
        "mode": 0
    },
    "Disgust": {
        "danceability": (0.0, 0.5),
        "energy": (0.3, 0.6),
        "valence": (0.0, 0.3),
        "loudness": (-12, -6),
        "acousticness": (0.4, 0.9),
        "tempo": (60, 100),
        "mode": 0
    },
    "Fear": {
        "danceability": (0.0, 0.4),
        "energy": (0.4, 0.8),
        "valence": (0.0, 0.3),
        "loudness": (-15, -8),
        "acousticness": (0.3, 0.7),
        "tempo": (70, 120),
        "mode": 0
    },
    "Neutral": {
        "danceability": (0.4, 0.6),
        "energy": (0.4, 0.6),
        "valence": (0.4, 0.6),
        "loudness": (-10, -6),
        "acousticness": (0.2, 0.6),
        "tempo": (80, 120),
        "mode": 1  # or 0, both acceptable
    },
    "Surprise": {
        "danceability": (0.6, 1.0),
        "energy": (0.7, 1.0),
        "valence": (0.5, 0.9),
        "loudness": (-6, 0),
        "acousticness": (0.0, 0.4),
        "tempo": (110, 150),
        "mode": 1
    }
}