import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.music.moods import MOOD_RANGES

class MusicRecommender:
    def __init__(self,csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        self.df_new = self.df.copy()

        self.df_new.drop(columns=['Time_Signature', 'Key', 'Speechiness', 'Instrumentalness'], inplace=True)
        self.features = self.df_new[['Danceability', 'Energy', 'Loudness', 'Mode', 'Acousticness', 'Valence', 'Tempo']]
        
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
    
    def select_mood(self, mood):
        ranges = MOOD_RANGES[mood]
        row = {}
        # randomly generated the inpute_vector
        row['Danceability'] = np.random.uniform(*ranges['danceability']) 
        row['Energy'] = np.random.uniform(*ranges['energy']) 
        row['Loudness'] = np.random.uniform(*ranges['loudness']) 
        row['Mode'] = ranges['mode']
        row['Acousticness'] = np.random.uniform(*ranges['acousticness']) 
        row['Valence'] = np.random.uniform(*ranges['valence']) 
        row['Tempo'] = np.random.uniform(*ranges['tempo']) 
        # create df for prediction
        mood_df = pd.DataFrame([row], columns=self.features.columns)
        return mood_df
    
    def recommend(self,mood,top_k: int=5):

        mood_vector = self.select_mood(mood)
        scaled_input = self.scaler.transform(mood_vector)
        #for debugging
        #print(f"{mood=} raw vector:", mood_vector)
        #print(f"{mood=} scaled vector:", scaled_input)
        
        #finding similarity
        similarities = cosine_similarity(scaled_input, self.scaled_features)[0]

        #getting top k songs
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = self.df.iloc[top_indices].copy()
        results_track = results['Track']
        results_artist = results['Artist']
        return results_track,results_artist