import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

movies = pd.read_csv("movies_metadata.csv", low_memory=False)  # Fix potential DtypeWarning
ratings_small = pd.read_csv("ratings_small.csv")
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")
links = pd.read_csv("links.csv")
links_small = pd.read_csv("links_small.csv")
ratings = pd.read_csv("ratings.csv")

#4 Extract year from release_date
movies['year'] = movies['release_date'].dt.year

# Group by genre (assuming 'genres' column contains JSON-like strings)
import ast
movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if isinstance(x, str) else [])
movies_exploded = movies.explode('genres')  # One row per genre per movie

# Group by genre
genre_group = movies_exploded.groupby('genres').agg({'title': 'count'}).rename(columns={'title': 'count'})
print(genre_group.sort_values(by='count', ascending=False))

