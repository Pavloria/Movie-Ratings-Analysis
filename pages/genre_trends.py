import matplotlib.pyplot as plt
import pandas as pd
import ast
import streamlit as st

df_metadata=pd.read_csv("archive/movies_metadata.csv")
df_rating=pd.read_csv("archive/ratings.csv")



#Lösche doppelt Wete in 'id' Spalte
df_metadata = df_metadata.drop_duplicates(subset='id', keep='first')
#Konvertiert zu String
df_metadata['id'] = df_metadata['id'].astype(str)
df_rating['movieId'] = df_rating['movieId'].astype(str)
df_metadata=df_metadata[['id','genres']].dropna()
df_rating=df_rating[['movieId','rating']].dropna()


#Konvertieren eine String zu eine Liste,wenn möglich ist
def convert_string_to_list(value):
    if isinstance(value, str):  
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []  # Falls Fehler auftreten, gib eine leere Liste zurück
    return value 
# 2 DataFrames zusammenfügen
df_merged=pd.merge(df_metadata,df_rating,left_on='id',right_on='movieId',how='inner')
#df_merged = df_merged.sample(n=4000)  
#Values in 'genre' Spalte in eine String konvertieren
df_merged['genres'] = df_merged['genres'].apply(lambda x:convert_string_to_list(x)) 
#'genre_names' Spalte erstellen
#Namen von Genres von 'genres' in genre_names' übertragen
df_merged['genre_names'] = df_merged['genres'].apply(lambda x: [genre['name'] for genre in x])

df_merged['genre_names'] = df_merged['genre_names'].apply(lambda x:convert_string_to_list(x)) 
#Für jerer element in Liste in  'genre_nammes' Spalte sepatrate Zeile erstellen
df_merged['genre_names']=df_merged['genre_names'].apply(lambda li:[i for i in li])
#Füllen entstehende Zeile in andere Zeilen mit entsprechende Values
exploded_df = df_merged.explode('genre_names')

#Finden Mittlewert von Bewertungen pro ein Film
rating_per_movie=exploded_df.groupby('movieId')['rating'].mean()
#Ene Serie und ein DataFrame zusammenfügen
df_rating_per_movie=pd.merge(exploded_df[['movieId','genre_names']],rating_per_movie,on='movieId',how='inner')
#eine Serie mit Mittlewert von Bewertungen pro ein Genre erstellen
rating_per_genre=df_rating_per_movie.groupby('genre_names')['rating'].mean()




# Balkendiagramm erstellen
plt.figure(figsize=(15, 8))
plt.bar(rating_per_genre.index, rating_per_genre.values, color="royalblue")

# Achsenbeschriftungen und Titel setzen
plt.xticks(rotation=45)
plt.xlabel("Genre")
plt.ylabel('Durchschnittliche Bewertung')
plt.title("Durchschnittliche Bewertung pro Genre")

# Diagramm anzeigen
st.pyplot(plt)

