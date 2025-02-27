import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import ast

movies = pd.read_csv("movies_metadata.csv", low_memory=False)  # Fix potential DtypeWarning
ratings_small = pd.read_csv("ratings_small.csv")
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")
links = pd.read_csv("links.csv")
links_small = pd.read_csv("links_small.csv")
ratings = pd.read_csv("ratings.csv")


adresse = 'D:\\WORK\\INTO_CODE\\AGILE_Projekt\\Projekte\\PRJ_Movie\\satze\\'
adresse = 'C:\\Users\\Tetiana\\OneDrive\\Документы\\New folder\\satze\\'

import os

file_path = adresse+'credits.csv'
if not os.path.exists(file_path):
    print("Файл не найден! Проверьте путь.")

credits = pd.read_csv(adresse+'credits.csv',low_memory = False) #read_csv_file(adresse+'credits.csv') 
keywords = pd.read_csv(adresse+'keywords.csv',low_memory = False)
links_small = pd.read_csv(adresse+'links_small.csv',low_memory = False)
links = pd.read_csv(adresse+'links.csv',low_memory = False) 
ratings_small = pd.read_csv(adresse+'ratings_small.csv',low_memory = False)
ratings = pd.read_csv(adresse+'ratings.csv')
movies = pd.read_csv(adresse+'movies_metadata.csv',low_memory = False) #read_csv_file(adresse+'movies.csv')



# TICKET_2: Korrekte Behandlung und Standardisierung der Film- und Bewertungsdaten. Edit by tet.sydorenko 26.02.2025
def info_datatypes(df_name):
    print('COLUMNS and DATATYPES:\n')
    print(df_name.info())
  
# Delete spaces
def del_space(df, field_name):
    return df[field_name].str.strip()

# movies.csv: The main Movies Metadata file
info_datatypes(movies)

# adult #
print('Check data in first column:\n', movies['adult'].unique())
fltr_movies_adult = movies[(movies['adult'] != 'False') & 
                                                (movies['adult'] != 'True')]
#print(fltr_movies_adult)
movies = movies[(movies['adult'] == 'False') | 
                                        (movies['adult'] == 'True')]
print('Rows contain incorrect data and have been removed from the origin dataset.\nIncorrect raws have been saved in "fltr_movies_adult"')
#print(movies['adult'].unique())

movies['adult'] = movies['adult'].astype(bool) 
#movies['belongs_to_collection']
movies['budget'] = movies['budget'].astype('Int64') 
#movies['genres']
#movies['homepage']
movies['id'] = movies['id'].astype('int')
movies['imdb_id'] = del_space(movies, 'imdb_id')
movies['original_language'] = del_space(movies, 'original_language')
movies['original_title'] = del_space(movies, 'original_title')
movies['overview'] = del_space(movies, 'overview')
movies['popularity'] = movies['popularity'].astype(float).round(2)
#movies['poster_path']
#movies['production_companies']
#movies['production_countries']
movies['release_date'] = movies['release_date'].astype('datetime64[s]')
movies['revenue'] = movies['revenue'].astype('Int64') 
movies['runtime'] = movies['runtime'].astype('Int64') 
movies['spoken_languages'] = del_space(movies, 'spoken_languages')
movies['status'] = del_space(movies, 'status')
movies['tagline'] = del_space(movies, 'tagline')
movies['title'] = del_space(movies, 'title') 
#movies['video']
movies['vote_average'] = movies['vote_average'].astype(float).round(1)
movies['vote_count'] = movies['vote_count'].astype('Int64') 

## DUPLICATES FIND AND REMOVE
movies_duplicate_id = movies[movies['id'].duplicated(keep=False)]
print('movies_duplicate_by_id \n', movies_duplicate_id.loc[:,['id','title','imdb_id','original_language']].sort_values(by=['id', 'title'], ascending=[True, False]))
duplicate_cnt = movies_duplicate_id['id'].value_counts()
print('duplicate_cnt:\t', duplicate_cnt)

movies_duplicate_id = movies_duplicate_id.set_index('id')

print (movies_duplicate_id.sort_values(by=['id','title','imdb_id','original_language']))

def remove_duplicates_if_exist(df, columns, keep='first'):
    """
    :param df: DataFrame
    :param columns: List of columns
    :param keep: ('first', 'last', False)
                 - 'first' (default) — stay the first
                 - 'last' — stay the last
                 - False — delete all duplicates
    :return: DataFrame or Message
    """
    duplicates = df.duplicated(subset=columns, keep=False)

    if duplicates.any():  
        print(f"Duplicates by {columns} were found and will be deleted")
        df.drop_duplicates(subset=columns,keep=keep, inplace=True)
    else:
        print("There are no duplicates in DataFrame")
        return df

remove_duplicates_if_exist(movies,columns = ['id','title'])

st.write(movies)

###### 

# Convert the 'genres' column from a string to a list of dictionaries
movies['genres'] = movies['genres'].apply(ast.literal_eval)

# Expand the list of genres into separate rows
df_exploded = movies.explode('genres')

# Convert the 'genres' dictionary column into separate columns
df_genres = pd.json_normalize(df_exploded['genres'])

# Сброс индекса перед объединением
df_exploded = df_exploded.reset_index(drop=True)
df_genres = df_genres.reset_index(drop=True)

# Add the 'id' column from the original table as 'movie_id' for reference
df_genres['movie_id'] = df_exploded['id']

# Display the first few rows of the resulting DataFrame
print(df_genres.head())

# Save the processed data to a new CSV file
df_genres.to_csv("movies_genres.csv", index=False)

st.write(df_genres)

# TICKET_3: Erkennung und Behandlung fehlender Einträge in den Datensätzen. - Обнаружение отсутствующих или некорректных значений:

# Check Null None Nat ... and give Count of Nulls in Columns
def cnt_null_in_column(df, field_name):
    return f'Number of Null-Values in column "{field_name}": {df[field_name].isna().sum()}'

#####

for col in movies.columns:
    print('Column: ', col, '\tCount Null: ', movies[col].isna().sum())

movies_fltr_data = movies[movies['release_date'] < '1900-01-01']
print('Raws with release_date < "1900-01-01":\n', movies[movies['release_date'] < '1900-01-01'])

print('POPULARITY')
movies['popularity'] = movies['popularity'].fillna(movies['popularity'].mean)

print(cnt_null_in_column(movies,'vote_count'))

## TICKET 16-17 Integration einer Suchfunktion zur gezielten Ausfindung von Filmen.

@st.cache_data
def load_data():
    df = movies
    df = df[['title', 'genres', 'release_date', 'vote_average', 'vote_count']].dropna()
    return df

df = load_data()

def search_movies(query, data):
    return data[data['title'].str.contains(query, case=False, na=False)]

st.title("Movie Ratings and Trends Analysis")

search_query_title = st.text_input("Find the film by name:")

search_query_date = st.text_input("Find the film by realease date:")

if search_query_title:
    search_results = search_movies(search_query_title, df)
    st.write(f"### {len(search_results)} of films were found")
    st.dataframe(search_results)

if st.button("🔄 Clean filters"):
    st.experimental_rerun()
    
#4 Extract year from release_date
movies['year'] = movies['release_date'].dt.year

# Group by genre (assuming 'genres' column contains JSON-like strings)
import ast
movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if isinstance(x, str) else [])
movies_exploded = movies.explode('genres')  # One row per genre per movie

# Group by genre
genre_group = movies_exploded.groupby('genres').agg({'title': 'count'}).rename(columns={'title': 'count'})
print(genre_group.sort_values(by='count', ascending=False))

#5 Merge movies with ratings
ratings_summary = ratings.groupby('movieId').agg({'rating': ['mean', 'count', 'sum']})
ratings_summary.columns = ['average_rating', 'rating_count', 'total_rating']
ratings_summary.reset_index(inplace=True)

# Merge with movie titles
movies['movieId'] = pd.to_numeric(movies['id'], errors='coerce')
movies_ratings = movies.merge(ratings_summary, on='movieId', how='left')

# Movie_Ratings Display
movies_ratings[['title', 'average_rating', 'rating_count', 'total_rating']].head()


