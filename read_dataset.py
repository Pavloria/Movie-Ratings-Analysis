import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import ast

# TICKET_1: Read info from csv-file. Edit by tet.sydorenko 24.02.2025
def read_csv_file(file_name):
    return pd.read_csv(file_name,low_memory = False)

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
movies_metadata = pd.read_csv(adresse+'movies_metadata.csv',low_memory = False) #read_csv_file(adresse+'movies_metadata.csv')



# TICKET_2: Korrekte Behandlung und Standardisierung der Film- und Bewertungsdaten. Edit by tet.sydorenko 26.02.2025
def info_datatypes(df_name):
    print('COLUMNS and DATATYPES:\n')
    print(df_name.info())
  
# Delete spaces
def del_space(df, field_name):
    return df[field_name].str.strip()

# movies_metadata.csv: The main Movies Metadata file
info_datatypes(movies_metadata)

# adult #
print('Check data in first column:\n', movies_metadata['adult'].unique())
fltr_movies_metadata_adult = movies_metadata[(movies_metadata['adult'] != 'False') & 
                                                (movies_metadata['adult'] != 'True')]
#print(fltr_movies_metadata_adult)
movies_metadata = movies_metadata[(movies_metadata['adult'] == 'False') | 
                                        (movies_metadata['adult'] == 'True')]
print('Rows contain incorrect data and have been removed from the origin dataset.\nIncorrect raws have been saved in "fltr_movies_metadata_adult"')
#print(movies_metadata['adult'].unique())

movies_metadata['adult'] = movies_metadata['adult'].astype(bool) 
#movies_metadata['belongs_to_collection']
movies_metadata['budget'] = movies_metadata['budget'].astype('Int64') 
#movies_metadata['genres']
#movies_metadata['homepage']
movies_metadata['id'] = movies_metadata['id'].astype('int')
movies_metadata['imdb_id'] = del_space(movies_metadata, 'imdb_id')
movies_metadata['original_language'] = del_space(movies_metadata, 'original_language')
movies_metadata['original_title'] = del_space(movies_metadata, 'original_title')
movies_metadata['overview'] = del_space(movies_metadata, 'overview')
movies_metadata['popularity'] = movies_metadata['popularity'].astype(float).round(2)
#movies_metadata['poster_path']
#movies_metadata['production_companies']
#movies_metadata['production_countries']
movies_metadata['release_date'] = movies_metadata['release_date'].astype('datetime64[s]')
movies_metadata['revenue'] = movies_metadata['revenue'].astype('Int64') 
movies_metadata['runtime'] = movies_metadata['runtime'].astype('Int64') 
movies_metadata['spoken_languages'] = del_space(movies_metadata, 'spoken_languages')
movies_metadata['status'] = del_space(movies_metadata, 'status')
movies_metadata['tagline'] = del_space(movies_metadata, 'tagline')
movies_metadata['title'] = del_space(movies_metadata, 'title') 
#movies_metadata['video']
movies_metadata['vote_average'] = movies_metadata['vote_average'].astype(float).round(1)
movies_metadata['vote_count'] = movies_metadata['vote_count'].astype('Int64') 

## DUPLICATES FIND AND REMOVE
movies_metadata_duplicate_id = movies_metadata[movies_metadata['id'].duplicated(keep=False)]
print('movies_metadata_duplicate_by_id \n', movies_metadata_duplicate_id.loc[:,['id','title','imdb_id','original_language']].sort_values(by=['id', 'title'], ascending=[True, False]))
duplicate_cnt = movies_metadata_duplicate_id['id'].value_counts()
print('duplicate_cnt:\t', duplicate_cnt)

movies_metadata_duplicate_id = movies_metadata_duplicate_id.set_index('id')

print (movies_metadata_duplicate_id.sort_values(by=['id','title','imdb_id','original_language']))

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

remove_duplicates_if_exist(movies_metadata,columns = ['id','title'])

st.write(movies_metadata)

###### 

# Convert the 'genres' column from a string to a list of dictionaries
movies_metadata['genres'] = movies_metadata['genres'].apply(ast.literal_eval)

# Expand the list of genres into separate rows
df_exploded = movies_metadata.explode('genres')

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

for col in movies_metadata.columns:
    print('Column: ', col, '\tCount Null: ', movies_metadata[col].isna().sum())

movies_metadata_fltr_data = movies_metadata[movies_metadata['release_date'] < '1900-01-01']
print('Raws with release_date < "1900-01-01":\n', movies_metadata[movies_metadata['release_date'] < '1900-01-01'])

print('POPULARITY')
movies_metadata['popularity'] = movies_metadata['popularity'].fillna(movies_metadata['popularity'].mean)

print(cnt_null_in_column(movies_metadata,'vote_count'))

## TICKET 16-17 Integration einer Suchfunktion zur gezielten Ausfindung von Filmen.

@st.cache_data
def load_data():
    df = movies_metadata
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
