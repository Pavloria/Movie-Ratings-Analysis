import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# TICKET_1: Read info from csv-file. Edit by tet.sydorenko 24.02.2025
def read_csv_file(file_name):
    return pd.read_csv(file_name,low_memory = False)

adresse = 'D:\\WORK\\INTO_CODE\\AGILE_Projekt\\Projekte\\PRJ_Movie\\satze\\'
adresse = 'C:\\Users\\Tetiana\\OneDrive\\Документы\\New folder\\satze\\'
df_credits = read_csv_file(adresse+'credits.csv') 
df_keywords = read_csv_file(adresse+'keywords.csv') 
df_links_small =read_csv_file(adresse+'links_small.csv') 
df_links = read_csv_file(adresse+'links.csv') 
df_ratings_small = read_csv_file(adresse+'ratings_small.csv') 
df_ratings = read_csv_file(adresse+'ratings.csv')
df_movies_metadata = read_csv_file(adresse+'movies_metadata.csv')

# TICKET_2: Korrekte Behandlung und Standardisierung der Film- und Bewertungsdaten. Edit by tet.sydorenko 26.02.2025
def info_datatypes(df_name):
    print('COLUMNS and DATATYPES:\n')
    print(df_name.info())
  
# Delete spaces
def del_space(df, field_name):
    return df[field_name].str.strip()

# movies_metadata.csv: The main Movies Metadata file
info_datatypes(df_movies_metadata)

### ID!!!!

# adult #
print('Check data in first column:\n', df_movies_metadata['adult'].unique())
fltr_movies_metadata_adult = df_movies_metadata[(df_movies_metadata['adult'] != 'False') & 
                                                (df_movies_metadata['adult'] != 'True')]
#print(fltr_movies_metadata_adult)
df_movies_metadata = df_movies_metadata[(df_movies_metadata['adult'] == 'False') | 
                                        (df_movies_metadata['adult'] == 'True')]
print('Rows contain incorrect data and have been removed from the origin dataset.\nIncorrect raws have been saved in "fltr_movies_metadata_adult"')
#print(df_movies_metadata['adult'].unique())

df_movies_metadata['adult'] = df_movies_metadata['adult'].astype(bool) 
#df_movies_metadata['belongs_to_collection']
df_movies_metadata['budget'] = df_movies_metadata['budget'].astype('Int64') 
#df_movies_metadata['genres']
#df_movies_metadata['homepage']
df_movies_metadata['id'] = df_movies_metadata['id'].astype('int')
df_movies_metadata['imdb_id'] = del_space(df_movies_metadata, 'imdb_id')
df_movies_metadata['original_language'] = del_space(df_movies_metadata, 'original_language')
df_movies_metadata['original_title'] = del_space(df_movies_metadata, 'original_title')
df_movies_metadata['overview'] = del_space(df_movies_metadata, 'overview')
df_movies_metadata['popularity'] = df_movies_metadata['popularity'].astype(float).round(2)
#df_movies_metadata['poster_path']
#df_movies_metadata['production_companies']
#df_movies_metadata['production_countries']
df_movies_metadata['release_date'] = df_movies_metadata['release_date'].astype('datetime64[s]')
df_movies_metadata['revenue'] = df_movies_metadata['revenue'].astype('Int64') 
df_movies_metadata['runtime'] = df_movies_metadata['runtime'].astype('Int64') 
df_movies_metadata['spoken_languages'] = del_space(df_movies_metadata, 'spoken_languages')
df_movies_metadata['status'] = del_space(df_movies_metadata, 'status')
df_movies_metadata['tagline'] = del_space(df_movies_metadata, 'tagline')
df_movies_metadata['title'] = del_space(df_movies_metadata, 'title') 
#df_movies_metadata['video']
df_movies_metadata['vote_average'] = df_movies_metadata['vote_average'].astype(float).round(1)
df_movies_metadata['vote_count'] = df_movies_metadata['vote_count'].astype('Int64') 

## DUPLICATES FIND AND REMOVE
df_duplicate_id = df_movies_metadata[df_movies_metadata['id'].duplicated(keep=False)]
print('df_duplicate_by_id \n', df_duplicate_id.loc[:,['id','title','imdb_id','original_language']].sort_values(by=['id', 'title'], ascending=[True, False]))
duplicate_cnt = df_duplicate_id['id'].value_counts()
print('duplicate_cnt:\t', duplicate_cnt)

df_duplicate_id = df_duplicate_id.set_index('id')

print (df_duplicate_id.sort_values(by=['id','title','imdb_id','original_language']))

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

remove_duplicates_if_exist(df_movies_metadata,columns = ['id','title'])

st.write(df_movies_metadata)

# TICKET_3: Erkennung und Behandlung fehlender Einträge in den Datensätzen. - Обнаружение отсутствующих или некорректных значений:

# Check Null None Nat ... and give Count of Nulls in Columns
def cnt_null_in_column(df, field_name):
    return f'Number of Null-Values in column "{field_name}": {df[field_name].isna().sum()}'

# Change Values with null in specific columns to middle-value
def fill_durchschnitt_null_strings(df,colmn_name): # z.B. colmn_names ='release_date', 'rating'
    df['colmn_name'] = df['colmn_name'].fillna(df['colmn_name'].mean())

# Delete strings with null in specific columns
def del_null_strings(df,colmn_names): # z.B. colmn_names ='release_date', 'rating'
    df.dropna(subset=[colmn_names], inplace=True)
#####

for col in df_movies_metadata.columns:
    print('Column: ', col, '\tCount Null: ', df_movies_metadata[col].isna().sum())

df_fltr_data = df_movies_metadata[df_movies_metadata['release_date'] < '1900-01-01']
print('Raws with release_date < "1900-01-01":\n', df_movies_metadata[df_movies_metadata['release_date'] < '1900-01-01'])
# print(df_fltr_data['release_date'])
# df_invalid_release_date = df_movies_metadata[df_movies_metadata['release_date'].isna()]
# print(cnt_null_in_column(df_movies_metadata,'release_date'))

print('POPULARITY')
# df_movies_metadata['popularity'] = df_movies_metadata['popularity'].fillna(0)
df_movies_metadata['popularity'] = df_movies_metadata['popularity'].fillna(df_movies_metadata['popularity'].mean)

print(cnt_null_in_column(df_movies_metadata,'vote_count'))
#print(df_movies_metadata.head())

# num_nan = df_movies_metadata['release_date'].isna().sum()
# valid_dates = df_movies_metadata['release_date'].dropna()
# num_before_1900 = (valid_dates.dt.year < 1900).sum()
# num_after_1900 = (valid_dates.dt.year >= 1900).sum()

## TICKET 16-17 Integration einer Suchfunktion zur gezielten Ausfindung von Filmen.

# Загрузка данных
@st.cache_data
def load_data():
    df = df_movies_metadata
    df = df[['title', 'genres', 'release_date', 'vote_average', 'vote_count']].dropna()
    return df

df = load_data()

# Функция для поиска фильмов
def search_movies(query, data):
    return data[data['title'].str.contains(query, case=False, na=False)]

# Интерфейс Streamlit
st.title("Movie Ratings and Trends Analysis")

# Field for looking for films by name
search_query_title = st.text_input("Find the film by name:")

# Field for looking for films by realease
search_query_date = st.text_input("Find the film by realease date:")


if search_query_title:
    search_results = search_movies(search_query_title, df)
    st.write(f"### {len(search_results)} of films were found")
    st.dataframe(search_results)

# if search_query_date:
#     search_results = search_movies(search_query_date, df)
#     st.write(f"### {len(search_results)} of films were found")
#     st.dataframe(search_results)

# clean filters
if st.button("🔄 Clean filters"):
    st.experimental_rerun()

