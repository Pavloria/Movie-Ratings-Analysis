import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# TICKET_2: Korrekte Behandlung und Standardisierung der Film- und Bewertungsdaten. Edit by tet.sydorenko 25.02.2025
def info_datatypes(df_name):
    print('Origin columns and datatypes:\n')
    print(df_name.info())
  
# Delete spaces
def del_space(df, field_name):
    return df[field_name].str.strip()

# movies_metadata.csv: The main Movies Metadata file
info_datatypes(df_movies_metadata)

print('Check ID uniqu:\n', df_movies_metadata['id'].is_unique)

# adult #
print('Check data in first column:\n', df_movies_metadata['adult'].unique())
fltr_movies_metadata_adult = df_movies_metadata[(df_movies_metadata['adult'] != 'False') & 
                                                (df_movies_metadata['adult'] != 'True')]
#print(fltr_movies_metadata_adult)
df_movies_metadata = df_movies_metadata[(df_movies_metadata['adult'] == 'False') | 
                                        (df_movies_metadata['adult'] == 'True')]
print('Rows contain incorrect data and have been removed from the origin dataset.\nIncorrect raws have been saved in "fltr_movies_metadata_adult"')
#print(df_movies_metadata['adult'].unique())

df_movies_metadata['popularity']  = df_movies_metadata['popularity'].astype(float).round(2)

df_movies_metadata['release_date'] = df_movies_metadata['release_date'].astype('datetime64[s]')

df_movies_metadata['vote_count']  = df_movies_metadata['vote_count'].astype('Int64') 

df_movies_metadata['overview'] = del_space(df_movies_metadata, 'overview')
df_movies_metadata['original_title'] = del_space(df_movies_metadata, 'original_title') 
df_movies_metadata['title'] = del_space(df_movies_metadata, 'title') 
df_movies_metadata['tagline'] = del_space(df_movies_metadata, 'tagline')

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
print(df_movies_metadata.head())

# num_nan = df_movies_metadata['release_date'].isna().sum()
# valid_dates = df_movies_metadata['release_date'].dropna()
# num_before_1900 = (valid_dates.dt.year < 1900).sum()
# num_after_1900 = (valid_dates.dt.year >= 1900).sum()

# categories = ['NaN', '< 1900', '>= 1900']
# counts = [num_nan, num_before_1900, num_after_1900]
# # Pie-Diagramm
# plt.figure(figsize=(6, 6))
# plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
# plt.title('Date in categories')
# plt.show()

# # credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
# info_datatypes(df_credits)
# df_credits.head()

# # keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
# info_datatypes(df_keywords)
# df_keywords.head()

# # links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.
# info_datatypes(df_links_small)
# df_links_small.head()

# # links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
# info_datatypes(df_links)
# df_links['tmdbId']  = float_to_int(df_links, 'tmdbId') 
# df_links.head()

# # ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.
# info_datatypes(df_ratings_small)
# df_ratings_small.head()

# # Too much Data!!! 0 to 26 024 288
# info_datatypes(df_ratings)
# df_ratings.head()