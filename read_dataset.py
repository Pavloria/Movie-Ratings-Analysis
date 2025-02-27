import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import ast
from matplotlib.backends.backend_pdf import PdfPages


movies = pd.read_csv("movies_metadata.csv", low_memory=False)  # Fix potential DtypeWarning
ratings_small = pd.read_csv("ratings_small.csv")
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")
links = pd.read_csv("links.csv")
links_small = pd.read_csv("links_small.csv")
ratings = pd.read_csv("ratings.csv")

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

#9 Entwicklung einer Benutzerober äche zur dynamischen Filterung und Darstellung der Daten
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load and process genre dataset
@st.cache_data
def load_genre_data():
    genres_df = pd.read_csv("movies_genres.csv")
    return genres_df

# Load and process movies dataset
@st.cache_data
def load_movies():
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)

    # Select necessary columns
    movies = movies[['id', 'title', 'release_date', 'budget', 'vote_average', 'vote_count']]

    # Convert release_date to year format
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['year'] = movies['release_date'].dt.year

    # Ensure numeric values in budget and vote fields
    movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce').fillna(0)
    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(0)
    movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)

    # Convert 'id' to string (to match movieId from ratings)
    movies['id'] = movies['id'].astype(str)

    return movies

# Load and process ratings_small dataset
@st.cache_data
def load_ratings_small():
    ratings_small = pd.read_csv("ratings_small.csv")

    # Convert movieId to string (to match id from movies)
    ratings_small['movieId'] = ratings_small['movieId'].astype(str)

    # Calculate average rating per movie
    ratings_summary = ratings_small.groupby('movieId')['rating'].mean().reset_index()
    ratings_summary.rename(columns={'rating': 'user_rating'}, inplace=True)

    return ratings_summary

# Load data
genres_df = load_genre_data()
movies = load_movies()
ratings_small = load_ratings_small()

# Merge user ratings with movies dynamically
movies = movies.merge(ratings_small, left_on='id', right_on='movieId', how='left').drop(columns=['movieId'])
movies['user_rating'] = movies['user_rating'].fillna(0)  # Fill missing ratings with 0

# Streamlit App
st.title("🎬 Movies Dataset Analysis")

# Sidebar Filters
st.sidebar.header("Filters")

# Select Genre
unique_genres = sorted(genres_df['genre'].dropna().unique())
selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + unique_genres)

# Select Year Range
year_min, year_max = int(movies['year'].min()), int(movies['year'].max())
year_range = st.sidebar.slider("Select Year Range", year_min, year_max, (year_min, year_max))

# Budget Range (Handling Edge Cases)
budget_min, budget_max = int(movies['budget'].min()), int(movies['budget'].max())
budget_range = st.sidebar.slider("Select Budget Range", budget_min, budget_max, (budget_min, budget_max))

# Vote Average Range
vote_min, vote_max = float(movies['vote_average'].min()), float(movies['vote_average'].max())
vote_range = st.sidebar.slider("Select Vote Average Range", vote_min, vote_max, (vote_min, vote_max))

# User Rating Range (from ratings_small.csv)
user_rating_min, user_rating_max = float(movies['user_rating'].min()), float(movies['user_rating'].max())
user_rating_range = st.sidebar.slider("Select User Rating Range", user_rating_min, user_rating_max, (user_rating_min, user_rating_max))

# Apply Filters
filtered_movies = movies[
    (movies['year'].between(year_range[0], year_range[1])) &
    (movies['budget'].between(budget_range[0], budget_range[1])) &
    (movies['vote_average'].between(vote_range[0], vote_range[1])) &
    (movies['user_rating'].between(user_rating_range[0], user_rating_range[1]))
]

# Apply genre filter dynamically
if selected_genre != "All":
    genre_movie_ids = genres_df[genres_df['genre'] == selected_genre]['movie_id'].astype(str)
    filtered_movies = filtered_movies[filtered_movies['id'].isin(genre_movie_ids)]

# Display Filtered Movies
st.write("### Filtered Movies")
st.dataframe(filtered_movies[['title', 'year', 'budget', 'vote_average', 'vote_count', 'user_rating']])

# Visualization Options
st.sidebar.header("Visualizations")

# Histogram: Distribution of Ratings
if st.sidebar.checkbox("Show Rating Distribution"):
    fig = px.histogram(filtered_movies, x="vote_average", nbins=20, title="Rating Distribution")
    st.plotly_chart(fig)

# Scatter Plot: Budget vs Vote Average
if st.sidebar.checkbox("Show Budget vs Vote Average"):
    fig = px.scatter(filtered_movies, x="budget", y="vote_average", color="year",
                     title="Budget vs Vote Average", size="vote_count", hover_data=["title"])
    st.plotly_chart(fig)

# Bar Chart: Top Rated Movies
if st.sidebar.checkbox("Show Top Rated Movies"):
    top_movies = filtered_movies.sort_values(by="vote_average", ascending=False).head(10)
    fig = px.bar(top_movies, x="vote_average", y="title", orientation='h', title="Top 10 Rated Movies")
    st.plotly_chart(fig)

# Scatter Plot: User Rating vs Vote Average
if st.sidebar.checkbox("Show User Ratings vs Vote Average"):
    fig = px.scatter(filtered_movies, x="user_rating", y="vote_average", color="year",
                     title="User Ratings vs Vote Average", size="vote_count", hover_data=["title"])
    st.plotly_chart(fig)

# **NEW FILTER** 📊 Scatter Plot: Budget vs User Rating
if st.sidebar.checkbox("Show Budget vs User Rating"):
    fig = px.scatter(filtered_movies, x="budget", y="user_rating", color="year",
                     title="Budget vs User Rating", size="vote_count", hover_data=["title"])
    st.plotly_chart(fig)

# **NEW FILTER** 📊 Average Rating per Genre (Bar Chart with Red Linear Fit)
if st.sidebar.checkbox("Show Average Rating per Genre"):
    # Compute average rating per genre
    genre_ratings = genres_df.merge(movies[['id', 'vote_average']], left_on='movie_id', right_on='id', how='left')
    avg_ratings_per_genre = genre_ratings.groupby('genre')['vote_average'].mean().reset_index()

    # **Plot Bar Chart**
    fig = px.bar(avg_ratings_per_genre, x="genre", y="vote_average",
                 title="Average Rating per Genre",
                 labels={"vote_average": "Average Rating", "genre": "Genre"},
                 color_discrete_sequence=["blue"])

    # **Compute Linear Regression (Trend Line)**
    x_numeric = np.arange(len(avg_ratings_per_genre))
    y_values = avg_ratings_per_genre["vote_average"]
    slope, intercept = np.polyfit(x_numeric, y_values, 1)
    trend_line = slope * x_numeric + intercept

    # **Add Trend Line to Chart**
    fig.add_scatter(x=avg_ratings_per_genre["genre"], y=trend_line, mode="lines",
                    name="Trend Line", line=dict(color="red", width=2))

    st.plotly_chart(fig)
    
# 6
df = pd.read_csv("ratings.csv")

# Die Spalte "timestamp" (Zeitstempel in Sekunden) in ein Datumsformat umwandeln
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# Das Jahr aus dem Datum extrahieren und in einer neuen Spalte "year" speichern
df["year"] = df["timestamp"].dt.year
df["year"] = df["year"].astype(int)

# Daten filtern: Nur Bewertungen ab dem Jahr 2002 behalten
df = df[df["year"] >= 2000]

# Daten nach Jahren gruppieren und den Durchschnitt der Bewertungen für jedes Jahr berechnen
ratings_by_year = df.groupby("year")["rating"].mean().reset_index()
ratings_by_year["year"] = ratings_by_year["year"].astype(int)



plt.figure(figsize=(10, 10))

# Ein Liniendiagramm erstellen (Jahr auf der X-Achse, durchschnittliche Bewertung auf der Y-Achse)
sns.lineplot(x="year", y="rating", data=ratings_by_year, marker="o")


plt.title("Durchschnittliche Bewertungen über die Jahre")  
plt.xlabel("Jahr")  
plt.ylabel("Durchschnittliche Bewertung")  

plt.xticks(ratings_by_year["year"], rotation=45)

# Gitterlinien aktivieren, um das Diagramm lesbarer zu machen
plt.grid(True)

# Diagramm anzeigen
plt.show()


#8
#  from Reza
### Zusammenhang zwischen Produktionsbudget und durchschnittlicher Bewertung von Filmen
plt.figure(figsize=(12,7))
plt.scatter(movies['budget'], movies['vote_average'], alpha=0.7, color='green')
plt.title('Bewertung vs. Produktionsbudget')
plt.xlabel('Budget')
plt.ylabel('Bewertung')
plt.xscale('log')  # Logarithmische Skalierung für bessere Darstellung
plt.yscale('linear')
plt.show()
#############
### Entwicklung der durchschnittlichen Filmbewertungen über die Jahre




movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['year'] = movies['release_date'].dt.year
avg_rating_per_year = movies.groupby('year')['vote_average'].mean()

# Liniendiagramm erstellen
plt.figure(figsize=(12, 6))
avg_rating_per_year.plot(kind='line', color='darkgreen', marker='*')
plt.title('Durchschnittsbewertung von Filmen pro Jahr')
plt.xlabel('Jahr')
plt.ylabel('Durchschnittsbewertung')
plt.grid(True)
plt.show()

### Die 10 am höchsten bewerteten Filme basierend auf der durchschnittlichen Bewertung
# Gruppiere die Filme nach ihrer durchschnittlichen Bewertung und plotiere sie
avg_ratings = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False)
top_avg_ratings = avg_ratings.head(10)  # Die 10 höchsten bewerteten Filme holen

# Balkendiagramm erstellen
plt.figure(figsize=(10,6))
top_avg_ratings.plot(kind='bar', color='skyblue')
plt.title('Top 10 Filme nach Durchschnittsbewertun')
plt.xlabel('Filmid')
plt.ylabel('Durchschnittliche Bewertung')
plt.xticks(rotation=45)
plt.show()
## Line Chart
#### Entwicklung der durchschnittlichen Film-Bewertungen im Zeitverlauf
# Überprüfen auf nicht-numerische Werte in der 'id'-Spalte
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')  # Ungültige Werte werden in NaN umgewandelt

# Zeilen mit NaN-Werten in der 'id'-Spalte entfernen
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)
# Nun den Merge durchführen
movies_ratings = pd.merge(movies[['id', 'release_date']], ratings[['movieId', 'rating']], left_on='id', right_on='movieId')
movies_ratings['release_year'] = pd.to_datetime(movies_ratings['release_date']).dt.year
avg_ratings_yearly = movies_ratings.groupby('release_year')['rating'].mean()
# Liniendiagramm erstellen
plt.figure(figsize=(10,6))
avg_ratings_yearly.plot(kind='line', color='green', marker='o')
plt.title('Durchschnittliche Film-Bewertungen im Laufe der Zeit')
plt.xlabel('Jahr')
plt.ylabel('Durchschnittliche Bewertung')
plt.grid(True)
plt.show()
# Scatter Plot: 
### To visualize the relationship between budget (if available) and the ratings for movies.
# Angenommen, wir haben Budget- und Bewertungsinformationen im 'movies'-Datensatz
movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')  # Budget in numerisch umwandeln
movies_ratings = pd.merge(movies[['id', 'budget']], ratings[['movieId', 'rating']], left_on='id', right_on='movieId')
# Streudiagramm erstellen
plt.figure(figsize=(10,6))
plt.scatter(movies_ratings['budget'], movies_ratings['rating'], alpha=0.5, color='purple')
plt.title('Streudiagramm: Film-Budget vs. Bewertung')
plt.xlabel('Budget')
plt.ylabel('Bewertung')
plt.show()

# 10 und 11
# Streamlit App-Überschrift
st.title("🎬 Filmdaten-Analyse")

# Benutzerfreundliche Navigation
st.sidebar.header("🔍 Filteroptionen")
min_votes = st.sidebar.slider("Minimale Stimmenanzahl", 0, 500, 50)
min_rating = st.sidebar.slider("Minimale Bewertung", 0.0, 10.0, 5.0)

# Jahrzehnte-Filter
movies_metadata["release_date"] = pd.to_datetime(movies_metadata["release_date"], errors='coerce')
movies_metadata["year"] = movies_metadata["release_date"].dt.year
movies_metadata["decade"] = (movies_metadata["year"] // 10) * 10 

decades = sorted(movies_metadata["decade"].dropna().unique().astype(int))
selected_decade = st.sidebar.selectbox("📅 Wähle ein Jahrzehnt", ["Alle"] + decades)

# Jahr-Filter
if selected_decade != "Alle":
    years = sorted(movies_metadata[movies_metadata["decade"] == selected_decade]["year"].dropna().unique().astype(int))
    selected_year = st.sidebar.selectbox("📆 Wähle ein Jahr", ["Alle"] + years)
else:
    selected_year = "Alle"

# Daten filtern
movies_metadata["vote_average"] = pd.to_numeric(movies_metadata["vote_average"], errors="coerce")
movies_metadata["vote_count"] = pd.to_numeric(movies_metadata["vote_count"], errors="coerce")

filtered_movies = movies_metadata[(movies_metadata["vote_count"] >= min_votes) & (movies_metadata["vote_average"] >= min_rating)]
if selected_decade != "Alle":
    filtered_movies = filtered_movies[filtered_movies["decade"] == selected_decade]
if selected_year != "Alle":
    filtered_movies = filtered_movies[filtered_movies["year"] == selected_year]

# Fügen Sie dem movie_filter "Jahrzehnt" hinzu
filtered_movies["decade"] = (filtered_movies["year"] // 10) * 10

# Anzahl der gefundenen Filme anzeigen
st.write(f"🎥 **Gefundene Filme:** {len(filtered_movies)}")
st.dataframe(filtered_movies[["title", "vote_average", "vote_count", "release_date"]].sort_values(by="vote_average", ascending=False))

# Diagramm: Bewertungshäufigkeit
fig, ax = plt.subplots()
ax.hist(filtered_movies["vote_average"].dropna(), bins=20, color="skyblue", edgecolor="black")
ax.set_xlabel("Bewertung")
ax.set_ylabel("Anzahl der Filme")
ax.set_title("Verteilung der Bewertungen")
st.pyplot(fig)

st.write("✅ **Diese Anwendung ist optimiert für alle Endgeräte und ermöglicht eine interaktive Analyse von Filmdaten.**")


def export_reports():
    
    ratings_by_decade = filtered_movies.groupby("decade")["vote_average"].mean().reset_index()
    ratings_by_decade["decade"] = ratings_by_decade["decade"].astype(int)

    # PDF
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=ratings_by_decade, x="decade", y="vote_average", palette="viridis", ax=ax)
        ax.set_title("Durchschnittliche Bewertungen pro Jahrzehnt")
        ax.set_xlabel("Jahrzehnt")
        ax.set_ylabel("Durchschnittliche Bewertung")
        ax.grid(True)
        pdf.savefig(fig)  # PDF
        plt.close(fig)

    pdf_buffer.seek(0)  
    return pdf_buffer

# Button zum Herunterladen des PDFs
if st.button("📥 PDF herunterladen"):
    pdf_data = export_reports()
    st.download_button(
        label="📄 Analyse-Bericht herunterladen",
        data=pdf_data,
        file_name="analyse_bericht.pdf",
        mime="application/pdf"
    )


