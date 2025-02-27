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

#5 Merge movies with ratings
ratings_summary = ratings.groupby('movieId').agg({'rating': ['mean', 'count', 'sum']})
ratings_summary.columns = ['average_rating', 'rating_count', 'total_rating']
ratings_summary.reset_index(inplace=True)

# Merge with movie titles
movies['movieId'] = pd.to_numeric(movies['id'], errors='coerce')
movies_ratings = movies.merge(ratings_summary, on='movieId', how='left')

# Display
movies_ratings[['title', 'average_rating', 'rating_count', 'total_rating']].head()

#9 Entwicklung einer Benutzerober äche zur dynamischen Filterung und Darstellung der Daten
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 📌 Datensätze laden und verarbeiten
@st.cache_data
def load_genre_data():
    genres_df = pd.read_csv("movies_genres.csv")
    return genres_df

@st.cache_data
def load_movies():
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)

    # Wichtige Spalten auswählen
    movies = movies[['id', 'title', 'release_date', 'budget', 'vote_average', 'vote_count']]

    # Veröffentlichungsdatum in Jahr umwandeln
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['year'] = movies['release_date'].dt.year

    # Zahlenwerte sicherstellen
    movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce').fillna(0)
    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(0)
    movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)

    # ID in String umwandeln (zum Abgleich mit movieId)
    movies['id'] = movies['id'].astype(str)

    return movies

@st.cache_data
def load_ratings_small():
    ratings_small = pd.read_csv("ratings_small.csv")

    # movieId in String umwandeln (zum Abgleich mit id)
    ratings_small['movieId'] = ratings_small['movieId'].astype(str)

    # Durchschnittliche Bewertung pro Film berechnen
    ratings_summary = ratings_small.groupby('movieId')['rating'].mean().reset_index()
    ratings_summary.rename(columns={'rating': 'user_rating'}, inplace=True)

    return ratings_summary

# Daten laden
genres_df = load_genre_data()
movies = load_movies()
ratings_small = load_ratings_small()

# Nutzwertungen mit den Filmen verbinden
movies = movies.merge(ratings_small, left_on='id', right_on='movieId', how='left').drop(columns=['movieId'])
movies['user_rating'] = movies['user_rating'].fillna(0)  # Fehlende Werte mit 0 füllen

# 📌 Streamlit App
st.title("🎬 Filmdaten-Analyse")

# **Seitenleiste für Filter**
st.sidebar.header("🔍 Filteroptionen")

# Genre auswählen
unique_genres = sorted(genres_df['genre'].dropna().unique())
selected_genre = st.sidebar.selectbox("🎭 Genre auswählen", ["Alle"] + unique_genres)

# Jahrgangsbereich auswählen
year_min, year_max = int(movies['year'].min()), int(movies['year'].max())
year_range = st.sidebar.slider("📅 Jahr auswählen", year_min, year_max, (year_min, year_max))

# Budget-Bereich
budget_min, budget_max = int(movies['budget'].min()), int(movies['budget'].max())
budget_range = st.sidebar.slider("💰 Budget-Bereich auswählen", budget_min, budget_max, (budget_min, budget_max))

# Durchschnittliche Bewertung auswählen
vote_min, vote_max = float(movies['vote_average'].min()), float(movies['vote_average'].max())
vote_range = st.sidebar.slider("⭐ Durchschnittsbewertung", vote_min, vote_max, (vote_min, vote_max))

# Nutzerrating auswählen
user_rating_min, user_rating_max = float(movies['user_rating'].min()), float(movies['user_rating'].max())
user_rating_range = st.sidebar.slider("🎟️ Nutzerrating auswählen", user_rating_min, user_rating_max, (user_rating_min, user_rating_max))

# **Filter anwenden**
filtered_movies = movies[
    (movies['year'].between(year_range[0], year_range[1])) &
    (movies['budget'].between(budget_range[0], budget_range[1])) &
    (movies['vote_average'].between(vote_range[0], vote_range[1])) &
    (movies['user_rating'].between(user_rating_range[0], user_rating_range[1]))
]

# Genre-Filter dynamisch anwenden
if selected_genre != "Alle":
    genre_movie_ids = genres_df[genres_df['genre'] == selected_genre]['movie_id'].astype(str)
    filtered_movies = filtered_movies[filtered_movies['id'].isin(genre_movie_ids)]

# **Gefilterte Filme anzeigen**
st.write("### 🎬 Gefilterte Filme")
st.dataframe(filtered_movies[['title', 'year', 'budget', 'vote_average', 'vote_count', 'user_rating']])

# 📊 **Visualisierungsoptionen**
st.sidebar.header("📊 Diagramme")

# **Histogramm: Verteilung der Bewertungen**
if st.sidebar.checkbox("📈 Bewertung-Verteilung anzeigen"):
    fig = px.histogram(filtered_movies, x="vote_average", nbins=20, title="Verteilung der Bewertungen")
    st.plotly_chart(fig)

# **Streudiagramm: Budget vs Durchschnittsbewertung**
if st.sidebar.checkbox("📊 Budget vs Durchschnittsbewertung"):
    fig = px.scatter(filtered_movies, x="budget", y="vote_average", color="year",
                     title="Budget vs Durchschnittsbewertung", size="vote_count", hover_data=["title"])
    st.plotly_chart(fig)

# **Balkendiagramm: Die am besten bewerteten Filme**
if st.sidebar.checkbox("🏆 Top-bewertete Filme anzeigen"):
    top_movies = filtered_movies.sort_values(by="vote_average", ascending=False).head(10)
    fig = px.bar(top_movies, x="vote_average", y="title", orientation='h', title="Top 10 der bestbewerteten Filme")
    st.plotly_chart(fig)

# **Streudiagramm: Nutzerbewertung vs Durchschnittsbewertung**
if st.sidebar.checkbox("🔄 Nutzerbewertung vs Durchschnittsbewertung"):
    fig = px.scatter(filtered_movies, x="user_rating", y="vote_average", color="year",
                     title="Nutzerbewertung vs Durchschnittsbewertung", size="vote_count", hover_data=["title"])
    st.plotly_chart(fig)

# **Streudiagramm: Budget vs Nutzerrating**
if st.sidebar.checkbox("💰 Budget vs Nutzerrating"):
    fig = px.scatter(filtered_movies, x="budget", y="user_rating", color="year",
                     title="Budget vs Nutzerrating", size="vote_count", hover_data=["title"])
    st.plotly_chart(fig)

# **📊 Durchschnittliche Bewertung pro Genre**
if st.sidebar.checkbox("📈 Durchschnittliche Bewertung pro Genre anzeigen"):
    genre_ratings = genres_df.merge(movies[['id', 'vote_average']], left_on='movie_id', right_on='id', how='left')
    avg_ratings_per_genre = genre_ratings.groupby('genre')['vote_average'].mean().reset_index()

    fig = px.bar(avg_ratings_per_genre, x="genre", y="vote_average",
                 title="Durchschnittliche Bewertung pro Genre",
                 labels={"vote_average": "Durchschnittliche Bewertung", "genre": "Genre"},
                 color_discrete_sequence=["blue"])

    # **Trendlinie berechnen**
    x_numeric = np.arange(len(avg_ratings_per_genre))
    y_values = avg_ratings_per_genre["vote_average"]
    slope, intercept = np.polyfit(x_numeric, y_values, 1)
    trend_line = slope * x_numeric + intercept

    fig.add_scatter(x=avg_ratings_per_genre["genre"], y=trend_line, mode="lines",
                    name="Trendlinie", line=dict(color="red", width=2))

    st.plotly_chart(fig)

# **📥 PDF-Download Funktion**
import io
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

def export_reports(filtered_movies, selected_genre):
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_movies, x="budget", y="vote_average", hue="year", size="vote_count", ax=ax)
        ax.set_title("Budget vs Durchschnittliche Bewertung")
        pdf.savefig(fig)
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer

if st.button("📥 PDF-Bericht herunterladen"):
    pdf_data = export_reports(filtered_movies, selected_genre)
    st.download_button(
        label="📄 Analysebericht herunterladen",
        data=pdf_data,
        file_name="analyse_bericht.pdf",
        mime="application/pdf"
    )

