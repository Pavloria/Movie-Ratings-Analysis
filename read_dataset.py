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

