import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import io

# Daten laden
adresse = 'C:\\Users\\hshakademie9\\Desktop\\archive\\'

credits = pd.read_csv(adresse + 'credits.csv') 
keywords = pd.read_csv(adresse + 'keywords.csv') 
links_small = pd.read_csv(adresse + 'links_small.csv') 
links = pd.read_csv(adresse + 'links.csv') 
ratings_small = pd.read_csv(adresse + 'ratings_small.csv') 
ratings = pd.read_csv(adresse + 'ratings.csv')
movies_metadata = pd.read_csv(adresse + 'movies_metadata.csv', low_memory=False)

# Streamlit App-Überschrift
st.title("🎬 Filmdaten-Analyse")

# Benutzerfreundliche Navigation
st.sidebar.header("🔍 Filteroptionen")
min_votes = st.sidebar.slider("Minimale Stimmenanzahl", 0, 500, 50)
min_rating = st.sidebar.slider("Minimale Bewertung", 0.0, 10.0, 5.0)

# Jahrzehnte-Filter
movies_metadata["release_date"] = pd.to_datetime(movies_metadata["release_date"], errors='coerce')
movies_metadata["year"] = movies_metadata["release_date"].dt.year
movies_metadata["decade"] = (movies_metadata["year"] // 10) * 10  # 📌 Десятилетие

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
