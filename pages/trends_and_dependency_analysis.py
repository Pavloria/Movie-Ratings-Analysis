import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast


#Bekommen Daten von csv-Datai, die Information über Rating vom Filmen enthalten
df_rating=pd.read_csv(r"ratings.csv")
#Bekommen Daten mit Filmenmetadaten von csv-Datai(Laufzeit,Budget...).
df_metadata=pd.read_csv(r"movies_metadata.csv")
#Löschen alle Zeile, wo im bestimmte Spallte  NaN order nicht numeric Daten geben

def clean_not_numbers_value(df,subset):
    #Konvertieren nich numeric Daten in einer Spalter in NaN
    df[subset]=pd.to_numeric(df[subset], errors='coerce')
    #Löschen alle NaN Values
    df=df.dropna(subset=subset)
    return df

#Reinigt  Daten für die Analyse 
#Nimmt als Argument den Name einer Zeile in DataFrame mit Metadate, die quantative Values hat
def prepare_data_for_rating_analysys(df_metadata,df_rating,x_value):
    #Ratings,die zum gleigen Film gehören zuerst Gruppieren  und dann mittlere Wert von Rating für jeder Film finden
    rating_y=df_rating.groupby('movieId')['rating'].mean()
    #Indeces, die den Filmen Id entsprechen, in  Integer converieren
    rating_y.index = rating_y.index.astype(int)

    #Löschen alle Zeile, wo im bestimmte Spallte  NaN order nicht numeric Daten geben
    df_metadata=clean_not_numbers_value(df_metadata,x_value)
    #Filmenid im DataFrame in Integer convertieren
    df_metadata['id'] = df_metadata['id'].astype(int)
    #Löschen die Duplikate von Filmen(ID) im DataFrame
    df_metadata = df_metadata.drop_duplicates(subset='id', keep='first')

    #Lassen in DateFrame nur die Zeile, wo id von einem Film entspricht einem Index in der Serie und sorten mittthilfe Id vom Film
    df_metadata=df_metadata[df_metadata['id'].isin(rating_y.index)].sort_values(by='id')
    #Lassen in DateFrame nur die Zeile, wo id von einem Film entspricht einem Index in der Serie sorten mittthilfe Id vom Film
    rating_y=rating_y[rating_y.index.isin(df_metadata['id'])].sort_index()

    #Bekommen eine Serie vom DataFrame und Id von Filmen gelten as Indeces
    values_x = df_metadata.set_index("id")[x_value]

    #Test, ob die Grosse von der Serien gleich ist
    print(rating_y.size)
    print(values_x.size)
    return values_x, rating_y
#values_x,rating_y=prepare_data_for_rating_analysys(df_metadata=df_metadata,df_rating=df_rating,'budget')

#Zeigen Zusammenhang zwischen Bewertunen(rating_y Argument) und Values for X-Axis(values_x Argument)
#x_axis wird X-Axis Label
def show_rating_correlation(x_axis,values_x, rating_y):
    #Name für den x-Axis ändern
    match x_axis:
        case "budget":
            x_axis="Budget"
        case "runtime":
            x_axis="Laufzeit"
    
    
    
    #Ausrechnen Koefizients für die lineale Funktion
    #a,b in ax+b=y
    coefficients = np.polyfit(values_x,rating_y, deg=1)
    #Gibt einen Array mit Y-Values zurück
    py=np.polyval(coefficients,values_x)
    # Setzen Diagrammgrosse(in inches)
    plt.figure(figsize=(10, 6))

    #Zeigen Grid Hintergrund
    plt.grid(True)
    #Streudiagramm erstellen 
    plt.scatter(values_x, rating_y, alpha=0.5)

    # Achsenbeschriftungen setzen
    plt.xlabel(x_axis)
    plt.ylabel("Bewertungen")

    # Titel des Diagramms setzen
    plt.title(f"Zusammenhangs zwischen Bewertung und {x_axis}")

    # Eine lineare Anpassungskurve hinzufügen
    #label wird für die Legende verwendet
    # color="red"' macht die Linie rot
    plt.plot(values_x, py, label='Linear Fit', color='red')
    #Legende anzeigen
    plt.legend()
    #diagramm anzeigen
    st.pyplot(plt)

st.subheader("Analyse des Zusammenhangs zwischen Budget und Bewertung order Laufzeit von Filmen")
    

x_parameter=st.selectbox("Bitte wählen Sie,das zweite Kriterium, das Sie für die Analyse verwenden möchten:", ["budget", "runtime"])

values_x,rating_y=prepare_data_for_rating_analysys(df_metadata, df_rating,x_parameter)
show_rating_correlation(x_parameter,values_x,rating_y)
st.markdown("<br><br>", unsafe_allow_html=True)   
st.subheader("Analyse der durchschnittlichen Filmbewertung pro Genre")


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
df_merged = df_merged.sample(n=2000)  
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

