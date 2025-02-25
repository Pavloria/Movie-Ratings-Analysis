import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  


df = pd.read_csv("C:\\Users\\hshakademie9\\Desktop\\Projekt1\\archive\\ratings.csv")

# Die Spalte "timestamp" (Zeitstempel in Sekunden) in ein Datumsformat umwandeln
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# Das Jahr aus dem Datum extrahieren und in einer neuen Spalte "year" speichern
df["year"] = df["timestamp"].dt.year

# Daten filtern: Nur Bewertungen ab dem Jahr 2002 behalten
df = df[df["year"] >= 2002]

# Daten nach Jahren gruppieren und den Durchschnitt der Bewertungen für jedes Jahr berechnen
ratings_by_year = df.groupby("year")["rating"].mean().reset_index()


plt.figure(figsize=(10, 10))

# Ein Liniendiagramm erstellen (Jahr auf der X-Achse, durchschnittliche Bewertung auf der Y-Achse)
sns.lineplot(x="year", y="rating", data=ratings_by_year, marker="o")


plt.title("Durchschnittliche Bewertungen über die Jahre")  
plt.xlabel("Jahr")  
plt.ylabel("Durchschnittliche Bewertung")  

# Gitterlinien aktivieren, um das Diagramm lesbarer zu machen
plt.grid(True)

# Diagramm anzeigen
plt.show()
