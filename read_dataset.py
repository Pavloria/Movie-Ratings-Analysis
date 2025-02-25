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
