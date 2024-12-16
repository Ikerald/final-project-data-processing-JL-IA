import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
from nltk.corpus import stopwords

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df = pd.read_json("data/full_format_recipes.json")

#df['categories_cleaned'] = df['categories'].apply(preprocess_text)

#print(df["categories_cleaned"])

print("Categories sample:", df['categories'].head())
print("Rating statistics:", df['rating'].describe())

# Explode the categories list to get one row per category-recipe pair
exploded_df = df.explode('categories')

print(exploded_df)

# Calculate basic statistics for each category
category_stats = exploded_df.groupby('categories').agg({'rating': ['count', 'mean', 'std', 'median']}).round(2)
print(category_stats)

# Flatten column names
category_stats.columns = ['count', 'mean_rating', 'std_rating', 'median_rating']
category_stats = category_stats.reset_index()

print(category_stats)


# Filter for categories with significant number of recipes (e.g., > 100)
top_categories = category_stats[category_stats['count'] > 1200].sort_values('mean_rating', ascending=False).reset_index()

print(top_categories)

#python -m spacy download en_core_web_md
spacy.info('en_core_web_md')

nlp = spacy.load('en_core_web_md')
