import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re

def download_nltk_data():
    """Download required NLTK data packages"""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

class RecipeProcessor:
    def __init__(self, json_path):
        """Initialize the RecipeProcessor with the path to the JSON file."""
        try:
            self.df = pd.read_json(json_path)
            download_nltk_data()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error initializing RecipeProcessor: {str(e)}")
            raise
        
    def basic_preprocess(self, text):
        """Simplified text preprocessing function."""
        if not isinstance(text, str):
            return ''
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Basic word tokenization (split by space)
            tokens = text.split()
            
            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return ''

    def prepare_data(self):
        """Prepare the dataset for analysis."""
        try:
            # Handle directions (list of strings) by joining them
            self.df['directions_text'] = self.df['directions'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            )
            
            # Combine processed directions and description
            self.df['text'] = self.df['directions_text'] + ' ' + self.df['desc'].fillna('')
            
            # Preprocess text using the simplified function
            print("Starting text preprocessing...")
            self.df['processed_text'] = self.df['text'].apply(self.basic_preprocess)
            print("Text preprocessing completed!")
            
            # Convert rating to numeric, handling any non-numeric values
            self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
            
            # Remove rows with missing ratings
            self.df = self.df.dropna(subset=['rating'])
            
            # Print some basic statistics
            print(f"Total number of recipes: {len(self.df)}")
            print(f"Average rating: {self.df['rating'].mean():.2f}")
            
            # Print sample of processed text to verify
            print("\nSample of processed text:")
            print(self.df['processed_text'].iloc[0][:200] + "...")
            
            return self.df
        
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise

    def analyze_categories(self):
        """Analyze relationship between categories and ratings."""
        try:
            # Explode categories to analyze individual categories
            categories_df = self.df.explode('categories')
            
            # Calculate average rating per category
            category_ratings = categories_df.groupby('categories')['rating'].agg(['mean', 'count'])
            category_ratings = category_ratings[category_ratings['count'] > 50]  # Filter for categories with >50 recipes
            category_ratings = category_ratings.sort_values('mean', ascending=False)
            
            # Plot top 20 categories by average rating
            plt.figure(figsize=(15, 8))
            sns.barplot(data=category_ratings.head(20), x='mean', y=category_ratings.head(20).index)
            plt.title('Average Rating by Category (Top 20)')
            plt.xlabel('Average Rating')
            plt.ylabel('Category')
            plt.tight_layout()
            plt.show()
            
            return category_ratings
            
        except Exception as e:
            print(f"Error in analyze_categories: {str(e)}")
            return None

    def show_data_sample(self):
        """Display a sample of the data to verify structure"""
        try:
            print("\nSample of raw data:")
            print("\nDirections (first recipe):")
            print(self.df['directions'].iloc[0])
            print("\nDescription (first recipe):")
            print(self.df['desc'].iloc[0])
            print("\nRating (first recipe):")
            print(self.df['rating'].iloc[0])
        except Exception as e:
            print(f"Error showing data sample: {str(e)}")

# Example usage with error handling
if __name__ == "__main__":
    try:
        # Initialize processor
        print("Initializing RecipeProcessor...")
        processor = RecipeProcessor("full_format_recipes.json")
        
        # Show sample of raw data
        print("\nShowing sample of raw data to verify structure...")
        processor.show_data_sample()
        
        # Prepare data
        print("\nPreparing data...")
        df = processor.prepare_data()
        
        # Analyze categories
        print("\nAnalyzing categories...")
        category_analysis = processor.analyze_categories()
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")