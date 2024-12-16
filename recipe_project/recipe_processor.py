import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns


class RecipeProcessor:
    def __init__(self, json_path):
        """Initialize the RecipeProcessor with the path to the JSON file."""
        try:
            self.df = pd.read_json(json_path)
            self.stop_words = set(stopwords.words("english"))
        except Exception as e:
            print(f"Error initializing RecipeProcessor: {str(e)}")
            raise

    def basic_preprocess(self, text):
        """Simplified text preprocessing function."""
        if not isinstance(text, str):
            return ""

        try:
            # Convert to lowercase
            text = text.lower()

            # Remove special characters and digits
            text = re.sub(r"[^a-zA-Z\s]", "", text)

            # Remove extra whitespace
            text = " ".join(text.split())

            # Basic word tokenization (split by space)
            tokens = text.split()

            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stop_words]

            return " ".join(tokens)
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return ""

    def prepare_data(self):
        """Prepare the dataset for analysis."""
        try:
            # Handle directions (list of strings) by joining them
            self.df["directions_text"] = self.df["directions"].apply(
                lambda x: " ".join(x) if isinstance(x, list) else str(x)
            )

            # Combine processed directions and description
            self.df["text"] = (
                self.df["directions_text"] + " " + self.df["desc"].fillna("")
            )

            # Preprocess text
            print("Starting text preprocessing...")
            self.df["processed_text"] = self.df["text"].apply(self.basic_preprocess)
            print("Text preprocessing completed!")

            # Convert rating to numeric, handling any non-numeric values
            self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce")

            # Remove rows with missing ratings
            self.df = self.df.dropna(subset=["rating"])

            print(f"Total number of recipes: {len(self.df)}")
            print(f"Average rating: {self.df['rating'].mean():.2f}")

            return self.df

        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise

    def create_tfidf_vectors(self, max_features=1000):
        """Create TF-IDF vectors from processed text."""
        try:
            print("Creating TF-IDF vectors...")
            tfidf = TfidfVectorizer(max_features=max_features)
            tfidf_vectors = tfidf.fit_transform(self.df["processed_text"])
            print(f"Created TF-IDF vectors with shape: {tfidf_vectors.shape}")
            return tfidf_vectors
        except Exception as e:
            print(f"Error creating TF-IDF vectors: {str(e)}")
            raise

    def create_word2vec_vectors(self, vector_size=100):
        """Create Word2Vec vectors from processed text."""
        try:
            print("Creating Word2Vec vectors...")
            # Prepare sentences for Word2Vec
            sentences = [text.split() for text in self.df["processed_text"]]

            # Train Word2Vec model
            w2v_model = Word2Vec(
                sentences, vector_size=vector_size, window=5, min_count=1, workers=4
            )

            # Create document vectors by averaging word vectors
            doc_vectors = []
            for sentence in sentences:
                word_vectors = [
                    w2v_model.wv[word] for word in sentence if word in w2v_model.wv
                ]
                if word_vectors:
                    doc_vectors.append(np.mean(word_vectors, axis=0))
                else:
                    doc_vectors.append(np.zeros(vector_size))

            vectors = np.array(doc_vectors)
            print(f"Created Word2Vec vectors with shape: {vectors.shape}")
            return vectors

        except Exception as e:
            print(f"Error creating Word2Vec vectors: {str(e)}")
            raise

    def create_bert_vectors(self, model_name="bert-base-uncased"):
        """Create BERT embeddings from processed text."""
        try:
            print("Creating BERT vectors...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            bert_vectors = []
            batch_size = 32

            # Process in batches to handle memory constraints
            for i in range(0, len(self.df), batch_size):
                batch_texts = (
                    self.df["processed_text"].iloc[i : i + batch_size].tolist()
                )
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move inputs to the same device as model
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use [CLS] token embeddings as document representation
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    bert_vectors.extend(batch_embeddings)

                if (i + batch_size) % 1000 == 0:
                    print(f"Processed {i + batch_size} documents...")

            vectors = np.array(bert_vectors)
            print(f"Created BERT vectors with shape: {vectors.shape}")
            return vectors

        except Exception as e:
            print(f"Error creating BERT vectors: {str(e)}")
            raise

    def analyze_categories(self):
        """Analyze relationship between categories and ratings."""
        try:
            # Explode categories to analyze individual categories
            categories_df = self.df.explode("categories")

            # Calculate average rating per category
            category_ratings = categories_df.groupby("categories")["rating"].agg(
                ["mean", "count"]
            )
            category_ratings = category_ratings[
                category_ratings["count"] > 50
            ]  # Filter for categories with >50 recipes
            category_ratings = category_ratings.sort_values("mean", ascending=False)

            # Plot top 20 categories by average rating
            plt.figure(figsize=(15, 8))
            sns.barplot(
                data=category_ratings.head(20),
                x="mean",
                y=category_ratings.head(20).index,
            )
            plt.title("Average Rating by Category (Top 20)")
            plt.xlabel("Average Rating")
            plt.ylabel("Category")
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
            print(self.df["directions"].iloc[0])
            print("\nDescription (first recipe):")
            print(self.df["desc"].iloc[0])
            print("\nRating (first recipe):")
            print(self.df["rating"].iloc[0])
        except Exception as e:
            print(f"Error showing data sample: {str(e)}")
