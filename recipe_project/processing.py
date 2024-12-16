"""
Authors: Iker Aldasoro
         Jon Lejardi

Date: 16/12/2024
"""

# processing.py

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

import pandas as pd


def preprocessing(df):
    """Transforms the dataframe into strings

    Args:
        df (pandas df): Dataframe to process

    Returns:
        _type_: _description_
    """

    # Handle directions (list of strings) by joining them
    df["directions_pre"] = df["directions"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )
    df["categories_pre"] = df["categories"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )
    df["ingredients_pre"] = df["ingredients"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )
    df["rating_pre"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_pre"] = df["rating_pre"].fillna(df["rating_pre"].mean())
    df["desc_pre"] = df["desc"].fillna("")

    return df


def NTLK_clean(text: str):
    """NLT pipeline using NTLK

    Args:
        text (str): Category of the dataset to apply the NLP.

    Returns:
        str: Tokenized set of words.
    """
    stop_words = set(stopwords.words("english"))
    if not isinstance(text, str):
        return ""

    try:
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove extra whitespace
        text = lemmatizer.lemmatize(text)

        # Basic word tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]

        return " ".join(tokens)
    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return ""
