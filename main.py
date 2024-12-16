"""
Authors: Iker Aldasoro
         Jon Lejardi

Date: 16/12/2024
"""

# main.py

import nltk
import pandas as pd
import numpy as np

import recipe_project
import recipe_project.auxiliar
import recipe_project.processing
import recipe_project.vectorization
import recipe_project.visualization


def main():
    """Main function of the proyect. Handles order execution of the package."""
    # Download required NLTK data
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("wordnet")

    df = pd.read_json("./data/full_format_recipes.json")

    print("Checking cuda device")
    device = recipe_project.auxiliar.get_cuda()

    print(" ")
    print("1. Preprocessing of the variable")
    df = recipe_project.processing.preprocessing(df)

    print(" ")
    print("2. Analyzing dataframe")
    recipe_project.visualization.visualize(df)

    print(" ")
    print("3. Applying NLT Pipeline")
    df["directions_post"] = df["directions_pre"].apply(
        recipe_project.processing.NTLK_clean
    )
    df["categories_post"] = df["categories_pre"].apply(
        recipe_project.processing.NTLK_clean
    )
    df["ingredients_post"] = df["ingredients_pre"].apply(
        recipe_project.processing.NTLK_clean
    )

    print(df["directions_pre"].head())
    print("")
    print(df["directions_post"].head())

    print(" ")
    print("4. Vectorizing")
    vectors_bert = recipe_project.vectorization.bert_vectorization(device, df)
    vectors_TF = recipe_project.vectorization.TF_vectorization(df)
    vectors_W2V = recipe_project.vectorization.Word2Vec_vectorization(df)
    # To reduce computational power
    vectors_TF = vectors_TF.astype(np.float32)

    print("")
    print("5. Neural Networks and Random Forest")
    recipe_project.vectorization.NeuralNetwork(vectors_bert, df)
    recipe_project.vectorization.NeuralNetwork(vectors_TF, df)
    recipe_project.vectorization.NeuralNetwork(vectors_W2V, df)

    print("")
    print("6. Hugging Face model - RoBERTa")
    vectors_roberta = recipe_project.vectorization.roberta_vectorization(device, df)
    recipe_project.vectorization.NeuralNetwork(vectors_roberta, df)


if __name__ == "__main__":
    main()
