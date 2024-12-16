"""
Authors: Iker Aldasoro
         Jon Lejardi

Date: 16/12/2024
"""

import nltk
import pandas as pd

import recipe_project
import recipe_project.auxiliar
import recipe_project.processing
import recipe_project.visualization

def main():

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
    #recipe_project.visualization.visualize(df)

    print(" ")
    print("3. Applying NLT Pipeline")
    df["directions_post"] = df["directions_pre"].apply(recipe_project.processing.NTLK_clean)
    df["categories_post"] = df["categories_pre"].apply(recipe_project.processing.NTLK_clean)
    df["ingredients_post"] = df["ingredients_pre"].apply(recipe_project.processing.NTLK_clean)

    print(" ")
    print("4. Vectorizing")
    





if __name__ == "__main__":
    main()
