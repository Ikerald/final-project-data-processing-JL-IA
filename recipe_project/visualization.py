"""
Authors: Iker Aldasoro
         Jon Lejardi

Date: 16/12/2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def visualize(df):
    """Visualizes the obtained graphs

    Args:
        df (pandas df): Dataframe to examine
    """
    # Get the Top 10 most common categories
    category_counts = df["categories"].explode().value_counts()[:15]
    top_categories = category_counts.index

    df["categories"] = df["categories"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    filtered_data = df[
        df["categories"].apply(lambda x: any(cat in top_categories for cat in x))
    ]

    # Create a category-rating relationship
    category_rating = filtered_data.explode("categories")
    category_rating = category_rating[
        category_rating["categories"].isin(top_categories)
    ]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="categories", y="rating", data=category_rating)
    plt.xticks(rotation=45)
    plt.title("Average Ratings per Category")
    plt.xlabel("Category")
    plt.ylabel("Rating")
    plt.show()

    numerical_cols = ["rating", "fat", "protein", "calories", "sodium"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation between Numerical Features")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="rating_pre", bins=20)
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()

    # Add boxplot to see outliers
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df["rating_pre"])
    plt.title("Rating Distribution and Outliers")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.scatterplot(data=df, x="fat", y="rating", ax=axes[0, 0])
    sns.scatterplot(data=df, x="protein", y="rating", ax=axes[0, 1])
    sns.scatterplot(data=df, x="calories", y="rating", ax=axes[1, 0])
    sns.scatterplot(data=df, x="sodium", y="rating", ax=axes[1, 1])
    plt.tight_layout()
    plt.show()

    df["directions_length"] = df["directions"].str.len()
    df["desc_length"] = df["desc"].str.len()
    df["title_length"] = df["title"].str.len()

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    sns.boxplot(y=df["directions_length"])
    plt.title("Directions Length")
    plt.subplot(132)
    sns.boxplot(y=df["desc_length"])
    plt.title("Description Length")
    plt.subplot(133)
    sns.boxplot(y=df["title_length"])
    plt.title("Title Length")
    plt.tight_layout()
    plt.show()

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="year", y="rating", data=df)
    plt.title("Rating Distribution by Year")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    df["categories"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Recipe Categories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
